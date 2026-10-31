"""Serves /api/clock from a small DEDICATED thread pool, with a last-good fallback.

Why: every sync (`def`) route runs in the shared worker pool. Heavy endpoints (model performance, stats, prediction/next waiting on the
predictor lock) can occupy all of it, and then even the cheap clock request waits in the queue until the browser's timeout fires - the
page then reports "Disconnected from the server". The clock is the connection heartbeat of the UI, so it must never queue behind them.
Semantics are unchanged: every request still computes a fresh clock. Only when that computation fails or is too slow is the last good
snapshot returned (re-stamped with the current server time and marked stale=true) instead of an error.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.timing import iso

log = logging.getLogger("wingo.clock")
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="clock")


class ClockServer:
    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self._last = {}                                             # mode -> (payload, server time when it was computed)

    async def get(self, mode: str, service) -> dict:
        loop = asyncio.get_running_loop()
        try:
            data = await asyncio.wait_for(loop.run_in_executor(_POOL, service.clock), self.timeout)
        except Exception as e:                                      # noqa: BLE001 - includes TimeoutError and database errors
            last = self._last.get(mode)
            if last is None:
                raise
            log.warning("clock compute failed for %s (%s: %s); serving last good snapshot", mode, type(e).__name__, e)
            return self._restamp(last, service)
        self._last[mode] = (data, service.now_fn())
        return data

    @staticmethod
    def _restamp(last, service) -> dict:
        data, then = last
        now = service.now_fn()
        out = dict(data)
        out["server_time"], out["server_epoch_ms"] = iso(now), int(now * 1000)
        cd = dict(out.get("countdown") or {})
        if cd.get("seconds_remaining") is not None:
            cd["seconds_remaining"] = max(0.0, cd["seconds_remaining"] - (now - then))
        out["countdown"] = cd
        out["stale"], out["stale_age_seconds"] = True, round(now - then, 1)
        return out
