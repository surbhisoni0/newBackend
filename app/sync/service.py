"""Keeps the app in step with a configured result source and drives the predict -> confirm -> evaluate -> predict loop.

Per new round (strictly in round order):  store the confirmed result -> score the saved prediction for that round ->
create the prediction for the next round from data that existed before it. All of that is done by PredictorService.
This module only decides WHICH rounds may be ingested, and reports honest status.
"""
import asyncio
import logging
import threading
import time
from typing import Callable, Optional
from app.sync.sources import RoundRecord
from app.timing import next_period_id

log = logging.getLogger("wingo.sync")


def plan_ingest(records, anchor: Optional[str], has_period: Callable[[str], bool], allow_gaps: bool = False) -> dict:
    """Decide what to ingest. `anchor` = round id of the newest stored round.
    Duplicates and rounds at/below the anchor are ignored; ingestion stops at the first missing round unless allow_gaps."""
    uniq, dup = {}, 0
    for r in records:
        if r.period_id in uniq:
            dup += 1
        else:
            uniq[r.period_id] = r
    fresh, old = [], 0
    for r in sorted(uniq.values(), key=lambda r: int(r.period_id)):
        if has_period(r.period_id):
            dup += 1
        elif anchor and int(r.period_id) <= int(anchor):
            old += 1
        else:
            fresh.append(r)
    ingest, gap = [], None
    expected = int(anchor) + 1 if anchor else None
    for r in fresh:
        p = int(r.period_id)
        if expected is not None and p != expected and not allow_gaps:
            gap = {"expected": str(expected), "first_available": r.period_id}
            break
        ingest.append(r)
        expected = p + 1
    return {"ingest": ingest, "duplicates": dup, "ignored_old": old, "gap": gap}


def _iso(ts: Optional[float]) -> Optional[str]:
    return None if ts is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class SyncService:
    def __init__(self, predictor, source, enabled: bool = False, interval: float = 30, stale_after: float = 300,
                 allow_gaps: bool = False, start_after: str = "", fast_interval: float = 5.0):
        self.predictor, self.source, self.enabled = predictor, source, enabled
        self.interval, self.stale_after = max(10.0, float(interval)), stale_after
        self.allow_gaps, self.start_after = allow_gaps, (start_after or "").strip()
        self.fast_interval = max(3.0, float(fast_interval))
        self._lock = threading.Lock()
        self.s = {"state": "disabled" if not enabled else "starting", "last_attempt": None, "last_success": None, "last_error": None,
                  "consecutive_failures": 0, "ingested_total": 0, "last_ingested": 0, "duplicates_seen": 0, "ignored_old": 0,
                  "invalid_items": 0, "gap": None}

    def _fail(self, state: str, msg: str, count_failure: bool = True):
        self.s.update(state=state, last_error=msg)
        if count_failure:
            self.s["consecutive_failures"] += 1
        log.warning("sync %s: %s", state, msg)

    def run_once(self) -> dict:
        with self._lock:
            self.s["last_attempt"] = time.time()
            if not self.source.configured:
                self.s.update(state="not_configured", last_error="No authorized data source is configured (see SYNC_* in backend/.env).")
                return self.status()
            try:
                records = self.source.fetch()
            except Exception as e:                                           # noqa: BLE001
                self._fail("error", str(e)[:300])
                return self.status()
            repo = self.predictor.repo
            with self.predictor.lock:
                last = repo.last_result()
                anchor = last["period_id"] if last else None
                if last and not anchor:
                    anchor = self.start_after or None
                    if not anchor:
                        self._fail("needs_anchor", "Stored history has no round IDs, so the sequence cannot be verified. Set "
                                   "SYNC_START_AFTER_PERIOD to the round ID of your newest stored round.", count_failure=False)
                        self.s["last_success"] = time.time()
                        return self.status()
                plan = plan_ingest(records, anchor, repo.has_period, self.allow_gaps)
                rows = [{"number": r.number, "period_id": r.period_id, "timestamp": r.timestamp} for r in plan["ingest"]]
                try:
                    if len(rows) == 1:
                        self.predictor.add_result(**rows[0])
                    elif rows:
                        self.predictor.import_rows(rows)
                except Exception as e:                                       # noqa: BLE001
                    self._fail("rejected", f"validation failed, nothing stored: {e}"[:300])
                    return self.status()
            self.s.update(last_success=time.time(), consecutive_failures=0, last_ingested=len(rows), gap=plan["gap"],
                          invalid_items=getattr(self.source, "last_invalid", 0), state="ok", last_error=None)
            self.s["ingested_total"] += len(rows)
            self.s["duplicates_seen"] += plan["duplicates"]
            self.s["ignored_old"] += plan["ignored_old"]
            if plan["gap"]:
                self.s.update(state="gap_blocked", last_error=f"Missing rounds: expected {plan['gap']['expected']} but the source's "
                              f"next round is {plan['gap']['first_available']}. Nothing after the gap was stored.")
            return self.status()

    def status(self) -> dict:
        s = dict(self.s)
        age = None if not s["last_success"] else max(0, int(time.time() - s["last_success"]))
        if s["state"] == "ok" and age is not None and age > self.stale_after:
            s["state"] = "stale"
        last = self.predictor.repo.last_result()
        nxt = None
        if last:
            nxt = next_period_id(last.get("period_id")) or f"#{last['seq'] + 1}"
        return {**s, "enabled": self.enabled, "source": self.source.name, "interval_seconds": self.interval,
                "last_attempt": _iso(s["last_attempt"]), "last_success": _iso(s["last_success"]), "seconds_since_success": age,
                "latest_confirmed": last, "expected_next_round": nxt}

    def next_delay(self) -> float:
        """Normal interval, exponential back-off after failures. While a round has just ended and its result is not in yet,
        look a little more often (never faster than fast_interval, never while failing) so the result is confirmed promptly."""
        failures = self.s["consecutive_failures"]
        delay = min(self.interval * (2 ** min(failures, 5)), 600)
        if failures == 0:
            try:
                if self.predictor.clock()["countdown"]["state"] == "awaiting_result":
                    delay = min(delay, self.fast_interval)
            except Exception:                                                  # noqa: BLE001 - timing must never break the loop
                log.exception("clock check failed")
        return delay

    async def run_forever(self):
        log.info("live sync started (every %ss)", self.interval)
        while True:
            try:
                await asyncio.to_thread(self.run_once)
            except asyncio.CancelledError:
                raise
            except Exception:                                                  # noqa: BLE001
                log.exception("sync loop error")
            await asyncio.sleep(self.next_delay())
