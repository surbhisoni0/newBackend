"""Orchestrates: validate -> store -> resolve previous prediction -> update live accuracy -> predict next.

No betting/deposit/withdrawal/auto-bet functionality exists anywhere in this project; this service only
records data, produces statistical candidates and measures how they perform.
"""
import threading
import time
from typing import Callable, Optional
import numpy as np
from app.backtesting.walk_forward import run_walk_forward
from app.database.base import Repository
from app.prediction.engine import PredictionEngine
from app.prediction.signal import evaluate_signal
from app.services import analysis
from app.services.colour_analysis import colour_analysis
from app.services.size_analysis import size_analysis
from app.services.stats_report import accuracy_report, daily_summary, hot_cold, streaks
from app.services.data_service import clean_rows, parse_csv, parse_paste
from app.services.errors import ServiceError, ValidationFailure
from app.timing import ClockConfig, advance_period_id, assumed_countdown, countdown, iso, next_period_id, verify_schedule
from app.utils.constants import BASELINES, METRICS, color_group_of, size_of, COLOR_NAMES
from app.utils.stats import rate_stats

MAX_CLOCK_ROWS = 40
DISCLAIMER = ("Model-generated statistical candidates, not predictions of a guaranteed outcome. Walk-forward validation on "
              "historical data has shown no reliable advantage over random guessing. Nothing here places bets.")


class PredictorService:
    def __init__(self, repo: Repository, engine: PredictionEngine, min_history: int = 100, min_validation_n: int = 300,
                 clock_cfg: Optional[ClockConfig] = None, now_fn: Callable[[], float] = time.time):
        self.repo, self.engine = repo, engine
        self.clock_cfg, self.now_fn = clock_cfg or ClockConfig(), now_fn
        self.min_history, self.min_validation_n = min_history, min_validation_n
        self.lock = threading.RLock()
        self.backtest_state = {"status": "idle", "progress": 0, "total": 0, "error": None}
        self.live_source = None

    # ------------------------------------------------------------ helpers
    def _numbers(self) -> np.ndarray:
        return np.array(self.repo.get_numbers(), dtype=int)

    def live_metrics(self) -> dict:
        outs = [o["outcome"] for o in self.repo.all_outcomes()]
        keys = {"top1": "top1_hit", "top3": "top3_hit", "bs": "bs_hit", "color": "color_hit"}
        return {m: rate_stats([o[keys[m]] for o in outs], BASELINES[m]) for m in METRICS}

    def _validation(self) -> Optional[dict]:
        bt = self.repo.latest_backtest()
        if not bt:
            return None
        s = bt["summary"]
        return {"metrics": s["consensus"], "K": s["K"], "data_n": s["data_n"], "run_at": s["created_at"]}

    def current_signal(self) -> dict:
        v = self._validation()
        sig = evaluate_signal(v, self.live_metrics(), min_n=self.min_validation_n)
        sig["validated_on"] = None if not v else {"data_n": v["data_n"], "run_at": v["run_at"]}
        return sig

    # ------------------------------------------------------------ data flow
    def _store(self, items: list) -> list:
        last = self.repo.last_result()
        cleaned = clean_rows(items, last["period_id"] if last else None, self.repo.has_period)
        return self.repo.add_results(cleaned)

    def _resolve(self, row: dict) -> Optional[dict]:
        pred = self.repo.get_prediction_by_target(row["seq"])
        if not pred or pred["status"] != "pending":
            self.repo.expire_pending_except(row["seq"] + 1)
            return None
        a = row["number"]
        outcome = {"actual": a, "top1_hit": pred["top1"] == a, "top3_hit": a in pred["top3"],
                   "bs_hit": pred["size"] == size_of(a), "color_hit": pred["color"] == COLOR_NAMES[color_group_of(a)]}
        self.repo.resolve_prediction(pred["id"], outcome)          # resolve first ...
        self.repo.expire_pending_except(row["seq"] + 1)            # ... then void anything stale
        live = self.live_metrics()
        self.repo.save_snapshot({"n": live["top1"]["n"], **{k: live[k]["rate"] for k in METRICS}})
        return {**pred, "status": "resolved", "outcome": outcome}

    def _next_prediction(self) -> Optional[dict]:
        last = self.repo.last_result()
        if not last or last["seq"] < self.min_history:
            return None
        target = last["seq"] + 1
        existing = self.repo.get_prediction_by_target(target)
        if existing and existing["status"] == "pending":
            return existing
        p = self.engine.predict(self._numbers())
        lp = last["period_id"]
        pred = {**p, "target_seq": target, "target_period": next_period_id(lp),
                "signal": self.current_signal()}
        self.repo.save_prediction(pred)
        return self.repo.get_prediction_by_target(target)

    def add_result(self, number, period_id=None, timestamp=None) -> dict:
        with self.lock:
            row = self._store([{"number": number, "period_id": period_id, "timestamp": timestamp}])[0]
            resolved = self._resolve(row)
            nxt = self._next_prediction()
            return {"result": row, "resolved_prediction": resolved, "next_prediction": self._public(nxt),
                    "live_performance": self.live_metrics()}

    def import_rows(self, rows: list) -> dict:
        if not rows:
            raise ValidationFailure("nothing to import: no rows provided")
        with self.lock:
            stored = self._store(rows)
            if not stored:
                raise ValidationFailure("nothing to import: no valid rows provided")
            self._resolve(stored[0])                       # a pending prediction can only target the first new round
            self.repo.expire_pending_except(stored[-1]["seq"] + 1)
            nxt = self._next_prediction()
            return {"imported": len(stored), "total": self.repo.last_result()["seq"], "next_prediction": self._public(nxt)}

    def import_csv(self, text: str, newest_first: bool = False) -> dict:
        return self.import_rows(parse_csv(text, newest_first))

    def import_paste(self, text: str, newest_first: bool = False) -> dict:
        return self.import_rows(parse_paste(text, newest_first))

    # ------------------------------------------------------------ reads
    @staticmethod
    def _public(p: Optional[dict]) -> Optional[dict]:
        if p is None:
            return None
        tp = p.get("target_period")
        return {**p, "target_label": tp or f"#{p['target_seq']}", "target_id_kind": "period_id" if tp else "sequence",
                "disclaimer": DISCLAIMER}

    def clock(self, now: Optional[float] = None) -> dict:
        """Server-authoritative timing. Cheap (no model work): safe to poll. The countdown exists ONLY if the schedule is
        verified from stored timestamps; otherwise `countdown.state` is 'unverified' and says why."""
        now = self.now_fn() if now is None else now
        rows, total = self.repo.list_results(MAX_CLOCK_ROWS, 0)
        base = {"server_time": iso(now), "server_epoch_ms": int(now * 1000), "total_results": total}
        if not rows:
            cd = {"state": "no_data", "round_ends_at": None, "seconds_remaining": None,
                  "overdue_seconds": None, "rounds_ended_since_confirmed": None, "reason": None}
            if self.clock_cfg.assumed:
                cd = assumed_countdown(now, self.clock_cfg); cd["round_ends_at"] = iso(cd["round_ends_at"])
            cd = self._live_round(now, {}) or cd
            return {**base, "schedule": {"verified": False, "reason": "No confirmed results stored yet."}, "latest_confirmed": None,
                    "next_round": None, "countdown": cd}
        latest = rows[0]
        sched = verify_schedule(rows, self.clock_cfg)
        cd = countdown(sched, now, self.clock_cfg)
        if self.clock_cfg.assumed and cd["state"] == "unverified":          # opt-in fallback; a verified schedule always wins
            cd = {**assumed_countdown(now, self.clock_cfg), "reason": (sched.get("reason") or "") + " Showing an ASSUMED schedule instead."}
        cd["round_ends_at"] = iso(cd["round_ends_at"])
        live = self._live_round(now, latest)
        if live:
            cd = live
        nxt_id = next_period_id(latest.get("period_id"))
        sched_public = {k: v for k, v in sched.items() if k not in ("anchor_epoch",)}
        sched_public["anchor_time"] = iso(sched["anchor_epoch"])
        sched_public["late_grace_seconds"] = self.clock_cfg.late_grace_seconds
        return {**base, "schedule": sched_public,
                "latest_confirmed": {**latest, "colour": COLOR_NAMES[color_group_of(latest["number"])], "size": size_of(latest["number"])},
                "next_round": {"period_id": nxt_id, "seq": latest["seq"] + 1, "label": nxt_id or f"#{latest['seq'] + 1}",
                               "id_kind": "period_id" if nxt_id else "sequence"},
                "countdown": cd}

    def _live_round(self, now: float, latest: dict) -> Optional[dict]:
        """Current round taken from the live feed's OPEN round (id + end time), corrected by the feed's clock offset.
        Between polls the round is rolled forward on the feed's own fixed schedule; stale feed -> None (fall back)."""
        src = self.live_source
        op = getattr(src, "open_round", None) if src else None
        P = float(self.clock_cfg.period_seconds)
        if not op or not op.get("end") or not src.fetched_at or now - src.fetched_at > 3 * P + 30:
            return None
        t = now + (src.clock_offset or 0)
        k = 0 if t < op["end"] else int((t - op["end"]) // P) + 1
        cur, ends = advance_period_id(op["period_id"], k), op["end"] + k * P
        lp = latest.get("period_id")
        pending = []
        p = next_period_id(lp)
        while p and cur and int(p) < int(cur) and len(pending) < 10:
            pending.append(p); p = next_period_id(p)
        return {"state": "open", "source": "feed", "id_source": "feed" if k == 0 else "feed+schedule",
                "current_period": cur, "next_period": next_period_id(cur), "round_ends_at": iso(ends - (src.clock_offset or 0)),
                "seconds_remaining": max(0.0, ends - t), "period_seconds": int(P), "overdue_seconds": None,
                "rounds_ended_since_confirmed": len(pending), "awaiting_results": pending,
                "upstream_clock_offset_seconds": src.clock_offset, "feed_age_seconds": round(now - src.fetched_at, 1), "reason": None}

    def next_prediction(self) -> dict:
        with self.lock:
            p = self._next_prediction()
        last = self.repo.last_result()
        timing = self.clock()
        if p is None:
            return {"available": False, "reason": f"Need at least {self.min_history} results (have {last['seq'] if last else 0}).",
                    "timing": timing, "disclaimer": DISCLAIMER}
        state = timing["countdown"]["state"]
        # The estimate is bound to ONE round and is never regenerated or swapped. If that round has already ended while its
        # result has not been confirmed yet, say so instead of presenting it as an estimate for a round still to come.
        est = {"open": "open", "awaiting_result": "round_ended", "late": "round_ended"}.get(state, "timing_unverified")
        return {"available": True, "latest_result": last, "estimate_state": est, "timing": timing, **self._public(p)}

    def results(self, limit=50, offset=0) -> dict:
        rows, total = self.repo.list_results(limit, offset)
        return {"total": total, "results": rows}

    def history(self, limit=50, offset=0) -> dict:
        rows, total = self.repo.list_predictions(limit, offset)
        return {"total": total, "predictions": [self._history_row(r) for r in rows]}

    @staticmethod
    def _history_row(r: dict) -> dict:
        o = r.get("outcome")
        result = "PENDING" if not o else ("HIT" if o["bs_hit"] else "MISS")          # Big/Small is the target
        return {"target_seq": r["target_seq"], "round": r["target_period"] or f"#{r['target_seq']}", "created_at": r["created_at"],
                "prediction": r["top1"], "top3": r["top3"], "size": r["size"], "color": r["color"],
                "signal": (r.get("signal") or {}).get("label"), "actual": o["actual"] if o else None,
                "top3_hit": o["top3_hit"] if o else None, "bs_hit": o["bs_hit"] if o else None,
                "color_hit": o["color_hit"] if o else None, "result": result}

    def data_stats(self) -> dict:
        nums = self._numbers()
        out = analysis.analyze(nums, self.repo.get_meta())
        if out.get("counts"):
            out["colour"] = colour_analysis(nums)
            out["size"] = size_analysis(nums)
        return out

    def performance(self) -> dict:
        outs = self.repo.all_outcomes()
        live = self.live_metrics()
        keys = {"top1": "top1_hit", "top3": "top3_hit", "bs": "bs_hit", "color": "color_hit"}
        series = {m: [] for m in METRICS}
        for m in METRICS:
            cum = np.cumsum([float(o["outcome"][keys[m]]) for o in outs]) if outs else np.array([])
            step = max(1, len(cum) // 200)
            series[m] = [{"n": i + 1, "rate": float(cum[i] / (i + 1))} for i in range(0, len(cum), step)]
        bt = self.repo.latest_backtest()
        return {"disclaimer": DISCLAIMER, "baselines": BASELINES, "signal": self.current_signal(),
                "live": {"total_predictions": live["top1"]["n"], "metrics": live, "series": series,
                         "note": "Live = predictions made before the actual result arrived. Needs 100+ to mean anything."},
                "backtest": None if not bt else {"id": bt["id"], **bt["summary"]}}

    def stats_report(self) -> dict:
        nums = self._numbers()
        rows, _ = self.repo.list_results(len(nums) or 1, 0)
        return {"total_results": len(nums), "backtest_state": self.backtest_state, "disclaimer": DISCLAIMER,
                "accuracy": accuracy_report(self.repo.latest_backtest(with_rows=True)), "hot_cold": hot_cold(nums),
                "streaks": streaks(nums), "daily": daily_summary(rows[::-1])}

    def backtest_results(self) -> dict:
        bt = self.repo.latest_backtest(with_rows=True)
        return {"state": self.backtest_state, "backtest": bt}

    # ------------------------------------------------------------ backtest (re-evaluation on all stored data)
    def run_backtest(self, use_ml: bool = True) -> dict:
        with self.lock:
            if self.backtest_state["status"] == "running":
                raise ServiceError("a backtest is already running")
            seq = self._numbers()
            self.backtest_state = {"status": "running", "progress": 0, "total": max(0, len(seq) - self.min_history), "error": None}
        try:
            def prog(i, t): self.backtest_state.update(progress=i, total=t)
            res = run_walk_forward(seq, start=self.min_history, refit_every=self.engine.refit_every, use_ml=use_ml, progress=prog)
            self.repo.save_backtest(res["summary"], res["rows"])
            self.backtest_state = {"status": "done", "progress": len(seq), "total": len(seq), "error": None}
            with self.lock:                                   # refresh the signal on the pending prediction
                last = self.repo.last_result()
                if last:
                    self.repo.expire_pending_except(-1)
                    self._next_prediction()
            return res["summary"]
        except Exception as e:                                # noqa: BLE001 - reported through state
            self.backtest_state = {"status": "error", "progress": 0, "total": 0, "error": str(e)}
            raise
