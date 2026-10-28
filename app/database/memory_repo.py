"""In-memory repository (tests / demos). Data is lost on restart. DATABASE_URL=memory://"""
import copy
import threading
from datetime import datetime, timezone
from typing import Optional
from app.database.base import Repository


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryRepository(Repository):
    def __init__(self):
        self.lock = threading.RLock()
        self.raw, self.results, self.preds, self.outcomes = [], [], [], {}
        self.backtests, self.snapshots = [], []

    def init(self): pass

    def add_results(self, items):
        with self.lock:
            out = []
            for it in items:
                self.raw.append({"id": len(self.raw) + 1, "payload": it["raw"], "received_at": _now()})
                row = {"seq": len(self.results) + 1, "period_id": it.get("period_id"), "number": it["number"],
                       "timestamp": it.get("timestamp"), "raw_id": len(self.raw), "created_at": _now()}
                self.results.append(row); out.append(dict(row))
            return out

    def get_numbers(self): return [r["number"] for r in self.results]
    def get_meta(self): return [{"seq": r["seq"], "period_id": r["period_id"], "timestamp": r["timestamp"]} for r in self.results]

    def list_results(self, limit, offset):
        rows = self.results[::-1][offset: offset + limit]
        return [dict(r) for r in rows], len(self.results)

    def last_result(self): return dict(self.results[-1]) if self.results else None
    def has_period(self, period_id): return any(r["period_id"] == period_id for r in self.results)

    def save_prediction(self, pred):
        with self.lock:
            p = copy.deepcopy(pred); p["id"] = len(self.preds) + 1; p["created_at"] = _now(); p["status"] = "pending"
            self.preds.append(p); return p["id"]

    def _join(self, p):
        q = copy.deepcopy(p); q["outcome"] = self.outcomes.get(p["id"]); return q

    def get_prediction_by_target(self, target_seq):
        for p in reversed(self.preds):
            if p["target_seq"] == target_seq and p["status"] != "void":
                return self._join(p)
        return None

    def expire_pending_except(self, target_seq):
        with self.lock:
            for p in self.preds:
                if p["status"] == "pending" and p["target_seq"] != target_seq:
                    p["status"] = "void"

    def resolve_prediction(self, prediction_id, outcome):
        with self.lock:
            self.outcomes[prediction_id] = {**outcome, "resolved_at": _now()}
            self.preds[prediction_id - 1]["status"] = "resolved"

    def list_predictions(self, limit, offset):
        rows = [p for p in self.preds if p["status"] != "void"][::-1]
        return [self._join(p) for p in rows[offset: offset + limit]], len(rows)

    def all_outcomes(self):
        rows = [self._join(p) for p in self.preds if p["id"] in self.outcomes]
        return sorted(rows, key=lambda r: r["target_seq"])

    def save_backtest(self, summary, rows):
        with self.lock:
            self.backtests.append({"id": len(self.backtests) + 1, "summary": summary, "rows": rows}); return len(self.backtests)

    def latest_backtest(self, with_rows=False):
        if not self.backtests: return None
        b = copy.deepcopy(self.backtests[-1])
        if not with_rows: b.pop("rows")
        return b

    def save_snapshot(self, snap): self.snapshots.append({**snap, "created_at": _now()})
    def list_snapshots(self, limit=500): return self.snapshots[-limit:]
