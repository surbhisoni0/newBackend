"""MongoDB repository (pymongo). DATABASE_URL=mongodb://localhost:27017/wingo   (or a mongodb+srv:// Atlas URL).

Collections: raw_results (never modified) | results | predictions | prediction_outcomes | backtest_runs |
performance_snapshots | counters.  results._id IS the sequence number (1, 2, 3 ...), so order is exact and gap-free.
A standalone local MongoDB has no multi-document transactions, so batch inserts are made "all or nothing" by
removing whatever was inserted if any step fails.
"""
import json
import threading
from datetime import datetime, timezone
from typing import Optional
from app.database.base import Repository


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(obj):
    """Make values BSON-safe (numpy numbers/bools -> Python, int dict keys -> str), like a JSON round trip."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


class MongoRepository(Repository):
    def __init__(self, url: str, client=None):
        if client is None:
            from pymongo import MongoClient
            client = MongoClient(url, serverSelectionTimeoutMS=5000)
        self.url, self.client = url, client
        self.db = client.get_default_database(default="wingo")
        self.lock = threading.RLock()

    # ------------------------------------------------------------ setup
    def init(self):
        try:
            self.client.admin.command("ping")
        except Exception as e:                                        # noqa: BLE001
            raise RuntimeError("Cannot reach MongoDB. Start the MongoDB service (or fix DATABASE_URL in backend/.env). "
                               f"Details: {e}") from e
        self.db.results.create_index("period_id", unique=True, partialFilterExpression={"period_id": {"$type": "string"}})
        self.db.predictions.create_index([("target_seq", 1), ("status", 1)])

    def _next_id(self, name: str, n: int = 1) -> int:
        """Atomically reserve n ids; returns the FIRST reserved id."""
        from pymongo import ReturnDocument
        doc = self.db.counters.find_one_and_update({"_id": name}, {"$inc": {"v": n}}, upsert=True, return_document=ReturnDocument.AFTER)
        return doc["v"] - n + 1

    # ------------------------------------------------------------ results
    @staticmethod
    def _row(d: dict) -> dict:
        return {"seq": d["_id"], "period_id": d.get("period_id"), "number": d["number"], "timestamp": d.get("timestamp"),
                "raw_id": d.get("raw_id"), "created_at": d.get("created_at")}

    def add_results(self, items):
        if not items:
            return []
        with self.lock:
            last = self.db.results.find_one({}, sort=[("_id", -1)])
            first_seq = (last["_id"] + 1) if last else 1
            first_raw = self._next_id("raw_results", len(items))
            raws = [{"_id": first_raw + i, "payload": _clean(it["raw"]), "received_at": _now()} for i, it in enumerate(items)]
            rows = [{"_id": first_seq + i, "raw_id": first_raw + i, "period_id": it.get("period_id"), "number": int(it["number"]),
                     "timestamp": it.get("timestamp"), "created_at": _now()} for i, it in enumerate(items)]
            try:
                self.db.raw_results.insert_many(raws)
                self.db.results.insert_many(rows)
            except Exception:
                self.db.results.delete_many({"_id": {"$gte": first_seq}})
                self.db.raw_results.delete_many({"_id": {"$gte": first_raw, "$lt": first_raw + len(items)}})
                raise
            return [self._row(r) for r in rows]

    def get_numbers(self):
        return [int(d["number"]) for d in self.db.results.find({}, {"number": 1}).sort("_id", 1)]

    def get_meta(self):
        return [{"seq": d["_id"], "period_id": d.get("period_id"), "timestamp": d.get("timestamp")}
                for d in self.db.results.find({}, {"period_id": 1, "timestamp": 1}).sort("_id", 1)]

    def list_results(self, limit, offset):
        rows = self.db.results.find({}).sort("_id", -1).skip(offset).limit(limit)
        return [self._row(d) for d in rows], int(self.db.results.count_documents({}))

    def last_result(self):
        d = self.db.results.find_one({}, sort=[("_id", -1)])
        return self._row(d) if d else None

    def has_period(self, period_id):
        return self.db.results.count_documents({"period_id": period_id}, limit=1) > 0

    # ------------------------------------------------------------ predictions
    _PRED_KEYS = ("target_seq", "target_period", "engine_version", "n_history", "top1", "top3", "size", "color",
                  "signal", "ranking", "model_tops", "model_scores")
    _OPTIONAL_KEYS = ("color_probs", "size_probs")

    @staticmethod
    def _pred(d: dict, o: Optional[dict]) -> dict:
        out = {k: v for k, v in d.items() if k != "_id"}
        out["id"] = d["_id"]
        out["outcome"] = None if o is None else {k: v for k, v in o.items() if k not in ("_id",)}
        return out

    def _outcomes_for(self, ids):
        return {o["_id"]: o for o in self.db.prediction_outcomes.find({"_id": {"$in": list(ids)}})} if ids else {}

    def save_prediction(self, pred):
        pid = self._next_id("predictions")
        doc = {"_id": pid, **_clean({k: pred[k] for k in self._PRED_KEYS + tuple(k for k in self._OPTIONAL_KEYS if k in pred)}), "status": "pending", "created_at": _now()}
        self.db.predictions.insert_one(doc)
        return pid

    def get_prediction_by_target(self, target_seq):
        d = self.db.predictions.find_one({"target_seq": target_seq, "status": {"$ne": "void"}}, sort=[("_id", -1)])
        return self._pred(d, self.db.prediction_outcomes.find_one({"_id": d["_id"]})) if d else None

    def expire_pending_except(self, target_seq):
        self.db.predictions.update_many({"status": "pending", "target_seq": {"$ne": target_seq}}, {"$set": {"status": "void"}})

    def resolve_prediction(self, prediction_id, outcome):
        self.db.prediction_outcomes.replace_one({"_id": prediction_id}, {"_id": prediction_id, **_clean(outcome), "resolved_at": _now()}, upsert=True)
        self.db.predictions.update_one({"_id": prediction_id}, {"$set": {"status": "resolved"}})

    def list_predictions(self, limit, offset):
        q = {"status": {"$ne": "void"}}
        rows = list(self.db.predictions.find(q).sort([("target_seq", -1), ("_id", -1)]).skip(offset).limit(limit))
        outs = self._outcomes_for([r["_id"] for r in rows])
        return [self._pred(r, outs.get(r["_id"])) for r in rows], int(self.db.predictions.count_documents(q))

    def all_outcomes(self):
        outs = {o["_id"]: o for o in self.db.prediction_outcomes.find({})}
        if not outs:
            return []
        rows = self.db.predictions.find({"_id": {"$in": list(outs)}})
        return sorted((self._pred(r, outs[r["_id"]]) for r in rows), key=lambda r: r["target_seq"])

    # ------------------------------------------------------------ backtests / snapshots
    def save_backtest(self, summary, rows):
        rid = self._next_id("backtest_runs")
        self.db.backtest_runs.insert_one({"_id": rid, "summary": _clean(summary), "rows": _clean(rows)})
        return rid

    def latest_backtest(self, with_rows=False):
        d = self.db.backtest_runs.find_one({}, None if with_rows else {"rows": 0}, sort=[("_id", -1)])
        if not d:
            return None
        out = {"id": d["_id"], "summary": d["summary"]}
        if with_rows:
            out["rows"] = sorted(d.get("rows", []), key=lambda r: r["target_seq"])
        return out

    def save_snapshot(self, snap):
        self.db.performance_snapshots.insert_one({"_id": self._next_id("snapshots"), **_clean(snap), "created_at": _now()})

    def list_snapshots(self, limit=500):
        rows = list(self.db.performance_snapshots.find({}).sort("_id", -1).limit(limit))[::-1]
        return [{"n": r.get("n"), "top1": r.get("top1"), "top3": r.get("top3"), "bs": r.get("bs"), "color": r.get("color"),
                 "created_at": r.get("created_at")} for r in rows]
