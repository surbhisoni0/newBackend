import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import pymongo_stub  # noqa: F401  (installs a ReturnDocument stub when pymongo is missing)
from fake_mongo import FakeClient
from app.database.mongo_repo import MongoRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService


def item(n, p=None): return {"raw": {"number": n, "period_id": p}, "number": n, "period_id": p, "timestamp": None}


class TestMongoRepo(unittest.TestCase):
    def setUp(self):
        self.repo = MongoRepository("mongodb://fake/wingo", client=FakeClient()); self.repo.init()

    def test_results_are_sequential_and_ordered(self):
        self.repo.add_results([item(5), item(0), item(7)]); self.repo.add_results([item(2)])
        self.assertEqual(self.repo.get_numbers(), [5, 0, 7, 2])
        self.assertEqual(self.repo.last_result()["seq"], 4)
        rows, total = self.repo.list_results(2, 0)
        self.assertEqual((total, [r["seq"] for r in rows]), (4, [4, 3]))

    def test_failed_batch_leaves_nothing_behind(self):
        self.repo.add_results([item(1, "100")])
        with self.assertRaises(Exception):
            self.repo.add_results([item(2, "101"), item(3, "100")])      # duplicate period id -> whole batch rolled back
        self.assertEqual(self.repo.get_numbers(), [1])
        self.repo.add_results([item(4, "102")]); self.assertEqual(self.repo.last_result()["seq"], 2)

    def test_prediction_outcome_flow(self):
        pred = {"target_seq": 3, "target_period": None, "engine_version": "x", "n_history": 2, "top1": 1, "top3": [1, 2, 3],
                "size": "Small", "color": "Green", "signal": {"label": "NO"}, "ranking": [], "model_tops": {}, "model_scores": {}}
        pid = self.repo.save_prediction(pred)
        self.assertEqual(self.repo.get_prediction_by_target(3)["status"], "pending")
        self.repo.resolve_prediction(pid, {"actual": 1, "top1_hit": True, "top3_hit": True, "bs_hit": True, "color_hit": True})
        got = self.repo.get_prediction_by_target(3)
        self.assertEqual((got["status"], got["outcome"]["color_hit"]), ("resolved", True))
        self.assertEqual(len(self.repo.all_outcomes()), 1)
        self.assertEqual(self.repo.list_predictions(10, 0)[1], 1)
        self.repo.expire_pending_except(99); self.assertEqual(self.repo.list_predictions(10, 0)[1], 1)

    def test_backtest_and_snapshots(self):
        self.repo.save_backtest({"data_n": 3}, [{"target_seq": 2}, {"target_seq": 1}])
        self.assertNotIn("rows", self.repo.latest_backtest())
        self.assertEqual([r["target_seq"] for r in self.repo.latest_backtest(True)["rows"]], [1, 2])
        self.repo.save_snapshot({"n": 1, "top1": 0.1, "top3": 0.3, "bs": 0.5, "color": None})
        self.assertEqual(self.repo.list_snapshots()[0]["n"], 1)

    def test_full_service_flow_on_mongo_repo(self):
        svc = PredictorService(self.repo, PredictionEngine(use_ml=False), min_history=100, min_validation_n=300)
        nums = [(i * 7 + i // 3) % 10 for i in range(130)]
        svc.import_rows([{"number": n, "period_id": None, "timestamp": None} for n in nums])
        p = svc.next_prediction(); self.assertTrue(p["available"])
        r = svc.add_result(4); self.assertIsNotNone(r["resolved_prediction"])
        self.assertEqual(svc.live_metrics()["color"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
