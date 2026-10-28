import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.backtesting.colour_eval import SIZE_PRIOR, size_eval
from app.backtesting.walk_forward import run_walk_forward
from app.database.memory_repo import InMemoryRepository
from app.prediction import size as sz
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.services.size_analysis import size_analysis


class TestSizeProbs(unittest.TestCase):
    def test_probabilities_sum_to_one_and_pick_matches(self):
        scores = {"a": np.ones(10) / 10, "b": np.arange(1, 11) / 55}        # b leans Big
        p = sz.size_probs(scores); self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertEqual(sz.size_pick(p), "Big"); self.assertEqual(sz.size_pick(np.array([.5, .5])), "Small")

    def test_prediction_pick_is_the_argmax_of_the_shown_probabilities(self):
        nums = np.array([(i * 7 + i // 3) % 10 for i in range(150)])
        p = PredictionEngine(use_ml=False).predict(nums)
        self.assertAlmostEqual(sum(p["size_probs"].values()), 1.0, places=2)
        self.assertEqual(p["size"], "Big" if p["size_probs"]["Big"] > p["size_probs"]["Small"] else "Small")


class TestSizeEval(unittest.TestCase):
    def test_coin_flip_probabilities_have_zero_skill(self):
        g = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        e = size_eval(np.tile(SIZE_PRIOR, (8, 1)), g)
        self.assertAlmostEqual(e["brier_skill"], 0.0, places=6); self.assertAlmostEqual(e["logloss_diff"], 0.0, places=6)
        self.assertEqual(e["classes"], ["Small", "Big"])

    def test_perfect_and_confidently_wrong(self):
        g = np.array([0, 1, 0, 1]); good = np.array([[.9, .1], [.1, .9]] * 2); bad = good[::-1]
        self.assertEqual(size_eval(good, g)["confusion"], [[2, 0], [0, 2]])
        e = size_eval(bad, g); self.assertEqual(e["accuracy"], 0.0); self.assertLess(e["brier_skill"], 0)

    def test_walk_forward_reports_size_eval_without_future_leakage(self):
        rng = np.random.default_rng(5); a = rng.integers(0, 10, 170); b = a.copy(); b[140:] = rng.integers(0, 10, 30)
        ra, rb = run_walk_forward(a, start=100, refit_every=10, use_ml=False), run_walk_forward(b, start=100, refit_every=10, use_ml=False)
        self.assertEqual(ra["summary"]["size_eval"]["n"], 70)
        for x, y in zip(ra["rows"][:40], rb["rows"][:40]):
            self.assertEqual((x["target_seq"], x["size"]), (y["target_seq"], y["size"]))


class TestSizeAnalysis(unittest.TestCase):
    def test_windows_transitions_streaks(self):
        nums = [1, 2, 7, 8, 9, 3] * 20                                       # S S B B B S ...
        c = size_analysis(nums)
        self.assertEqual((c["n"], sum(map(sum, c["transitions"]["counts"]))), (120, 119))
        self.assertEqual(sorted(c["windows"]), ["10", "100", "25", "50"])
        self.assertEqual(c["streaks"]["longest"], {"Small": 3, "Big": 3})      # S,S + next cycle's S = 3 Smalls in a row
        for row in c["transitions"]["probabilities"]: self.assertAlmostEqual(sum(row), 1.0, places=2)

    def test_service_stats_and_history_use_size(self):
        svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), 100, 300); svc.repo.init()
        svc.import_rows([{"number": (i * 7 + i // 3) % 10, "period_id": str(5001 + i), "timestamp": None} for i in range(130)])
        self.assertIn("size", svc.data_stats())
        svc.add_result(8)                                                    # resolves the pending prediction
        pending, resolved = svc.history(10, 0)["predictions"][:2]               # newest first: next round, then the one just scored
        self.assertEqual(pending["result"], "PENDING")
        self.assertIn(resolved["size"], ("Big", "Small")); self.assertEqual(resolved["actual"], 8)
        self.assertEqual(resolved["bs_hit"], resolved["size"] == "Big"); self.assertEqual(resolved["result"], "HIT" if resolved["bs_hit"] else "MISS")


if __name__ == "__main__":
    unittest.main()
