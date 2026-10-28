import os, sys, time, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.backtesting.walk_forward import run_walk_forward
from app.database.memory_repo import InMemoryRepository
from app.health import build_health
from app.prediction.engine import PredictionEngine
from app.services.colour_analysis import colour_analysis
from app.services.predictor_service import PredictorService
from app.sync.service import SyncService
from app.sync.sources import SourceError


class Src:
    configured, name, last_invalid = True, "fixture", 0
    def __init__(self, error=None): self.error = error
    def fetch(self):
        if self.error: raise SourceError(self.error)
        return []


def service(n=130):
    svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), 100, 300); svc.repo.init()
    svc.import_rows([{"number": (i * 7 + i // 3) % 10, "period_id": str(5001 + i), "timestamp": None} for i in range(n)])
    return svc


class TestColourAnalysis(unittest.TestCase):
    def test_counts_windows_and_streaks(self):
        nums = [1, 3, 7, 2, 4, 0, 5, 9, 1, 3] * 12                     # 120 rounds
        c = colour_analysis(nums)
        self.assertEqual(c["n"], 120); self.assertEqual(sum(map(sum, c["transitions"]["counts"])), 119)
        self.assertEqual(sorted(c["windows"]), ["10", "100", "25", "50"])    # 500 needs 500 rounds
        for row in c["transitions"]["probabilities"]: self.assertAlmostEqual(sum(row), 1.0, places=2)
        self.assertEqual(c["streaks"]["longest"]["Green"], 6)         # 9,1,3 then 1,3,7
        self.assertEqual(c["streaks"]["current"], {"colour": "Green", "length": 3})
        self.assertIsNone(c["windows"]["10"]["p_vs_fair"])               # too small for a chi-square test

    def test_too_little_data(self):
        self.assertIn("message", colour_analysis([1, 2, 3]))

    def test_service_exposes_colour_analysis(self):
        self.assertIn("colour", service().data_stats())


class TestHealth(unittest.TestCase):
    def test_ok_degraded_and_down(self):
        svc = service(); sync = SyncService(svc, Src(), enabled=True)
        sync.run_once(); body, code = build_health(svc.repo, sync, 30)
        self.assertEqual((body["status"], code, body["rounds"]), ("ok", 200, 130))
        sync.source = Src("HTTP 500"); sync.run_once()
        body, code = build_health(svc.repo, sync, 30); self.assertEqual((body["status"], code), ("degraded", 200))
        class Broken(InMemoryRepository):
            def last_result(self): raise RuntimeError("db down")
        body, code = build_health(Broken(), sync, 30); self.assertEqual((body["status"], code), ("down", 503))


class TestStale(unittest.TestCase):
    def test_status_turns_stale_when_no_recent_success(self):
        svc = service(); sync = SyncService(svc, Src(), enabled=True, stale_after=300)
        self.assertEqual(sync.run_once()["state"], "ok")
        sync.s["last_success"] = time.time() - 1000
        self.assertEqual(sync.status()["state"], "stale")


class TestNoFutureLeakage(unittest.TestCase):
    def test_changing_future_rounds_never_changes_earlier_predictions(self):
        rng = np.random.default_rng(3); a = rng.integers(0, 10, 180)
        b = a.copy(); b[140:] = rng.integers(0, 10, 40)                   # same past, different future
        ra = run_walk_forward(a, start=100, refit_every=10, use_ml=False)["rows"]
        rb = run_walk_forward(b, start=100, refit_every=10, use_ml=False)["rows"]
        for x, y in zip(ra[:40], rb[:40]):                               # targets 101..140 only see rounds before them
            self.assertEqual((x["target_seq"], x["top1"], x["top3"], x["color"]), (y["target_seq"], y["top1"], y["top3"], y["color"]))

    def test_prediction_for_next_round_uses_only_stored_history(self):
        svc = service(); p = svc.next_prediction()
        self.assertEqual((p["n_history"], p["target_seq"]), (130, 131))


if __name__ == "__main__":
    unittest.main()
