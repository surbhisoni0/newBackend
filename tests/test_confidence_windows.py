"""Rolling-window live accuracy (50/100/500) and the honest confidence figure. No mocked results: rounds come from the real stored history."""
import unittest
from pathlib import Path
import numpy as np
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService, ROLLING_WINDOWS

CSV = Path(__file__).resolve().parents[1] / "data" / "wingo_docx_clean.csv"


class Windows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = CSV.read_text().splitlines()
        cls.svc = PredictorService(InMemoryRepository(), PredictionEngine(background_refit=False), 100, 300)
        cls.svc.import_csv("\n".join(rows[:1001]))                 # header + 1000 real rounds as history
        cls.live = rows[1001:1061]                                  # 60 later real rounds, fed one by one -> each resolves the pick made before it
        for line in cls.live:
            cls.svc.next_prediction()                               # the pick for the next round is made BEFORE its result exists
            _, num, *_ = line.split(",")
            cls.svc.add_result(int(num), period_id=line.split(",")[0])

    def test_windows_are_consistent_with_history(self):
        perf = self.svc.performance()
        hist = self.svc.history(500)["predictions"]
        done = [h for h in hist if h["result"] in ("HIT", "MISS")]
        w = perf["live"]["windows"]["bs"]
        self.assertEqual(set(w), {str(x) for x in ROLLING_WINDOWS})
        for size in ROLLING_WINDOWS:
            x = w[str(size)]
            last = sorted(done, key=lambda h: h["target_seq"])[-size:]
            self.assertEqual(x["n"], len(last))                      # uses only what exists
            self.assertEqual(x["correct"], sum(h["result"] == "HIT" for h in last))
            self.assertEqual(x["correct"] + x["incorrect"], x["n"])
            self.assertEqual(x["full"], len(done) >= size)
        self.assertEqual(perf["live"]["total_predictions"], len(done))
        self.assertGreater(len(done), 0)

    def test_every_pick_was_made_before_its_result(self):
        for h in self.svc.history(500)["predictions"]:
            if h["result"] != "PENDING":
                self.assertIsNotNone(h["actual"])

    def test_incremental_numbers_equal_full_read(self):
        got = self.svc._numbers()
        full = np.array(self.svc.repo.get_numbers(), dtype=int)
        self.assertTrue(np.array_equal(got, full))                   # extended from the cache, identical to a full database read
        self.assertEqual(len(got), self.svc.repo.last_result()["seq"])
        self.assertFalse(got.flags.writeable)

    def test_by_side_and_calibration_add_up(self):
        live = self.svc.performance()["live"]
        sides = live["by_side"]
        self.assertEqual(sides["Big"]["n"] + sides["Small"]["n"], live["total_predictions"])
        for sd in sides.values():
            self.assertEqual(sd["correct"] + sd["incorrect"], sd["n"])
        cal = live["calibration"]
        self.assertEqual(sum(b["n"] for b in cal), live["total_predictions"])
        for b in cal:
            if b["n"]:
                self.assertGreaterEqual(b["stated"], b["from"]); self.assertLess(b["stated"], max(b["to"], 1.0001))

    def test_no_prediction_when_no_validated_edge(self):
        c = self.svc.next_prediction()["confidence"]
        self.assertTrue(c["abstain"]); self.assertEqual(c["recommendation"], "NO PREDICTION")

    def test_confidence_is_a_coin_flip_without_validated_edge(self):
        p = self.svc.next_prediction()
        c = p["confidence"]
        self.assertEqual(c["signal"], "none")
        self.assertEqual(c["calibrated"], 0.5)                       # never above 50% without validation
        self.assertGreaterEqual(c["raw_model_probability"], 0.5)
        self.assertEqual(c["pick"], p["size"])

    def test_confidence_never_exceeds_validated_lower_bound(self):
        shown = {"size": "Big", "size_probs": {"Big": 0.9, "Small": 0.1}, "signal": {"level": "moderate"}}
        orig = self.svc._validation
        self.svc._validation = lambda: {"metrics": {"bs": {"lo": 0.53, "rate": 0.6}}}
        try:
            c = self.svc._confidence(shown)
            self.assertEqual(c["calibrated"], 0.53)                  # lower bound, not the 90% raw score and not the 60% point estimate
            shown["signal"] = {"level": "weak"}
            self.assertEqual(self.svc._confidence(shown)["calibrated"], 0.5)
        finally:
            self.svc._validation = orig


if __name__ == "__main__":
    unittest.main()
