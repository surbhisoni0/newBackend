"""Opt-in ASSUMED schedule: counted, but never presented as verified; a verified schedule always wins."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.timing import ClockConfig, assumed_countdown

T0 = 1_800_000_000 - (1_800_000_000 % 60)        # an exact minute boundary


def service(assumed=True, offset=0.0, now=T0 + 20.0):
    repo = InMemoryRepository(); repo.init()
    cfg = ClockConfig(assumed=assumed, assumed_offset_seconds=offset)
    return PredictorService(repo, PredictionEngine(), clock_cfg=cfg, now_fn=lambda: now)


class TestAssumed(unittest.TestCase):
    def test_counts_to_next_boundary(self):
        c = assumed_countdown(T0 + 20.0, ClockConfig(assumed=True))
        self.assertEqual(c["state"], "assumed"); self.assertAlmostEqual(c["seconds_remaining"], 40.0); self.assertEqual(c["round_ends_at"], T0 + 60)

    def test_exact_boundary_starts_a_new_round(self):
        self.assertAlmostEqual(assumed_countdown(T0 + 60.0, ClockConfig(assumed=True))["seconds_remaining"], 60.0)
        self.assertAlmostEqual(assumed_countdown(T0 + 59.999, ClockConfig(assumed=True))["seconds_remaining"], 0.001, places=3)

    def test_offset_shifts_boundary(self):
        c = assumed_countdown(T0 + 20.0, ClockConfig(assumed=True, assumed_offset_seconds=30))
        self.assertAlmostEqual(c["seconds_remaining"], 10.0); self.assertEqual(c["round_ends_at"], T0 + 30)
        self.assertAlmostEqual(assumed_countdown(T0 + 40.0, ClockConfig(assumed=True, assumed_offset_seconds=30))["seconds_remaining"], 50.0)

    def test_off_by_default_keeps_unverified(self):
        s = service(assumed=False); s.import_rows([{"number": i % 10, "period_id": None, "timestamp": None} for i in range(10)])
        self.assertEqual(s.clock()["countdown"]["state"], "unverified")
        self.assertEqual(service(assumed=False).clock()["countdown"]["state"], "no_data")

    def test_unverified_data_falls_back_to_assumed_and_never_claims_verified(self):
        s = service(); s.import_rows([{"number": i % 10, "period_id": None, "timestamp": None} for i in range(10)])
        ck = s.clock()
        self.assertEqual(ck["countdown"]["state"], "assumed"); self.assertFalse(ck["schedule"]["verified"])
        self.assertIn("ASSUMED", ck["countdown"]["reason"]); self.assertTrue(ck["countdown"]["round_ends_at"].endswith("Z"))

    def test_empty_database_also_counts_when_assumed(self):
        ck = service().clock(); self.assertEqual(ck["countdown"]["state"], "assumed"); self.assertIsNone(ck["next_round"])

    def test_verified_schedule_wins_over_assumed(self):
        from datetime import datetime, timezone
        iso = lambda e: datetime.fromtimestamp(e, timezone.utc).isoformat()
        now = T0 + 20.0
        s = service(now=now)
        rows = [{"number": i % 10, "period_id": str(700001 + i), "timestamp": iso(now - 5 - (29 - i) * 60)} for i in range(30)]
        s.import_rows(rows)
        ck = s.clock(); self.assertTrue(ck["schedule"]["verified"]); self.assertEqual(ck["countdown"]["state"], "open")

    def test_prediction_is_not_marked_open_when_assumed(self):
        s = service(); s.import_rows([{"number": (i * 7 + i // 3) % 10, "period_id": None, "timestamp": None} for i in range(130)])
        p = s.next_prediction(); self.assertEqual(p["estimate_state"], "timing_unverified"); self.assertEqual(p["target_id_kind"], "sequence")


if __name__ == "__main__":
    unittest.main()
