"""Live-feed timing: period id roll-over and the current round derived from the feed's open round."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.timing import ClockConfig, advance_period_id, next_period_id


class _Src:
    def __init__(self, pid, end, fetched, offset=0.0):
        self.open_round, self.fetched_at, self.clock_offset = {"period_id": pid, "end": end}, fetched, offset


class TestFeedTiming(unittest.TestCase):
    def test_rollover_at_end_of_utc_day(self):
        self.assertEqual(next_period_id("20261029100011440"), "20261030100010001")      # 1min: 1440 rounds/day
        self.assertEqual(next_period_id("20261029100052880"), "20261030100050001")      # 30s
        self.assertEqual(next_period_id("20261029101020480"), "20261030101020001")      # 3min
        self.assertEqual(next_period_id("20261231100030288"), "20270101100030001")      # 5min, year change
        self.assertEqual(next_period_id("20261029100010444"), "20261029100010445")
        self.assertEqual(advance_period_id("20261029100010444", 3), "20261029100010447")

    def _svc(self, P, src):
        s = PredictorService(InMemoryRepository(), PredictionEngine(), clock_cfg=ClockConfig(period_seconds=P)); s.live_source = src
        return s

    def test_open_round_and_roll_forward(self):
        s = self._svc(60, _Src("20261029100010444", 1000.0, 990.0))
        r = s._live_round(970.0, {"period_id": "20261029100010443"})
        self.assertEqual((r["current_period"], r["next_period"], round(r["seconds_remaining"])), ("20261029100010444", "20261029100010445", 30))
        self.assertEqual(r["awaiting_results"], [])
        r = s._live_round(1001.0, {"period_id": "20261029100010443"})               # boundary passed before the next poll
        self.assertEqual((r["current_period"], round(r["seconds_remaining"]), r["id_source"]), ("20261029100010445", 59, "feed+schedule"))
        self.assertEqual(r["awaiting_results"], ["20261029100010444"])              # its result is pending, never invented

    def test_clock_offset_and_stale_feed(self):
        s = self._svc(30, _Src("20261029100050886", 1000.0, 990.0, offset=2.0))
        self.assertEqual(round(s._live_round(990.0, {})["seconds_remaining"]), 8)  # feed clock is 2 s ahead
        self.assertIsNone(s._live_round(990.0 + 200, {}))                          # feed not read for > 3 periods: no live claim


if __name__ == "__main__":
    unittest.main()
