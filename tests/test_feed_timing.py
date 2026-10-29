"""Live-feed timing: period id roll-over and the current round derived from the feed's open round."""
import os, sys, time, unittest
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.config import Settings
from app.sync.sources import WinGoFeedSource
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


    def test_feed_ist_times_verify_even_when_source_timezone_is_utc(self):
        """Regression: feed endTime is IST without offset. It was stored raw, so SOURCE_TIMEZONE=UTC read it 5.5 h in the future
        and the clock said 'unverified'. Stored times must now be unambiguous."""
        ist, end = timezone(timedelta(hours=5, minutes=30)), (int(time.time()) // 60) * 60
        items = [{"state": 1, "number": i % 10, "issueNumber": str(20261029100010001 + i),
                  "endTime": datetime.fromtimestamp(end - 60 * (9 - i), ist).strftime("%Y-%m-%d %H:%M:%S")} for i in range(10)]
        recs = WinGoFeedSource("https://example.invalid/x", 1, 60)._settled(items)
        self.assertTrue(all(r.timestamp.endswith("Z") for r in recs))
        s = self._svc(60, None)                                                     # ClockConfig default source_tz = "UTC"
        s.import_rows([{"number": r.number, "period_id": r.period_id, "timestamp": r.timestamp} for r in recs])
        c = s.clock()
        self.assertTrue(c["schedule"]["verified"], c["schedule"])
        self.assertEqual(c["countdown"]["state"], "open")
        self.assertEqual(c["next_round"]["period_id"], "20261029100010011")
        bad = WinGoFeedSource("https://example.invalid/x", 1, 60)._settled([{"state": 1, "number": 3, "issueNumber": "1", "endTime": "not a time"}])
        self.assertIsNone(bad[0].timestamp)                                          # never guessed


    def test_rows_stored_before_the_fix_still_verify_in_feed_mode(self):
        """Regression: rows already in the DB are naive IST text. With the feed on, they must be read as IST even if SOURCE_TIMEZONE=UTC;
        with the feed off, SOURCE_TIMEZONE is still honoured."""
        ist, end = timezone(timedelta(hours=5, minutes=30)), (int(time.time()) // 60) * 60
        rows = [{"number": i % 10, "period_id": str(20261029100010001 + i),
                 "timestamp": datetime.fromtimestamp(end - 60 * (29 - i), ist).strftime("%Y-%m-%dT%H:%M:%S")} for i in range(30)]
        feed = Settings(feed_enabled=True, feed_url="https://example.invalid", source_timezone="UTC")
        s = PredictorService(InMemoryRepository(), PredictionEngine(), clock_cfg=feed.clock_config(60)); s.import_rows(rows)
        self.assertEqual(s.clock()["countdown"]["state"], "open")
        off = Settings(feed_enabled=False, feed_url="", source_timezone="UTC")
        self.assertEqual(off.clock_config(60).source_tz, "UTC")


if __name__ == "__main__":
    unittest.main()
