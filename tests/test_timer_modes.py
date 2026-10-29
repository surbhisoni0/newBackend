"""30s / 1min / 2min timer modes: countdown, period roll-over, one stable prediction per period, honest 2min (no source)."""
import os, random, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime, timezone
from app.config import Settings
from app.database.memory_repo import InMemoryRepository
from app.modes import MODES
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.timing import ClockConfig

NOW = 1793181600.0                                       # a fixed "now" so every second is reproducible


def iso(e): return datetime.fromtimestamp(e, timezone.utc).isoformat().replace("+00:00", "Z")


class _Feed:                                             # what WinGoFeedSource exposes once it has read the OPEN round
    def __init__(self, pid, end, fetched): self.open_round, self.fetched_at, self.clock_offset = {"period_id": pid, "end": end}, fetched, 0.0


class TestTimerModes(unittest.TestCase):
    def test_three_options_exist(self):
        self.assertEqual([MODES[k]["seconds"] for k in ("30s", "1min", "2min")], [30, 60, 120])
        self.assertIsNone(MODES["2min"]["type_id"])                         # no results feed is known for 2 minutes

    def _live(self, P, clock):
        rng = random.Random(P)
        s = PredictorService(InMemoryRepository(), PredictionEngine(), clock_cfg=ClockConfig(period_seconds=P), now_fn=lambda: clock[0])
        base = 20261029100010000
        s.import_rows([{"number": rng.randrange(10), "period_id": str(base + i), "timestamp": iso(NOW - P * (120 - i))} for i in range(1, 121)])
        s.live_source = _Feed(str(base + 121), NOW + P, NOW)                # round 121 is open and ends P seconds from NOW
        return s, base

    def test_30s_and_1min_period_transitions_and_stable_prediction(self):
        for P in (30, 60):
            clock = [NOW]
            s, base = self._live(P, clock)
            c = s.clock()["countdown"]
            self.assertEqual((c["state"], c["source"], c["current_period"], round(c["seconds_remaining"])), ("open", "feed", str(base + 121), P))
            first = s.next_prediction()
            self.assertEqual((first["available"], first["status"], first["target_period"]), (True, "pending", str(base + 121)))
            self.assertIsNone(first["outcome"])                               # an estimate is never presented as a result
            for sec in range(1, P):                                           # every second of the period: same prediction, countdown falls
                clock[0] = NOW + sec
                again = s.next_prediction()
                self.assertEqual((again["id"], again["created_at"], again["size"]), (first["id"], first["created_at"], first["size"]))
                self.assertEqual(round(s.clock()["countdown"]["seconds_remaining"]), P - sec)
            clock[0] = NOW + P + 1                                            # boundary passed: countdown resets, next period id, no invented result
            c = s.clock()["countdown"]
            self.assertEqual((c["current_period"], round(c["seconds_remaining"]), c["awaiting_results"]), (str(base + 122), P - 1, [str(base + 121)]))
            self.assertEqual(s.next_prediction()["id"], first["id"])          # still the same estimate until the real result arrives
            out = s.add_result(7, str(base + 121), iso(NOW + P))              # the confirmed result of the ended round
            self.assertEqual(out["resolved_prediction"]["status"], "resolved")
            self.assertEqual(out["resolved_prediction"]["outcome"]["actual"], 7)
            nxt = s.next_prediction()
            self.assertEqual((nxt["target_period"], nxt["status"]), (str(base + 122), "pending"))
            self.assertNotEqual(nxt["id"], first["id"])

    def test_2min_has_a_labelled_assumed_timer_and_no_predictions(self):
        cfg = Settings(assumed_schedule=False).clock_config(MODES["2min"]["seconds"], force_assumed=True)
        self.assertTrue(cfg.assumed)
        self.assertFalse(Settings(assumed_schedule=False).clock_config(60).assumed)       # other modes unchanged
        clock = [NOW + 10]
        s = PredictorService(InMemoryRepository(), PredictionEngine(), clock_cfg=cfg, now_fn=lambda: clock[0])
        c = s.clock()["countdown"]
        self.assertEqual((c["state"], c["period_seconds"], round(c["seconds_remaining"])), ("assumed", 120, 110))   # never 'open' / verified
        clock[0] = NOW + 120 + 5
        self.assertEqual(round(s.clock()["countdown"]["seconds_remaining"]), 115)       # resets for the next 2-minute period
        p = s.next_prediction()
        self.assertFalse(p["available"])                                                # no results -> no prediction is invented


if __name__ == "__main__":
    unittest.main()
