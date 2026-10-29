"""One estimate per completed cycle, for both timers (30s / 1min), over several consecutive cycles.
Estimate = model output bound to ONE round; verified result = what the feed/user confirmed. They are never mixed up."""
import os, random, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.sync.service import SyncService
from app.sync.sources import RoundRecord, WinGoFeedSource
from app.timing import ClockConfig, iso

NOW0, BASE = 1793181600.0, 20261029100050000


class FakeFeed(WinGoFeedSource):
    """Settled rounds show up `lag` s after their end; stops settling after `stall_at` (simulated outage)."""
    def __init__(self, P, clock, lag=2.0, stall_at=None):
        super().__init__("http://fake", 1, P); self.P, self.clock, self.lag, self.stall_at = P, clock, lag, stall_at
        self.rng, self.nums = random.Random(P), {}
    @property
    def configured(self): return True
    def fetch(self, backfill=False):
        t = self.clock[0]; self.clock_offset, self.fetched_at = 0.0, t
        ko = int((t - NOW0) // self.P) + 1
        self.open_round = {"period_id": str(BASE + ko), "end": NOW0 + ko * self.P, "start": NOW0 + (ko - 1) * self.P}
        te = t if self.stall_at is None else min(t, self.stall_at)
        return [RoundRecord(str(BASE + k), self.nums.setdefault(k, self.rng.randrange(10)), iso(NOW0 + k * self.P))
                for k in range(max(1, ko - 40), ko + 1) if te >= NOW0 + k * self.P + self.lag]


def live_service(P, stall_at=None):
    clock = [NOW0 + 0.5]
    svc = PredictorService(InMemoryRepository(), PredictionEngine(), clock_cfg=ClockConfig(period_seconds=P), now_fn=lambda: clock[0])
    rng = random.Random(1)
    svc.import_rows([{"number": rng.randrange(10), "period_id": str(BASE - 149 + i), "timestamp": iso(NOW0 - P * (149 - i))} for i in range(150)])
    feed = FakeFeed(P, clock, stall_at=stall_at); svc.live_source = feed
    sync = SyncService(svc, feed, True, 10, allow_gaps=True); sync.run_once()
    return svc, sync, clock


class TestCycleUpdates(unittest.TestCase):
    def test_30s_and_1min_one_new_estimate_per_cycle_stable_inside_it(self):
        for P in (30, 60):
            svc, sync, clock = live_service(P)
            prev = svc.next_prediction()
            for cycle in range(1, 7):
                end = NOW0 + cycle * P
                for sec in range(0, P, 3):                                    # inside the cycle: nothing changes
                    clock[0] = end - P + 0.5 + sec; sync.run_once(); p = svc.next_prediction()
                    self.assertEqual((p["id"], p["size"], p["target_period"]), (prev["id"], prev["size"], prev["target_period"]), f"P={P} cycle={cycle} +{sec}s")
                clock[0] = end + 0.5; sync.run_once()                          # boundary, verified result not out yet: still the same estimate
                self.assertEqual(svc.next_prediction()["id"], prev["id"])
                clock[0] = end + 2.5; sync.run_once(); cur = svc.next_prediction()   # verified result lands: exactly one new estimate
                self.assertNotEqual(cur["id"], prev["id"]); self.assertEqual(cur["status"], "pending"); self.assertIsNone(cur["outcome"])
                self.assertEqual(int(cur["target_period"]), int(prev["target_period"]) + 1)
                scored = svc.history(2)["predictions"][1]                      # the old estimate is scored against the VERIFIED number, separately
                self.assertEqual(scored["round"], prev["target_period"]); self.assertIn(scored["result"], ("HIT", "MISS")); self.assertIsNotNone(scored["actual"])
                prev = cur
            self.assertEqual(svc.history(50)["total"], 7)                      # 1 initial + 6 cycles, no duplicates

    def test_no_new_verified_result_means_no_new_estimate(self):
        svc, sync, clock = live_service(30, stall_at=NOW0 + 40)               # feed stops settling after the first cycle
        ids = set()
        for t in range(0, 30 * 6, 5):
            clock[0] = NOW0 + 0.5 + t; sync.run_once()
            if clock[0] > NOW0 + 45: ids.add(svc.next_prediction()["id"])
        self.assertEqual(len(ids), 1)                                          # frozen, not re-rolled, across ~4 further cycles
        self.assertEqual(svc.history(50)["total"], 2)


if __name__ == "__main__":
    unittest.main()
