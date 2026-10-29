"""Per-cycle prediction refresh for all three timers (30 s / 1 min / 2 min), over many consecutive cycles.

What is proven here (simulated clock, the REAL engine and service, in-memory database):
  * the estimate is identical for the whole countdown (no per-second change);
  * it is recalculated exactly once per completed cycle (counted by spying on engine.predict);
  * the Big/Small value is a deterministic function of the verified history: unchanged data -> unchanged value (never random);
  * a recalculated "projection" is never stored and never scored; only stored picks (made before their round) are scored;
  * predictions and verified results stay separate fields, and a late result is scored against the ORIGINAL pick of its own round.
Numbers come from the project's own historical CSV; they only drive code paths, no stored result is invented.
"""
import csv
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import SETTLE_SECONDS, PredictorService
from app.timing import ClockConfig
from app.utils.constants import size_of

REAL = [int(r["number"]) for r in csv.DictReader(open(Path(__file__).resolve().parents[1] / "data" / "wingo_transcribed.csv"))]
T0 = 1_800_000_000.0                                      # divisible by 30, 60 and 120
BASE = 20261029100010000
MODES = {"30s": 30, "1min": 60, "2min": 120}
HISTORY = 200


def make(P, offset=0.0, n=HISTORY):
    clock = [T0 + offset]
    svc = PredictorService(InMemoryRepository(), PredictionEngine(refit_every=10), 100, 300,
                           clock_cfg=ClockConfig(P, "UTC", "end", 5, 5, 20, True, offset), now_fn=lambda: clock[0])
    svc.import_rows([{"number": REAL[i], "period_id": str(BASE + i)} for i in range(n)])
    calls = []
    real_predict = svc.engine.predict
    svc.engine.predict = lambda seq: (calls.append(len(seq)), real_predict(seq))[1]     # spy: one entry per recalculation
    return svc, clock, calls


def view(p):
    """The parts of an answer that must not move while a cycle is running."""
    return (p["kind"], p["id"], p["created_at"], p["size"], p["top3"], p["target_label"], p["cycle"]["index"], p["n_history"])


def shown(p):
    """What the card displays as the prediction (the cycle counter alone is not a visible change)."""
    return (p["kind"], p["id"], p["created_at"], p["size"], p["top3"], p["target_label"], p["n_history"])


class TestNoResultsArrive(unittest.TestCase):
    """Feed late / unavailable: nothing verified arrives for many cycles."""

    def test_stable_in_cycle_recalculated_once_per_cycle_in_every_mode(self):
        for mode, P in MODES.items():
            with self.subTest(mode=mode):
                svc, clock, calls = make(P)
                first = svc.next_prediction()
                self.assertEqual((first["kind"], first["status"], first["unverified_cycles"]), ("pick", "pending", 0))
                stored_before = len(svc.repo.preds)
                n_cycles, seen, size0 = 8, [], first["size"]
                for c in range(n_cycles):
                    start = T0 + c * P
                    if c >= 1:                                                           # a recalculation is due once this cycle has settled
                        clock[0] = start + SETTLE_SECONDS + 0.6
                        ref = svc.next_prediction()
                        seen.append(ref)
                        self.assertEqual(ref["kind"], "projection")
                        for sec in range(int(SETTLE_SECONDS) + 1, P):                    # EVERY second of the rest of the countdown
                            clock[0] = start + sec
                            self.assertEqual(view(svc.next_prediction()), view(ref), f"{mode}: changed at +{sec}s of cycle {c}")
                self.assertEqual(len(calls), n_cycles - 1, "exactly one recalculation per completed cycle (the first cycle has the stored pick)")
                idx = [r["cycle"]["index"] for r in seen]
                self.assertEqual(idx, list(range(idx[0], idx[0] + len(idx))), "one answer per consecutive cycle")
                labels = [r["target_label"] for r in seen]
                self.assertEqual(labels, [str(BASE + HISTORY + c) for c in range(1, n_cycles)], "target round advances each cycle")
                self.assertEqual(len({r["created_at"] for r in seen}), len(seen), "each cycle has its own calculation time")
                self.assertEqual([r["unverified_cycles"] for r in seen], list(range(1, n_cycles)))
                self.assertTrue(all(r["id"] is None and r["status"] == "projection" and r["outcome"] is None for r in seen))
                self.assertEqual(len(svc.repo.preds), stored_before, "projections are never written to the database")
                self.assertEqual(svc.live_metrics()["bs"]["n"], 0, "projections are never scored")
                self.assertTrue(all(r["size"] == size0 for r in seen), "same verified data -> same Big/Small (deterministic, not random)")
                self.assertTrue(all(r["n_history"] == HISTORY for r in seen))

    def test_calculation_is_deterministic_across_independent_services(self):
        a, ca, _ = make(60); b, cb, _ = make(60)
        for clock in (ca, cb): clock[0] = T0 + 3 * 60 + 9
        pa, pb = a.next_prediction(), b.next_prediction()
        self.assertEqual((pa["size"], pa["top3"], pa["size_probs"]), (pb["size"], pb["top3"], pb["size_probs"]))

    def test_settling_window_then_projection(self):
        svc, clock, calls = make(60)
        first = svc.next_prediction()
        clock[0] = T0 + 60 + 1                                                           # 1 s after the boundary: result is "a moment away"
        p = svc.next_prediction()
        self.assertEqual((p["kind"], p["settling"], p["id"], p["created_at"]), ("pick", True, first["id"], first["created_at"]))
        self.assertEqual(calls, [], "no recalculation while waiting for the ended round's result")
        self.assertAlmostEqual(p["refresh_in_ms"], (SETTLE_SECONDS - 1 + 0.5) * 1000, delta=2)
        clock[0] = T0 + 60 + SETTLE_SECONDS + 0.5
        q = svc.next_prediction()
        self.assertEqual((q["kind"], q["settling"], q["unverified_cycles"]), ("projection", False, 1))
        self.assertEqual(len(calls), 1)

    def test_refresh_in_ms_is_exactly_when_the_answer_next_changes(self):
        """The browser re-reads once, after refresh_in_ms. Before that moment the answer must not change; at it, it must."""
        for mode, P in MODES.items():
            with self.subTest(mode=mode):
                svc, clock, _ = make(P)
                svc.next_prediction()
                for t in (3.0, P / 2, P - 0.4, P + 1, P + 7, 2 * P + 2, 3 * P + P / 2):   # arbitrary moments across several cycles
                    clock[0] = T0 + t; a = svc.next_prediction()
                    due = a["refresh_in_ms"] / 1000.0                  # includes a 0.5 s safety margin after the true change point
                    self.assertGreater(due, 0)
                    clock[0] = T0 + t + due - 0.5 - 0.05
                    self.assertEqual(shown(svc.next_prediction()), shown(a), f"{mode}: changed before its change point at t={t}")
                    clock[0] = T0 + t + due + 0.01
                    self.assertNotEqual(shown(svc.next_prediction()), shown(a), f"{mode}: not changed by refresh_in_ms at t={t}")

    def test_cycle_boundary_follows_the_countdown_phase_not_the_utc_minute(self):
        P, off = 60, 7.0                                                                # the game's rounds end at :07 of every minute
        svc, clock, _ = make(P, offset=off)
        svc.next_prediction()
        clock[0] = T0 + off + P - 0.5; before = svc.next_prediction()                    # 0.5 s before THIS game's boundary
        clock[0] = T0 + off + P + SETTLE_SECONDS + 0.6; after = svc.next_prediction()
        self.assertEqual(after["cycle"]["index"], before["cycle"]["index"] + 1)
        clock[0] = T0 + P + 1                                                             # a UTC minute boundary, NOT this game's
        self.assertEqual(svc.next_prediction()["cycle"]["index"], before["cycle"]["index"], "must not roll at the UTC minute")

    def test_unavailable_until_enough_verified_history_nothing_invented(self):
        for mode, P in MODES.items():
            with self.subTest(mode=mode):
                svc, clock, calls = make(P, n=40)
                for t in (1, P + 8, 5 * P + 8):
                    clock[0] = T0 + t
                    p = svc.next_prediction()
                    self.assertFalse(p["available"]); self.assertNotIn("size", p)
                self.assertEqual(calls, [])


class TestResultsArrive(unittest.TestCase):
    """The normal live path: the verified result of each ended round arrives a couple of seconds after the boundary."""

    def test_every_cycle_gets_a_new_scored_pick_in_every_mode(self):
        for mode, P in MODES.items():
            with self.subTest(mode=mode):
                svc, clock, calls = make(P)
                shown, ids, n_hist = svc.next_prediction(), set(), []
                for c in range(12):
                    start = T0 + c * P
                    for sec in (1 if c else 8, P / 2, P - 1):                            # during the countdown: nothing moves
                        if c == 0 and sec == 1: continue
                        clock[0] = start + sec
                        self.assertEqual(view(svc.next_prediction()), view(shown), f"{mode}: moved during cycle {c}")
                    i = HISTORY + c
                    clock[0] = start + P + 2                                              # verified result of the ended round arrives
                    out = svc.add_result(REAL[i], str(BASE + i))
                    res = out["resolved_prediction"]
                    self.assertEqual(res["id"], shown["id"], "scored against the pick made for THAT round")
                    self.assertEqual(res["target_period"], str(BASE + i))
                    self.assertEqual((res["outcome"]["actual"], res["outcome"]["bs_hit"]), (REAL[i], shown["size"] == size_of(REAL[i])))
                    clock[0] = start + P + 3
                    shown = svc.next_prediction()
                    self.assertEqual((shown["kind"], shown["unverified_cycles"], shown["settling"], shown["status"]), ("pick", 0, False, "pending"))
                    self.assertEqual(shown["target_period"], str(BASE + i + 1))
                    self.assertNotIn(shown["id"], ids); ids.add(shown["id"]); n_hist.append(shown["n_history"])
                self.assertEqual(n_hist, list(range(HISTORY + 1, HISTORY + 13)), "each pick is built from one more verified round")
                self.assertEqual(svc.live_metrics()["bs"]["n"], 12, "exactly one scored pick per cycle")
                self.assertEqual(calls, [HISTORY + c + 1 for c in range(12)], "one recalculation per cycle, no extras")
                live = [r for r in svc.repo.preds if r["status"] != "void"]
                self.assertEqual(len(live), 13); self.assertEqual(len({r["target_seq"] for r in live}), 13, "no duplicate pick for a round")

    def test_value_changes_only_through_the_data_and_matches_a_fresh_calculation(self):
        svc, clock, _ = make(60, n=400)
        ref = PredictionEngine(refit_every=10); ref.predict(REAL[:400])                # replay the service's own call sequence
        sizes = []
        for c in range(25):
            i = 400 + c
            clock[0] = T0 + c * 60 + 62
            svc.add_result(REAL[i], str(BASE + i))
            clock[0] = T0 + c * 60 + 63
            p = svc.next_prediction()
            fresh = ref.predict(REAL[:i + 1])                                            # independent engine, same verified history
            self.assertEqual(p["size"], fresh["size"], f"cycle {c}: displayed pick != independent calculation on the same verified data")
            sizes.append(p["size"])
        self.assertEqual({"Big", "Small"}, set(sizes), "over real data the pick does move when the data moves")

    def test_late_result_after_projections_scores_the_original_pick_then_projection_disappears(self):
        svc, clock, _ = make(60)
        stored = svc.next_prediction()
        clock[0] = T0 + 4 * 60 + 20                                                       # 4 cycles with no result
        proj = svc.next_prediction()
        self.assertEqual((proj["kind"], proj["unverified_cycles"], proj["id"]), ("projection", 4, None))
        self.assertEqual(proj["latest_result"]["number"], REAL[HISTORY - 1], "verified result shown is still the last real one")
        out = svc.add_result(REAL[HISTORY], str(BASE + HISTORY))                         # the late result of round #HISTORY finally arrives
        self.assertEqual(out["resolved_prediction"]["id"], stored["id"])                 # scored against the original, not the projection
        clock[0] += 1
        nxt = svc.next_prediction()
        # the new stored pick is made NOW (current cycle), from the data including the late result; the projection is gone
        self.assertEqual((nxt["kind"], nxt["unverified_cycles"], nxt["n_history"]), ("pick", 0, HISTORY + 1))
        self.assertEqual(nxt["target_period"], str(BASE + HISTORY + 1))
        self.assertEqual(svc.live_metrics()["bs"]["n"], 1)

    def test_predictions_and_verified_results_stay_separate(self):
        svc, clock, _ = make(30)
        clock[0] = T0 + 3 * 30 + 9
        p = svc.next_prediction()
        self.assertEqual(p["kind"], "projection")
        self.assertIsNone(p["outcome"]); self.assertEqual(p["status"], "projection")
        self.assertEqual(p["latest_result"]["period_id"], str(BASE + HISTORY - 1))        # verified data is its own field
        self.assertTrue(all("size" not in r and "top1" not in r for r in svc.repo.results))
        self.assertIn("Walk-forward", p["disclaimer"])                                    # the honesty disclaimer travels with every answer


if __name__ == "__main__":
    unittest.main()
