"""Direct vs Reverse: the maths, the no-look-ahead guarantee, and the API/service wiring.

Synthetic hit sequences appear ONLY to exercise code paths and prove properties (e.g. "changing the future cannot change today's choice").
Every reported accuracy in the dashboard comes from real walk-forward predictions on the project's own history.
"""
import csv
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.backtesting import direct_reverse as dr
from app.backtesting.walk_forward import run_walk_forward
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.timing import ClockConfig

REAL = [int(r["number"]) for r in csv.DictReader(open(Path(__file__).resolve().parents[1] / "data" / "wingo_docx_clean.csv"))]


def rec(seq, pick, actual, models=None):
    side = "Big" if actual >= 5 else "Small"
    return {"seq": seq, "date": None, "pick": pick, "actual_side": side, "hit": pick == side, "models": models or {}, "p_big": None}


class Maths(unittest.TestCase):
    def test_reverse_is_exact_complement(self):
        h = [True, False, True, True, False]
        b = dr.both(h)
        self.assertEqual(b["direct"]["correct"], 3); self.assertEqual(b["reverse"]["correct"], 2)
        self.assertAlmostEqual(b["direct"]["rate"] + b["reverse"]["rate"], 1.0)
        self.assertEqual(b["direct"]["n"], b["reverse"]["n"])

    def test_flip(self):
        self.assertEqual(dr.flip("Big"), "Small"); self.assertEqual(dr.flip("Small"), "Big")

    def test_verdict_labels(self):
        self.assertFalse(dr.verdict(2520, 5000)["significant"])               # 50.4% on 5000 rounds is noise
        self.assertTrue(dr.verdict(2300, 5000)["significant"])                # 46% on 5000 rounds is not
        self.assertIn("WORSE", dr.verdict(2300, 5000)["label"])
        self.assertTrue(dr.verdict(2425, 5000)["significant"])                # 48.5%: p~0.034 on its own...
        self.assertFalse(dr.verdict(2425, 5000, n_tests=10)["significant"])   # ...but not after correcting for 10 looks

    def test_reachability_of_75_percent(self):
        r = dr.target_reachability(4986, 0.75)
        self.assertGreater(r["z_score_of_target"], 30); self.assertLess(r["p_by_luck_alone"], 1e-100)

    def test_unverified_rows_never_scored(self):
        rows = [{"target_seq": 1, "actual": None, "size": "Big"}, {"target_seq": 2, "actual": 7, "size": "Big"}]
        self.assertEqual(len(dr.build_records(rows)), 1)


class NoLookAhead(unittest.TestCase):
    def test_choice_i_ignores_rounds_i_and_later(self):
        rng = np.random.RandomState(1)
        h = list(rng.rand(900) < 0.42)                                       # a deliberately bad model so the rule fires
        base = dr.adaptive_choices(h, window=100, z=2.0, min_history=100)
        self.assertIn("reverse", base)
        for i in (150, 400, 777):
            h2 = h[:i] + [not x for x in h[i:]]                              # flip EVERY outcome from round i on
            self.assertEqual(dr.adaptive_choices(h2, 100, 2.0, 100)[:i + 1], base[:i + 1], f"choice at {i} changed when the future changed")

    def test_next_choice_uses_only_given_history(self):
        h = [False] * 400
        self.assertEqual(dr.next_choice(h, 100, 2.0, 100)["strategy"], "reverse")
        self.assertEqual(dr.next_choice([True] * 400, 100, 2.0, 100)["strategy"], "direct")
        self.assertEqual(dr.next_choice([False] * 50, 100, 2.0, 100)["strategy"], "direct")   # too little history -> default

    def test_holdout_chooses_on_train_only(self):
        r = [rec(i, "Big", 1 if i < 300 else 9) for i in range(1, 501)]       # wrong in the first 60%, right afterwards
        h = dr.holdout(r, 0.6)
        self.assertEqual(h["chosen_on_train"], "reverse")                      # chosen from the past...
        self.assertEqual(h["chosen_strategy_on_test"]["rate"], 0.0)            # ...and then punished by the unseen future: no peeking

    def test_selector(self):
        r = [rec(i, "Big", 9) for i in range(1, 11)]
        self.assertEqual(len(dr.select(r, seq_from=3, seq_to=5)), 3)


class WalkForwardRows(unittest.TestCase):
    def test_rows_keep_old_keys_and_add_per_model_picks(self):
        out = run_walk_forward(np.array(REAL[:260]), start=100, use_ml=False)
        r = out["rows"][0]
        for k in ("target_seq", "actual", "size", "bs_hit", "top1", "color"):
            self.assertIn(k, r)
        self.assertEqual(set(r["size_by_model"]), {"CumFreq", "Recent50", "Markov1", "Markov2"})
        self.assertEqual(r["actual"], REAL[r["target_seq"] - 1])                # result mapping: row t+1 is scored against seq[t]

    def test_future_cannot_change_a_past_prediction(self):
        a = run_walk_forward(np.array(REAL[:200]), start=100, use_ml=False)["rows"]
        mutated = np.array(REAL[:200]); mutated[150:] = (mutated[150:] + 5) % 10
        b = run_walk_forward(mutated, start=100, use_ml=False)["rows"]
        self.assertEqual([(r["size"], r["p_big"]) for r in a[:50]], [(r["size"], r["p_big"]) for r in b[:50]])   # rounds 101..150 unaffected


class ServiceWiring(unittest.TestCase):
    def setUp(self):
        self.svc = PredictorService(InMemoryRepository(), PredictionEngine(refit_every=10, use_ml=False), 100, 300,
                                    clock_cfg=ClockConfig(60, "UTC", "end", 5, 5, 20, True, 0.0), now_fn=lambda: 1_800_000_000.0)
        base = 20261029100010000
        self.svc.import_rows([{"number": REAL[i], "period_id": str(base + i)} for i in range(300)])
        self.svc.repo.save_backtest({"data_n": 300}, run_walk_forward(np.array(REAL[:300]), 100, use_ml=False)["rows"])

    def test_direct_reverse_report(self):
        d = self.svc.direct_reverse()
        self.assertTrue(d["available"])
        s = d["summary"]
        self.assertEqual(s["direct"]["n"], 200); self.assertEqual(s["direct"]["correct"] + s["reverse"]["correct"], 200)
        self.assertTrue(d["has_algorithm_detail"]); self.assertEqual(len(d["algorithms"]), 5)

    def test_range_selector(self):
        d = self.svc.direct_reverse(seq_from=151, seq_to=250)
        self.assertEqual(d["range"]["n"], 100)
        self.assertEqual(self.svc.direct_reverse(date_from="2026-10-29", date_to="2026-10-29")["range"]["n"], 200)

    def test_modes_flip_original_pick_without_changing_it(self):
        direct = self.svc.strategy_prediction("direct"); rev = self.svc.strategy_prediction("reverse")
        self.assertEqual(direct["final_pick"], direct["original_pick"])
        self.assertEqual(rev["final_pick"], dr.flip(rev["original_pick"]))
        self.assertEqual(direct["original_pick"], self.svc.next_prediction()["size"])      # existing prediction path untouched
        best = self.svc.strategy_prediction("best")
        self.assertIn(best["strategy_used"], ("direct", "reverse")); self.assertTrue(best["selection"]["reason"])
        self.assertEqual(best["history_used"], 200)                                          # only rounds resolved before the target

    def test_bad_strategy_rejected(self):
        with self.assertRaises(Exception):
            self.svc.strategy_prediction("sideways")


if __name__ == "__main__":
    unittest.main()
