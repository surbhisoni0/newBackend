"""Run from project root:  PYTHONPATH=backend python -m unittest discover -s tests -v
Note: tests that need arbitrary sequences use tiny synthetic arrays ONLY to exercise code paths; no test result
or dataset stored by the app is synthetic."""
import csv
import unittest
from pathlib import Path
import numpy as np

from app.backtesting.walk_forward import run_walk_forward
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine, consensus, decide, rank_scores
from app.prediction.features import feats
from app.prediction.signal import evaluate_signal
from app.services import analysis
from app.services.data_service import clean_number, clean_period, clean_rows, parse_csv
from app.services.errors import Conflict, ValidationFailure
from app.services.predictor_service import PredictorService

CSV = Path(__file__).resolve().parents[1] / "data" / "wingo_transcribed.csv"
REAL = [int(r["number"]) for r in csv.DictReader(open(CSV))]


def stat(rate, n=700, base=0.3, lo=None, p=None, blocks=None):
    return {"rate": rate, "n": n, "baseline": base, "lo": lo if lo is not None else rate - 0.03,
            "p": p if p is not None else 0.5, "blocks": blocks or [rate] * 4}


class TestData(unittest.TestCase):
    def test_real_dataset_shape(self):
        self.assertEqual(len(REAL), 802); self.assertTrue(all(0 <= x <= 9 for x in REAL))

    def test_number_validation(self):
        for bad in (10, -1, True, "a", 3.5, None):
            with self.assertRaises(ValidationFailure): clean_number(bad)
        self.assertEqual(clean_number("7"), 7); self.assertEqual(clean_number(0), 0)

    def test_period_validation_rejects_excel_corruption(self):
        with self.assertRaises(ValidationFailure): clean_period("2.03E+16")
        with self.assertRaises(ValidationFailure): clean_period("abc")
        self.assertEqual(clean_period("20260930100051234"), "20260930100051234")

    def test_chronology_and_duplicates(self):
        rows = [{"number": 1, "period_id": "100"}, {"number": 2, "period_id": "101"}]
        self.assertEqual(len(clean_rows(rows, None)), 2)
        with self.assertRaises(Conflict): clean_rows([{"number": 1, "period_id": "5"}, {"number": 2, "period_id": "5"}], None)
        with self.assertRaises(Conflict): clean_rows([{"number": 1, "period_id": "99"}], "100")
        with self.assertRaises(ValidationFailure): clean_rows([{"number": 1}, {"number": 12}], None)

    def test_csv_keeps_period_text(self):
        rows = parse_csv("period,number\n20260930100051234,4\n20260930100051235,9\n")
        self.assertEqual(rows[0]["period_id"], "20260930100051234")
        self.assertEqual(parse_csv("number\n1\n2\n", newest_first=True)[0]["number"], "2")


class TestEngine(unittest.TestCase):
    def test_features_use_only_history(self):
        h = np.array(REAL[:200]); a = feats(h)
        self.assertTrue(np.array_equal(a, feats(np.concatenate([h, [1, 2, 3]])[:200])))

    def test_no_lookahead(self):
        S2 = np.array(REAL); S2[300:] = (S2[300:] + 3) % 10
        a = PredictionEngine().predict(np.array(REAL)[:300]); b = PredictionEngine().predict(S2[:300])
        self.assertEqual(a["ranking"], b["ranking"])

    def test_output_structure(self):
        p = PredictionEngine().predict(np.array(REAL[:200]))
        self.assertEqual(len(set(p["top3"])), 3); self.assertEqual(len(p["ranking"]), 10)
        self.assertIn(p["size"], ("Big", "Small")); self.assertIn(p["color"], ("Green", "Red", "Violet"))
        self.assertEqual(set(p["model_tops"]), {"CumFreq", "Recent50", "Markov1", "Markov2", "LogReg", "RandForest", "HistGB"})
        self.assertNotIn("probability", p)             # scores are ranks, never presented as probabilities

    def test_rank_scores_and_consensus(self):
        r = rank_scores(np.arange(10.0)); self.assertEqual(r[9], 1.0); self.assertEqual(r[0], 0.0)
        c = consensus({"a": np.arange(10.0), "b": np.arange(10.0)}); self.assertEqual(decide(c)["top1"], 9)


class TestSignal(unittest.TestCase):
    def val(self, **kw): return {"metrics": {"top1": stat(.1, base=.1), "top3": stat(.3, base=.3), "bs": stat(.5, base=.5),
                                             "color": stat(.4, base=.4), **kw}, "K": 32}

    def test_none_without_validation(self):
        self.assertEqual(evaluate_signal(None, None)["label"], "NO STRONG SIGNAL")

    def test_none_when_at_baseline(self):
        self.assertEqual(evaluate_signal(self.val(), None)["label"], "NO STRONG SIGNAL")

    def test_none_when_too_few_predictions(self):
        v = self.val(bs=stat(.6, n=50, p=1e-9, lo=.55, base=.5)); self.assertEqual(evaluate_signal(v, None)["level"], "none")

    def test_moderate_then_strong_with_live_confirmation(self):
        good = stat(.6, p=1e-8, lo=.56, base=.5, blocks=[.59, .61, .58, .62])
        self.assertEqual(evaluate_signal(self.val(bs=good), None)["level"], "moderate")
        live = {"bs": {"n": 150, "rate": .6, "baseline": .5, "p": .001}}
        self.assertEqual(evaluate_signal(self.val(bs=good), live)["level"], "strong")

    def test_nominal_only_is_weak_not_moderate(self):
        nominal = stat(.545, p=0.03, lo=.51, base=.5, blocks=[.54, .55, .53, .56])   # p*32 > 0.05
        self.assertEqual(evaluate_signal(self.val(bs=nominal), None)["level"], "weak")

    def test_only_the_big_small_target_counts(self):
        strong_elsewhere = stat(.45, p=1e-9, lo=.41, base=.3, blocks=[.44, .46, .43, .47])
        self.assertEqual(evaluate_signal(self.val(top3=strong_elsewhere), None)["level"], "none")   # lucky Top-3 cannot set the label

    def test_inconsistent_blocks_rejected(self):
        flaky = stat(.6, p=1e-8, lo=.56, base=.5, blocks=[.7, .7, .3, .3])
        self.assertEqual(evaluate_signal(self.val(bs=flaky), None)["level"], "none")


class TestBacktestAndService(unittest.TestCase):
    def test_walk_forward_is_chronological(self):
        A = np.array(REAL[:260]); B = A.copy(); B[160:] = (B[160:] + 5) % 10        # same past, different future
        ra = run_walk_forward(A, start=100, use_ml=False)["rows"]; rb = run_walk_forward(B, start=100, use_ml=False)["rows"]
        for x, y in zip(ra[:59], rb[:59]):                                            # targets 101..159 only see the shared past
            self.assertEqual((x["top1"], x["top3"], x["size"], x["color"]), (y["top1"], y["top3"], y["size"], y["color"]))
        self.assertEqual(len(ra), 160)

    def make(self):
        return PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), min_history=100, min_validation_n=300)

    def test_full_flow_resolves_and_tracks_live_accuracy(self):
        s = self.make()
        out = s.import_rows([{"number": n} for n in REAL[:150]])
        nxt = out["next_prediction"]; self.assertEqual(nxt["target_seq"], 151); self.assertEqual(nxt["signal"]["label"], "NO STRONG SIGNAL")
        actual = REAL[150]
        r = s.add_result(actual)
        res = r["resolved_prediction"]["outcome"]
        self.assertEqual(res["actual"], actual); self.assertEqual(res["top1_hit"], nxt["top1"] == actual)
        self.assertEqual(res["top3_hit"], actual in nxt["top3"])
        self.assertEqual(r["next_prediction"]["target_seq"], 152)
        self.assertEqual(r["live_performance"]["top1"]["n"], 1)
        for n in REAL[151:161]: s.add_result(n)
        perf = s.performance(); self.assertEqual(perf["live"]["total_predictions"], 11)
        h = s.history(limit=5); self.assertEqual(h["predictions"][0]["result"], "PENDING"); self.assertGreaterEqual(h["total"], 12)

    def test_invalid_result_rejected_and_nothing_stored(self):
        s = self.make(); s.import_rows([{"number": n} for n in REAL[:120]])
        with self.assertRaises(ValidationFailure): s.add_result(11)
        with self.assertRaises(ValidationFailure): s.import_rows([{"number": 1}, {"number": "x"}])
        self.assertEqual(s.repo.last_result()["seq"], 120)

    def test_insufficient_history(self):
        s = self.make(); s.import_rows([{"number": n} for n in REAL[:40]])
        self.assertFalse(s.next_prediction()["available"])

    def test_backtest_storage_and_signal_reads_it(self):
        s = self.make(); s.import_rows([{"number": n} for n in REAL[:260]])
        summ = s.run_backtest(use_ml=False)
        self.assertEqual(summ["n_predictions"], 160); self.assertEqual(summ["K"], 5 * 4)
        sig = s.current_signal(); self.assertEqual(sig["label"], "NO STRONG SIGNAL")
        self.assertEqual(s.backtest_results()["state"]["status"], "done")

    def test_analysis_on_real_data(self):
        a = analysis.analyze(REAL); self.assertEqual(a["n"], 802); self.assertEqual(sum(a["counts"]), 802)
        self.assertIn("uniformity", a["tests"])


if __name__ == "__main__":
    unittest.main()
