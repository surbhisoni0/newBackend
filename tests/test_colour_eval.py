import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.backtesting.colour_eval import PRIOR, colour_eval
from app.prediction.colour import colour_probs


class TestColourEval(unittest.TestCase):
    def test_prior_predictions_have_zero_skill(self):
        g = np.array([0, 1, 0, 2, 1, 0, 1, 0, 2, 1])
        e = colour_eval(np.tile(PRIOR, (10, 1)), g)
        self.assertAlmostEqual(e["brier_skill"], 0.0, places=6); self.assertAlmostEqual(e["logloss_diff"], 0.0, places=6)

    def test_perfect_predictions_and_confusion_matrix(self):
        g = np.array([0, 1, 2, 0, 1, 2]); P = np.eye(3)[g] * 0.9 + 0.05
        e = colour_eval(P, g)
        self.assertEqual(e["accuracy"], 1.0); self.assertEqual(e["confusion"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        self.assertEqual(e["precision"], [1.0, 1.0, 1.0]); self.assertGreater(e["brier_skill"], 0.5)

    def test_wrong_confident_predictions_are_worse_than_prior(self):
        g = np.array([0, 1, 0, 1]); P = np.array([[0.05, 0.9, 0.05]] * 4)
        e = colour_eval(P, g); self.assertLess(e["brier_skill"], 0); self.assertGreater(e["logloss_diff"], 0)
        self.assertIsNone(e["precision"][0])

    def test_colour_probs_sum_to_one(self):
        s = {"a": np.ones(10) / 10, "b": np.arange(1, 11) / 55}
        self.assertAlmostEqual(float(colour_probs(s).sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
