"""Big/Small probabilities. Big = 5-9, Small = 0-4. P(Big) is the mean, over all models, of the probability each model puts
on the numbers 5-9. It is an uncertain statistical estimate; backtesting/colour_eval.py measures whether it beats a fair coin."""
import numpy as np

SIZE_NAMES = ("Small", "Big")


def size_probs(scores: dict) -> np.ndarray:
    mean = np.mean([p / p.sum() for p in scores.values()], axis=0)
    return np.array([mean[:5].sum(), mean[5:].sum()])


def size_dict(p: np.ndarray) -> dict:
    return {"Small": round(float(p[0]), 4), "Big": round(float(p[1]), 4)}


def size_pick(p: np.ndarray) -> str:
    return "Big" if p[1] > p[0] else "Small"            # an exact tie goes to Small
