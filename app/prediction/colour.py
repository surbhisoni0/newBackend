"""Colour probabilities. The mean of the models' normalised number-probabilities, summed per colour group.
This is an uncertain statistical estimate; backtesting/colour_eval.py measures whether it beats the 40/40/20 prior."""
import numpy as np
from app.utils.constants import COLOR_GROUP, COLOR_NAMES


def colour_probs(scores: dict) -> np.ndarray:
    mean = np.mean([p / p.sum() for p in scores.values()], axis=0)
    return np.array([mean[COLOR_GROUP == g].sum() for g in range(3)])


def as_dict(p: np.ndarray) -> dict:
    return {name: round(float(x), 4) for name, x in zip(COLOR_NAMES, p)}
