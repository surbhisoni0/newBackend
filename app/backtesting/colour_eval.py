"""Evaluation of class probabilities from walk-forward predictions: accuracy, Brier, log loss, per-class precision/recall,
confusion matrix, calibration. `class_eval` is generic; colour_eval (Green/Red/Violet vs 40/40/20) and size_eval (Small/Big
vs a fair 50/50 coin) are thin wrappers."""
import numpy as np
from app.utils.constants import COLOR_NAMES

COLOUR_PRIOR = np.array([0.4, 0.4, 0.2])
SIZE_PRIOR = np.array([0.5, 0.5])
SIZE_NAMES = ("Small", "Big")
PRIOR = COLOUR_PRIOR                                    # kept for older imports


def _ratio(a, b):
    return float(a / b) if b else None


def class_eval(probs, groups, prior, classes, n_bins: int = 5) -> dict:
    k = len(prior)
    P = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    g = np.asarray(groups, dtype=int)
    n = len(g)
    Y = np.eye(k)[g]
    pick = P.argmax(axis=1)
    brier, brier_prior = float(((P - Y) ** 2).sum(axis=1).mean()), float(((prior - Y) ** 2).sum(axis=1).mean())
    ll, ll_prior = float(-np.log(P[np.arange(n), g]).mean()), float(-np.log(prior[g]).mean())
    cm = np.zeros((k, k), dtype=int)                       # rows = actual class, columns = predicted class
    for a, p in zip(g, pick):
        cm[a, p] += 1
    conf_all = P.max(axis=1)
    bins = []
    for idx in np.array_split(np.argsort(conf_all, kind="stable"), n_bins):
        if len(idx):
            bins.append({"n": int(len(idx)), "mean_confidence": float(conf_all[idx].mean()),
                         "observed_accuracy": float((pick[idx] == g[idx]).mean())})
    return {"n": int(n), "classes": list(classes), "accuracy": float((pick == g).mean()),
            "brier": brier, "brier_prior": brier_prior, "brier_skill": 1 - brier / brier_prior,   # > 0 means better than the prior
            "logloss": ll, "logloss_prior": ll_prior, "logloss_diff": ll - ll_prior,               # < 0 means better than the prior
            "confusion": cm.tolist(),
            "precision": [_ratio(cm[c, c], cm[:, c].sum()) for c in range(k)],
            "recall": [_ratio(cm[c, c], cm[c, :].sum()) for c in range(k)],
            "support": [int(cm[c, :].sum()) for c in range(k)],
            "calibration": bins, "ece": float(sum(b["n"] / n * abs(b["observed_accuracy"] - b["mean_confidence"]) for b in bins))}


def colour_eval(probs, groups, n_bins: int = 5) -> dict:
    return class_eval(probs, groups, COLOUR_PRIOR, COLOR_NAMES, n_bins)


def size_eval(probs, groups, n_bins: int = 5) -> dict:
    return class_eval(probs, groups, SIZE_PRIOR, SIZE_NAMES, n_bins)
