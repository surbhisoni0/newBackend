"""Statistical baselines. Each returns a length-10 score vector for the next number from history h only."""
import numpy as np


def _tb(t: int) -> np.ndarray:  # deterministic, tiny tie-break
    return np.random.RandomState(t).rand(10) * 1e-9


def cum_freq(h):
    return np.bincount(h, minlength=10) / len(h) + _tb(len(h))


def recent_freq(h, w: int = 50):
    return np.bincount(h[-w:], minlength=10) / min(w, len(h)) + _tb(len(h))


def markov1(h):
    C = np.ones((10, 10))
    for a, b in zip(h[:-1], h[1:]):
        C[a, b] += 1
    r = C[h[-1]]
    return r / r.sum()


def markov2(h, k: float = 5.0):
    """Order-2 transitions, backing off to order-1 with pseudo-count k."""
    lo = np.ones((10, 10))
    C = {}
    for i in range(1, len(h)):
        lo[h[i - 1], h[i]] += 1
        if i >= 2:
            C.setdefault((h[i - 2], h[i - 1]), np.zeros(10))[h[i]] += 1
    lo = lo[h[-1]] / lo[h[-1]].sum()
    c = C.get((h[-2], h[-1]), np.zeros(10)) if len(h) >= 2 else np.zeros(10)
    return (c + k * lo) / (c.sum() + k)
