"""Small statistics helpers (confidence intervals, exact binomial tests)."""
import math
import numpy as np
from scipy import stats


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_stats(hits, baseline: float, n_blocks: int = 0) -> dict:
    """Hit-rate summary with Wilson 95% CI and a one-sided exact binomial test vs `baseline`."""
    hits = np.asarray(hits, dtype=bool)
    n = int(len(hits)); k = int(hits.sum())
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None, "baseline": baseline, "p": None, "blocks": []}
    lo, hi = wilson(k, n)
    p = float(stats.binomtest(k, n, baseline, alternative="greater").pvalue)
    out = {"k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi, "baseline": baseline, "p": p, "blocks": []}
    if n_blocks and n >= n_blocks * 20:
        edges = np.linspace(0, n, n_blocks + 1).astype(int)
        out["blocks"] = [float(hits[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])]
    return out
