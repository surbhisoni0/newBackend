"""Colour-level history analysis: rolling shares, transition matrix, streaks, and tests against the fair-game shares.
Colour groups follow the game: Green = 1,3,7,9 | Red = 2,4,6,8 | Violet = 0 and 5 (0 also red, 5 also green)."""
import numpy as np
from scipy import stats
from app.utils.constants import COLOR_GROUP, COLOR_NAMES

WINDOWS = (10, 25, 50, 100, 500)
FAIR = np.array([0.4, 0.4, 0.2])


def colour_analysis(numbers) -> dict:
    g = COLOR_GROUP[np.asarray(numbers, dtype=int)]
    n = len(g)
    if n < 30:
        return {"n": n, "message": "Need at least 30 results for colour analysis."}
    counts = np.bincount(g, minlength=3)
    windows = {}
    for w in WINDOWS:
        if n >= w:
            c = np.bincount(g[-w:], minlength=3)
            p = float(stats.chisquare(c, FAIR * w).pvalue) if w >= 25 else None       # needs expected counts >= 5
            windows[str(w)] = {"n": w, "share": (c / w).round(4).tolist(), "p_vs_fair": p}
    trans = np.zeros((3, 3), dtype=int)
    np.add.at(trans, (g[:-1], g[1:]), 1)
    rows = trans.sum(axis=1)
    probs = [(trans[i] / rows[i]).round(4).tolist() if rows[i] else None for i in range(3)]
    p_ind = float(stats.chi2_contingency(trans)[1]) if n >= 60 and (rows > 0).all() and (trans.sum(axis=0) > 0).all() else None
    longest, run = [1, 1, 1], 1
    for i in range(1, n):
        run = run + 1 if g[i] == g[i - 1] else 1
        longest[g[i]] = max(longest[g[i]], run)
    cur = 1
    while cur < n and g[n - 1 - cur] == g[n - 1]:
        cur += 1
    return {"n": n, "classes": list(COLOR_NAMES), "overall_share": (counts / n).round(4).tolist(), "expected_share": FAIR.tolist(),
            "windows": windows,
            "transitions": {"counts": trans.tolist(), "probabilities": probs, "p_independent": p_ind},
            "streaks": {"current": {"colour": COLOR_NAMES[int(g[-1])], "length": int(cur)}, "longest": dict(zip(COLOR_NAMES, map(int, longest)))}}
