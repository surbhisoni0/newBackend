"""Big/Small history analysis: rolling shares of Big vs a fair 50 %, transitions, streaks. Big = 5-9, Small = 0-4."""
import numpy as np
from scipy import stats

WINDOWS = (10, 25, 50, 100, 500)
CLASSES = ("Small", "Big")


def size_analysis(numbers) -> dict:
    g = (np.asarray(numbers, dtype=int) >= 5).astype(int)          # 0 = Small, 1 = Big
    n = len(g)
    if n < 30:
        return {"n": n, "message": "Need at least 30 results for Big/Small analysis."}
    counts = np.bincount(g, minlength=2)
    windows = {}
    for w in WINDOWS:
        if n >= w:
            big = int(g[-w:].sum())
            windows[str(w)] = {"n": w, "share": [round((w - big) / w, 4), round(big / w, 4)],
                               "p_vs_fair": float(stats.binomtest(big, w, 0.5).pvalue)}
    trans = np.zeros((2, 2), dtype=int)
    np.add.at(trans, (g[:-1], g[1:]), 1)
    rows = trans.sum(axis=1)
    probs = [(trans[i] / rows[i]).round(4).tolist() if rows[i] else None for i in range(2)]
    p_ind = float(stats.chi2_contingency(trans)[1]) if n >= 60 and (rows > 0).all() and (trans.sum(axis=0) > 0).all() else None
    longest, run = [1, 1], 1
    for i in range(1, n):
        run = run + 1 if g[i] == g[i - 1] else 1
        longest[g[i]] = max(longest[g[i]], run)
    cur = 1
    while cur < n and g[n - 1 - cur] == g[n - 1]:
        cur += 1
    return {"n": n, "classes": list(CLASSES), "overall_share": (counts / n).round(4).tolist(), "expected_share": [0.5, 0.5],
            "windows": windows, "transitions": {"counts": trans.tolist(), "probabilities": probs, "p_independent": p_ind},
            "streaks": {"current": {"colour": CLASSES[int(g[-1])], "length": int(cur)}, "longest": dict(zip(CLASSES, map(int, longest)))}}
