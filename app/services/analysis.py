"""Statistical analysis / data-quality report for the stored history."""
import numpy as np
from scipy import stats
from app.utils.constants import COLOR_DETAIL, COLOR_GROUP, COLOR_NAMES


def analyze(numbers, meta=None) -> dict:
    s = np.asarray(numbers, dtype=int); n = len(s)
    if n < 30:
        return {"n": n, "message": "Need at least 30 results for analysis."}
    counts = np.bincount(s, minlength=10)
    chi = stats.chisquare(counts)
    T = np.zeros((10, 10))
    for a, b in zip(s[:-1], s[1:]): T[a, b] += 1
    c2, p_tr, dof, exp = stats.chi2_contingency(T)
    big = (s >= 5).astype(int)
    runs = 1 + int((big[1:] != big[:-1]).sum()); p1 = big.mean()
    er = 2 * n * p1 * (1 - p1) + 1; sd = np.sqrt(max((er - 1) * (er - 2) / (n - 1), 1e-12)); z = (runs - er) / sd
    rep = int((s[1:] == s[:-1]).sum())
    band = 1.96 / np.sqrt(n)
    ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    grp = np.bincount(COLOR_GROUP[s], minlength=3)
    gaps = {}
    for d in range(10):
        idx = np.where(s == d)[0]; gaps[d] = int(n - 1 - idx[-1]) if len(idx) else None
    def streak():
        best = cur = 1
        for a, b in zip(s[:-1], s[1:]):
            cur = cur + 1 if a == b else 1; best = max(best, cur)
        return best
    tests = {
        "uniformity": {"name": "Chi-square: numbers 0-9 equally likely", "p": float(chi.pvalue)},
        "transitions": {"name": "Chi-square: next number independent of previous", "p": float(p_tr),
                        "note": f"{int((exp < 5).sum())} of 100 cells have expected count < 5 (approximate test)"},
        "runs_big_small": {"name": "Runs test: Big/Small sequence random", "p": float(2 * (1 - stats.norm.cdf(abs(z))))},
        "repeat_rate": {"name": "Immediate repeat rate = 10%", "p": float(stats.binomtest(rep, n - 1, 0.1).pvalue),
                        "value": rep / (n - 1)},
        "autocorr_lag1": {"name": "Lag-1 autocorrelation within +-1.96/sqrt(n)", "value": ac1, "band": float(band),
                          "within_band": bool(abs(ac1) <= band)},
    }
    for t in tests.values():
        if "p" in t: t["verdict"] = "unusual - verify data" if t["p"] < 0.05 else "consistent with random"
    q = {"total": n, "unique_periods": None, "missing_period": None, "missing_timestamp": None,
         "duplicate_periods": None, "chronological": None, "invalid_numbers": 0}
    if meta:
        ps = [m["period_id"] for m in meta if m["period_id"]]
        q.update(missing_period=n - len(ps), unique_periods=len(set(ps)), duplicate_periods=len(ps) - len(set(ps)),
                 missing_timestamp=sum(1 for m in meta if not m["timestamp"]))
        nums = [int(p) for p in ps if p.isdigit()]
        q["chronological"] = bool(all(a < b for a, b in zip(nums[:-1], nums[1:]))) if len(nums) > 1 else None
    windows = {str(w): (np.bincount(s[-w:], minlength=10) / min(w, n)).round(4).tolist() for w in (20, 50, 100)}
    return {"n": n, "counts": counts.tolist(), "frequency": (counts / n).round(4).tolist(),
            "big_share": float(big.mean()), "big_small": {"Small": int((s < 5).sum()), "Big": int((s >= 5).sum())},
            "color_groups": {COLOR_NAMES[g]: int(grp[g]) for g in range(3)},
            "color_detail": {k: int(sum(1 for x in s if COLOR_DETAIL[x] == k)) for k in sorted(set(COLOR_DETAIL.values()))},
            "recent_frequency": windows, "current_gap": gaps, "max_same_number_streak": streak(),
            "tests": tests, "data_quality": q,
            "interpretation": ("Every test is consistent with a fair random draw." if all(t.get("p", 1) >= 0.05 for t in tests.values())
                               else "At least one test is unusual. Check for data-entry errors before reading anything into it; "
                                    "with several tests, one p<0.05 by chance is expected.")}
