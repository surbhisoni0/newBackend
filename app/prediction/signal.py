"""Signal-strength evaluator.

The signal label is derived from VALIDATION EVIDENCE about the consensus ranking, not from how "confident"
any model looks. Rules (all pre-specified, applied to walk-forward hit flags):

  a) Bonferroni-corrected one-sided binomial p < 0.05 on at least one metric (K = models x metrics tested)
  b) Wilson 95% lower bound above the metric's baseline
  c) hit rate above baseline in at least 3 of 4 chronological blocks
  d) confirmed on rounds that arrived AFTER the backtest (live n >= LIVE_MIN_N, live rate above baseline, live p < 0.05)

  STRONG   = a+b+c+d      MODERATE = a+b+c      WEAK = nominal p < 0.05 with b+c (does not survive correction)
  otherwise NO STRONG SIGNAL
"""
from typing import Optional
from app.utils.constants import METRICS

LIVE_MIN_N = 100
LABEL = {"strong": "Strong", "moderate": "Moderate", "weak": "Weak", "none": "NO STRONG SIGNAL"}


def evaluate_signal(validation: Optional[dict], live: Optional[dict], min_n: int = 300, alpha: float = 0.05,
                    metrics: tuple = ("bs",)) -> dict:
    """validation: {'metrics': {metric: rate_stats}, 'K': int}; live: {metric: rate_stats} or None."""
    base = {"level": "none", "label": LABEL["none"], "reasons": [], "evidence": {}}
    if not validation or not validation.get("metrics"):
        base["reasons"].append("No walk-forward validation has been run yet.")
        return base
    K = max(1, int(validation.get("K", 4)))
    best_level, rank = "none", {"none": 0, "weak": 1, "moderate": 2, "strong": 3}
    for m in metrics:                          # default: Big/Small only; K below still counts every metric tested in the backtest
        s = validation["metrics"].get(m)
        if not s or s["n"] < min_n:
            base["reasons"].append(f"{m}: fewer than {min_n} validation predictions.")
            continue
        p_corr = min(1.0, s["p"] * K)
        above = s["lo"] > s["baseline"]
        blocks = s.get("blocks") or []
        consistent = len(blocks) >= 4 and sum(b > s["baseline"] for b in blocks) >= 3
        ev = {"rate": s["rate"], "lo": s["lo"], "baseline": s["baseline"], "p": s["p"], "p_corrected": p_corr,
              "ci_above_baseline": above, "blocks_consistent": consistent}
        lv = (live or {}).get(m)
        live_ok = bool(lv and lv["n"] >= LIVE_MIN_N and lv["rate"] > lv["baseline"] and lv["p"] < alpha)
        ev["live_confirmed"] = live_ok
        base["evidence"][m] = ev
        level = "none"
        if p_corr < alpha and above and consistent:
            level = "strong" if live_ok else "moderate"
        elif s["p"] < alpha and above and consistent:
            level = "weak"
        if rank[level] > rank[best_level]:
            best_level = level
    base["level"], base["label"] = best_level, LABEL[best_level]
    if best_level == "none":
        base["reasons"].append("No metric beats its random baseline after accounting for multiple testing and time-consistency.")
    return base
