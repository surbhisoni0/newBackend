"""DIRECT vs REVERSE analysis.

Direct  = follow the model's Big/Small pick.      Reverse = play the opposite side.
Both are scored on exactly the same rounds, using only walk-forward predictions (each round predicted from earlier rounds only).

Big/Small has two outcomes and no ties, so reverse_hit == NOT direct_hit for every round. That has two consequences this module states
openly instead of hiding:
  * reverse accuracy is always exactly 100% - direct accuracy; "reverse beats direct" is the same statement as "direct is below 50%".
  * the only way reversal can help is if the model is systematically *worse* than a coin flip by more than chance allows.

Nothing here uses random or invented data. Every function takes plain lists / dicts so it can be unit-tested.
"""
import math
from typing import Optional, Sequence
import numpy as np
from scipy import stats
from app.utils.stats import wilson

SIDES = ("Big", "Small")
FLIP = {"Big": "Small", "Small": "Big"}
STRATEGIES = ("direct", "reverse", "best")


def flip(side: str) -> str:
    return FLIP[side]


# ----------------------------------------------------------------------------------------------------- basic rate maths
def rate(hits: Sequence) -> dict:
    """n / correct / incorrect / accuracy with a Wilson 95% interval."""
    a = np.asarray(hits, dtype=bool)
    n, k = int(a.size), int(a.sum())
    if n == 0:
        return {"n": 0, "correct": 0, "incorrect": 0, "rate": None, "lo": None, "hi": None}
    lo, hi = wilson(k, n)
    return {"n": n, "correct": k, "incorrect": n - k, "rate": k / n, "lo": lo, "hi": hi}


def both(direct_hits: Sequence) -> dict:
    """Direct and Reverse rates for the same rounds. Reverse = exact complement (binary outcome)."""
    d = np.asarray(direct_hits, dtype=bool)
    return {"direct": rate(d), "reverse": rate(~d)}


def binom_two_sided(k: int, n: int, p0: float = 0.5) -> Optional[float]:
    return None if n == 0 else float(stats.binomtest(int(k), int(n), p0, alternative="two-sided").pvalue)


def verdict(k_direct: int, n: int, alpha: float = 0.05, n_tests: int = 1) -> dict:
    """Is direct accuracy distinguishable from 50% (in either direction)? Bonferroni-corrected when several variants were looked at."""
    if n == 0:
        return {"p_two_sided": None, "p_corrected": None, "label": "no data"}
    p = binom_two_sided(k_direct, n)
    pc = min(1.0, p * max(1, n_tests))
    r = k_direct / n
    if pc >= alpha:
        label = "indistinguishable from a coin flip"
    else:
        label = "direct is significantly BETTER than 50%" if r > 0.5 else "direct is significantly WORSE than 50% (reverse is better)"
    return {"p_two_sided": p, "p_corrected": pc, "n_tests": n_tests, "alpha": alpha, "label": label,
            "significant": bool(pc < alpha)}


# ----------------------------------------------------------------------------------------------------- records
def build_records(rows: Sequence[dict], meta_by_seq: Optional[dict] = None) -> list:
    """Normalise walk-forward / live rows: {seq, date, pick, actual_side, hit, models:{name: pick}, p_big}."""
    out = []
    for r in rows:
        seq = int(r["target_seq"])
        actual = int(r["actual"]) if r.get("actual") is not None else None
        if actual is None:
            continue                                                       # unverified: never scored
        act_side = "Big" if actual >= 5 else "Small"
        pid = (meta_by_seq or {}).get(seq) or r.get("period_id")
        date = None
        if pid and len(str(pid)) >= 8 and str(pid)[:8].isdigit():
            s = str(pid)[:8]; date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        out.append({"seq": seq, "date": date, "pick": r["size"], "actual_side": act_side, "hit": r["size"] == act_side,
                    "models": r.get("size_by_model") or {}, "p_big": r.get("p_big")})
    out.sort(key=lambda x: x["seq"])
    return out


def select(records: list, seq_from=None, seq_to=None, date_from=None, date_to=None) -> list:
    """Date-range / sequence-range selector (dates are the YYYY-MM-DD embedded in the period id)."""
    def ok(r):
        if seq_from is not None and r["seq"] < seq_from: return False
        if seq_to is not None and r["seq"] > seq_to: return False
        if date_from and (not r["date"] or r["date"] < date_from): return False
        if date_to and (not r["date"] or r["date"] > date_to): return False
        return True
    return [r for r in records if ok(r)]


def data_quality(records: list, all_meta: Optional[Sequence[dict]] = None) -> dict:
    """Missing / duplicated / unverified accounting for the tested rounds."""
    seqs = [r["seq"] for r in records]
    dup = len(seqs) - len(set(seqs))
    gaps = 0
    if seqs:
        s = sorted(set(seqs)); gaps = (s[-1] - s[0] + 1) - len(s)
    q = {"predictions_tested": len(records), "duplicate_target_rounds": dup, "missing_sequence_numbers_in_range": gaps}
    if all_meta is not None:
        pids = [m.get("period_id") for m in all_meta]
        nz = [p for p in pids if p]
        q.update({"stored_rounds": len(all_meta), "rounds_without_period_id": len(all_meta) - len(nz),
                  "duplicate_period_ids": len(nz) - len(set(nz))})
        jumps = 0
        for a, b in zip(nz[:-1], nz[1:]):
            try:
                if int(b) - int(a) != 1: jumps += 1
            except ValueError:
                pass
        q["period_id_jumps"] = jumps                                       # >0 = skipped rounds in the source history
    return q


# ----------------------------------------------------------------------------------------------------- the comparison
def _blocks(records: list, n_blocks: int) -> list:
    n = len(records)
    if n < n_blocks * 20:
        return []
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    res = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = records[a:b]
        h = [r["hit"] for r in seg]
        res.append({"from_seq": seg[0]["seq"], "to_seq": seg[-1]["seq"], "date_from": seg[0]["date"], "date_to": seg[-1]["date"],
                    **both(h)})
    return res


def _drift(blocks: list) -> Optional[dict]:
    """Chi-square test that direct accuracy is the same in every chronological block (temporal drift)."""
    if len(blocks) < 2:
        return None
    ks = np.array([b["direct"]["correct"] for b in blocks]); ns = np.array([b["direct"]["n"] for b in blocks])
    tbl = np.array([ks, ns - ks])
    chi2, p, _, _ = stats.chi2_contingency(tbl)
    return {"chi2": float(chi2), "p": float(p), "label": "accuracy differs between periods" if p < 0.05 else "no evidence accuracy changed over time"}


def rolling(records: list, window: int = 100, points: int = 200) -> list:
    h = np.array([r["hit"] for r in records], dtype=float)
    n = len(h)
    if n < window:
        return []
    cs = np.concatenate([[0.0], np.cumsum(h)])
    idx = np.arange(window, n + 1)
    step = max(1, len(idx) // points)
    out = []
    for i in idx[::step]:
        d = (cs[i] - cs[i - window]) / window
        out.append({"n": int(i), "seq": records[i - 1]["seq"], "direct": float(d), "reverse": float(1 - d)})
    return out


def cumulative(records: list, points: int = 200) -> list:
    h = np.array([r["hit"] for r in records], dtype=float)
    if h.size == 0:
        return []
    c = np.cumsum(h) / np.arange(1, h.size + 1)
    step = max(1, h.size // points)
    return [{"n": int(i + 1), "seq": records[i]["seq"], "direct": float(c[i]), "reverse": float(1 - c[i])} for i in range(0, h.size, step)]


def by_side(records: list) -> dict:
    out = {}
    for s in SIDES:
        sel = [r for r in records if r["pick"] == s]
        out[s] = {"picks": len(sel), **both([r["hit"] for r in sel])}
        out[s]["share_of_picks"] = len(sel) / len(records) if records else None
    return out


def pick_bias(records: list) -> dict:
    """Does the model lean to one side, and does the outcome? Both against a fair 50/50."""
    n = len(records)
    if not n:
        return {}
    pb = sum(r["pick"] == "Big" for r in records); ab = sum(r["actual_side"] == "Big" for r in records)
    return {"n": n, "model_big_share": pb / n, "model_big_p": binom_two_sided(pb, n),
            "actual_big_share": ab / n, "actual_big_p": binom_two_sided(ab, n)}


def correlation(records: list) -> dict:
    """Phi coefficient between the model's pick and the actual side. Negative = predictions anti-correlated with outcomes (what reversal needs)."""
    n = len(records)
    if n < 10:
        return {}
    x = np.array([r["pick"] == "Big" for r in records], dtype=float); y = np.array([r["actual_side"] == "Big" for r in records], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return {"phi": None, "p": None, "note": "pick or outcome is constant"}
    phi = float(np.corrcoef(x, y)[0, 1])
    chi2 = n * phi * phi
    p = float(stats.chi2.sf(chi2, 1))
    return {"phi": phi, "p": p, "n": n,
            "label": "negative (reverse-friendly)" if phi < 0 and p < 0.05 else "positive (direct-friendly)" if phi > 0 and p < 0.05 else "no significant correlation"}


def by_algorithm(records: list) -> list:
    """Each underlying algorithm's own Big/Small pick, scored direct and reversed. Bonferroni over (algorithms + consensus)."""
    names = sorted({m for r in records for m in r["models"]})
    K = len(names) + 1
    rows = []
    cons_h = [r["hit"] for r in records]
    rows.append({"algorithm": "Consensus (app pick)", "n": len(cons_h), **both(cons_h), "big_share": _big_share([r["pick"] for r in records]),
                 **verdict(sum(cons_h), len(cons_h), n_tests=K)})
    for nm in names:
        sel = [(r["models"][nm], r["actual_side"]) for r in records if nm in r["models"]]
        h = [p == a for p, a in sel]
        rows.append({"algorithm": nm, "n": len(h), **both(h), "big_share": _big_share([p for p, _ in sel]), **verdict(sum(h), len(h), n_tests=K)})
    return rows


def _big_share(picks) -> Optional[float]:
    return (sum(p == "Big" for p in picks) / len(picks)) if picks else None


# ----------------------------------------------------------------------------------------------------- strategy selection (no look-ahead)
def adaptive_choices(direct_hits: Sequence, window: int = 300, z: float = 2.58, min_history: int = 300) -> list:
    """For round i choose 'direct' or 'reverse' using ONLY direct_hits[:i] (rounds already resolved when round i is predicted).

    Rule (fixed in advance, not tuned on the test): look at the last `window` resolved rounds. Switch to reverse only if direct accuracy there is
    below 50% by more than z standard errors; stay on direct only if it is above 50% by more than z standard errors; otherwise default to direct
    (the model's own output). With ~50% accuracy this almost never fires - which is the honest outcome."""
    h = np.asarray(direct_hits, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(h)])
    se = math.sqrt(0.25 / window)
    out = []
    for i in range(len(h)):
        if i < max(min_history, window):
            out.append("direct"); continue
        acc = (cs[i] - cs[i - window]) / window                           # hits of rounds i-window .. i-1  (strictly before i)
        out.append("reverse" if acc < 0.5 - z * se else "direct")
    return out


def adaptive_result(records: list, window: int = 300, z: float = 2.58, min_history: int = 300) -> dict:
    h = [r["hit"] for r in records]
    ch = adaptive_choices(h, window, z, min_history)
    hh = [(not x) if c == "reverse" else x for x, c in zip(h, ch)]
    sw = sum(1 for a, b in zip(ch[:-1], ch[1:]) if a != b)
    return {"window": window, "z": z, "min_history": min_history, "reverse_share": (sum(c == "reverse" for c in ch) / len(ch)) if ch else None,
            "switches": sw, "result": rate(hh), "direct": rate(h), "reverse": rate([not x for x in h]),
            "verdict": verdict(sum(hh), len(hh)), "choices": ch}


def next_choice(past_hits: Sequence, window: int = 300, z: float = 2.58, min_history: int = 300) -> dict:
    """The strategy 'Best Historically Validated' would use for the NEXT round, given only already-resolved hits."""
    h = list(past_hits)
    ch = adaptive_choices(h + [True], window, z, min_history)[-1]          # the trailing element's choice never looks at itself
    n = min(len(h), window)
    acc = (sum(h[-n:]) / n) if n else None
    return {"strategy": ch, "window_n": n, "window_direct_accuracy": acc, "history_n": len(h),
            "reason": ("direct accuracy over the last %d resolved rounds is significantly below 50%%" % window) if ch == "reverse" else
                      ("not enough history yet (%d < %d)" % (len(h), max(min_history, window))) if len(h) < max(min_history, window) else
                      "no statistically reliable edge in either direction; defaulting to the model's own (direct) pick"}


def holdout(records: list, train_frac: float = 0.6) -> dict:
    """Chronological out-of-sample test. Strategy (direct / reverse) and the 'best algorithm + direction' are CHOSEN on the first part,
    then frozen and scored on the later, unseen part."""
    n = len(records)
    if n < 200:
        return {"available": False, "note": "need at least 200 tested rounds"}
    cut = int(n * train_frac)
    tr, te = records[:cut], records[cut:]
    d_tr = sum(r["hit"] for r in tr) / len(tr)
    chosen = "reverse" if d_tr < 0.5 else "direct"
    te_h = [r["hit"] for r in te]
    te_c = [(not x) if chosen == "reverse" else x for x in te_h]
    out = {"available": True, "train": {"from_seq": tr[0]["seq"], "to_seq": tr[-1]["seq"], **both([r["hit"] for r in tr])},
           "test": {"from_seq": te[0]["seq"], "to_seq": te[-1]["seq"], **both(te_h)},
           "chosen_on_train": chosen, "chosen_strategy_on_test": rate(te_c), "verdict_on_test": verdict(sum(te_c), len(te_c))}
    names = sorted({m for r in records for m in r["models"]})
    cands = []
    for nm in names:
        for dr in ("direct", "reverse"):
            def acc(seg):
                hs = [(r["models"][nm] == r["actual_side"]) for r in seg if nm in r["models"]]
                if dr == "reverse": hs = [not x for x in hs]
                return hs
            cands.append((nm, dr, sum(acc(tr)) / max(1, len(acc(tr))), acc(te)))
    if cands:
        nm, dr, tr_acc, te_hits = max(cands, key=lambda c: c[2])
        out["best_algorithm_pick"] = {"algorithm": nm, "direction": dr, "train_accuracy": tr_acc, "candidates_searched": len(cands),
                                      "test": rate(te_hits), "verdict_on_test": verdict(sum(te_hits), len(te_hits))}
    return out


# ----------------------------------------------------------------------------------------------------- randomness / ceiling checks
def predictability(sizes: Sequence[str], max_lag: int = 3) -> dict:
    """Is there ANY exploitable structure in the Big/Small sequence itself? (This bounds what any model can reach.)
    Tests: runs test, lag autocorrelation, and chi-square of next side given the previous k sides."""
    x = np.array([1 if s == "Big" else 0 for s in sizes], dtype=int)
    n = len(x)
    if n < 100:
        return {}
    n1 = int(x.sum()); n0 = n - n1
    runs = 1 + int((x[1:] != x[:-1]).sum())
    mu = 2 * n1 * n0 / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1)
    z = (runs - mu) / math.sqrt(var) if var > 0 else 0.0
    out = {"n": n, "big_share": n1 / n, "big_share_p": binom_two_sided(n1, n),
           "runs": {"observed": runs, "expected": mu, "z": z, "p": float(2 * stats.norm.sf(abs(z)))}, "autocorrelation": [], "markov": []}
    xc = x - x.mean()
    for lag in range(1, max_lag + 1):
        r = float((xc[:-lag] * xc[lag:]).sum() / (xc ** 2).sum())
        zz = r * math.sqrt(n)
        out["autocorrelation"].append({"lag": lag, "r": r, "p": float(2 * stats.norm.sf(abs(zz)))})
    for k in range(1, max_lag + 1):
        ctx = np.zeros(n - k, dtype=int)
        for j in range(k):
            ctx = ctx * 2 + x[j:n - k + j]
        nxt = x[k:]
        tbl = np.zeros((2 ** k, 2))
        for c, v in zip(ctx, nxt):
            tbl[c, v] += 1
        chi2, p, _, _ = stats.chi2_contingency(tbl + 1e-9)
        best = float(np.max(tbl, axis=1).sum() / tbl.sum())                  # in-sample 'best guess per context' accuracy (optimistic)
        out["markov"].append({"order": k, "chi2": float(chi2), "p": float(p), "in_sample_best_accuracy": best})
    return out


def sample_size_needed(edge: float = 0.01, alpha: float = 0.05, power: float = 0.8) -> int:
    """Rounds needed to detect accuracy of 0.5+edge against 0.5 (two-sided)."""
    za, zb = stats.norm.isf(alpha / 2), stats.norm.isf(1 - power)
    p1 = 0.5 + edge
    return int(math.ceil(((za * 0.5 + zb * math.sqrt(p1 * (1 - p1))) / edge) ** 2))


def target_reachability(n: int, target: float = 0.75) -> dict:
    """How far above chance would 75% be for this sample, and how likely is it by luck alone?"""
    sd = math.sqrt(0.25 / n)
    z = (target - 0.5) / sd
    return {"n": n, "target": target, "std_error_of_a_coin": sd, "z_score_of_target": z,
            "p_by_luck_alone": float(stats.norm.sf(z)), "random_95pct_range": [0.5 - 1.96 * sd, 0.5 + 1.96 * sd]}


# ----------------------------------------------------------------------------------------------------- one call for the API / dashboard
def analyse(records: list, window: int = 100, n_blocks: int = 6, all_meta: Optional[Sequence[dict]] = None, adaptive_window: int = 300) -> dict:
    if not records:
        return {"available": False, "reason": "no verified walk-forward predictions in the selected range"}
    h = [r["hit"] for r in records]
    n = len(h)
    blocks = _blocks(records, n_blocks)
    by_date = {}
    for r in records:
        if r["date"]:
            by_date.setdefault(r["date"], []).append(r["hit"])
    summ = both(h)
    return {"available": True,
            "range": {"first_seq": records[0]["seq"], "last_seq": records[-1]["seq"], "date_from": records[0]["date"], "date_to": records[-1]["date"], "n": n},
            "summary": {**summ, "difference_reverse_minus_direct": summ["reverse"]["rate"] - summ["direct"]["rate"],
                        "test": verdict(sum(h), n), "baseline": 0.5},
            "by_side": by_side(records), "pick_bias": pick_bias(records), "correlation": correlation(records),
            "algorithms": by_algorithm(records), "blocks": blocks, "drift": _drift(blocks),
            "by_date": [{"date": d, **both(v)} for d, v in sorted(by_date.items())],
            "rolling": {"window": window, "points": rolling(records, window)}, "cumulative": cumulative(records),
            "adaptive": {k: v for k, v in adaptive_result(records, adaptive_window).items() if k != "choices"},
            "holdout": holdout(records), "predictability": predictability([r["actual_side"] for r in records]),
            "reachability": target_reachability(n), "rounds_needed_to_detect_1pt_edge": sample_size_needed(0.01),
            "data_quality": data_quality(records, all_meta)}
