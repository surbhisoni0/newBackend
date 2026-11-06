"""Deep research harness: is there ANY out-of-sample signal in the Big/Small sequence?

Protocol (fixed before looking at results):
  * Data is split chronologically: DEV = everything before the last FINAL_N rounds, FINAL = last FINAL_N rounds (default 1000).
  * Every model is walk-forwarded: the prediction for round t is made by a model trained ONLY on rounds < t
    (labels of rounds < t are known), refit every `refit_every` rounds. Features for round t use h[:t] only.
  * Model selection and any confidence threshold are chosen on DEV only. FINAL is scored once, never tuned on.
  * Direct, Reverse, ensemble and confidence-filtered results are all reported; Bonferroni correction over every model tested.

Run:  python -m app.backtesting.research --csv data/wingo_docx_clean.csv --out research_report.json
"""
import argparse
import json
import math
import time
import warnings
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from app.prediction.features import feats
from app.utils.stats import wilson

warnings.filterwarnings("ignore")
FINAL_N = 1000


# ------------------------------------------------------------------------------------------------ randomness tests
def randomness_report(h: np.ndarray, y: np.ndarray) -> dict:
    n = len(h)
    out = {"n": n}
    cnt = np.bincount(h, minlength=10)
    out["digit_counts"] = cnt.tolist()
    out["digit_chi2_p"] = float(stats.chisquare(cnt).pvalue)
    out["big_share"] = float(y.mean())
    out["big_share_p"] = float(stats.binomtest(int(y.sum()), n, 0.5).pvalue)
    p = cnt / n
    out["entropy_bits"] = float(-(p * np.log2(p)).sum())
    out["entropy_max_bits"] = math.log2(10)
    # transitions on contiguous pairs only (skip pairs that straddle a data gap is handled by caller via h being clean)
    T = np.bincount(h[:-1] * 10 + h[1:], minlength=100).reshape(10, 10)
    chi2, pv, _, _ = stats.chi2_contingency(T)
    out["transition_10x10_chi2_p"] = float(pv)
    T2 = np.bincount(y[:-1] * 2 + y[1:], minlength=4).reshape(2, 2)
    out["bs_transition_2x2"] = T2.tolist()
    out["bs_transition_chi2_p"] = float(stats.chi2_contingency(T2)[1])
    # order-k context tests on Big/Small
    ctx = []
    for k in (2, 3, 4, 5):
        idx = np.zeros(n - k, dtype=int)
        for j in range(k):
            idx = idx * 2 + y[j:n - k + j]
        tbl = np.zeros((2 ** k, 2))
        for c, nxt in zip(idx, y[k:]):
            tbl[c, nxt] += 1
        ctx.append({"order": k, "chi2_p": float(stats.chi2_contingency(tbl[tbl.sum(1) > 0])[1])})
    out["bs_context_tests"] = ctx
    # autocorrelation lags 1..50 and how many are "significant" (expect ~5% by chance)
    xc = y - y.mean()
    ac = [float((xc[:-l] * xc[l:]).sum() / (xc ** 2).sum()) for l in range(1, 51)]
    pvals = [float(2 * stats.norm.sf(abs(a) * math.sqrt(n))) for a in ac]
    out["autocorr_lags_1_50"] = {"max_abs_r": float(max(abs(a) for a in ac)), "n_nominal_sig_5pct": int(sum(p_ < 0.05 for p_ in pvals)),
                                 "expected_by_chance": 2.5, "n_sig_bonferroni": int(sum(p_ < 0.05 / 50 for p_ in pvals))}
    runs = 1 + int((y[1:] != y[:-1]).sum()); n1 = int(y.sum()); n0 = n - n1
    mu = 2 * n1 * n0 / n + 1; var = (mu - 1) * (mu - 2) / (n - 1)
    z = (runs - mu) / math.sqrt(var)
    out["runs_test"] = {"observed": runs, "expected": mu, "z": z, "p": float(2 * stats.norm.sf(abs(z)))}
    # streak-length distribution vs geometric
    lens, r = [], 1
    for a, b in zip(y[:-1], y[1:]):
        if a == b: r += 1
        else: lens.append(r); r = 1
    lens.append(r)
    L = np.bincount(lens)[1:]
    kmax = 8
    obs = np.r_[L[:kmax - 1], L[kmax - 1:].sum()]
    probs = np.array([0.5 ** k for k in range(1, kmax)] + [0.5 ** (kmax - 1)])
    out["streak_length_chi2_p"] = float(stats.chisquare(obs, probs / probs.sum() * obs.sum()).pvalue)
    # lagged mutual information with permutation null (digit level)
    rng = np.random.default_rng(0)
    def mi(a, b):
        j = np.bincount(a * 10 + b, minlength=100).reshape(10, 10) / len(a)
        pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
        m = j > 0
        return float((j[m] * np.log2(j[m] / (pa @ pb)[m])).sum())
    mis = []
    for lag in (1, 2, 3, 5, 10):
        obs_mi = mi(h[:-lag], h[lag:])
        null = [mi(h[:-lag], rng.permutation(h[lag:])) for _ in range(200)]
        mis.append({"lag": lag, "mi_bits": obs_mi, "perm_p": float((np.sum(np.array(null) >= obs_mi) + 1) / 201)})
    out["digit_mutual_information"] = mis
    return out


# ------------------------------------------------------------------------------------------------ features
def rich_features(h: np.ndarray, t: int) -> np.ndarray:
    """Existing engine features (history h[:t]) + extra Big/Small windows, streak/disagreement-free extras. Uses h[:t] only."""
    hist = h[:t]
    base = feats(hist)
    y = (hist >= 5).astype(float)
    ex = []
    for w in (5, 10, 20, 50, 100, 200):
        seg = y[-w:]
        ex += [seg.mean(), seg.mean() - 0.5, (seg.mean() - 0.5) * math.sqrt(len(seg)) / 0.5]   # frequency deviation z-ish
    ex += list(y[-12:]) + [0.5] * max(0, 12 - len(y))                                         # last 12 sides
    # Big/Small transition probabilities from the last side, long and short memory
    for w in (100, 500, 5000):
        s = y[-w:]
        if len(s) > 2:
            a = s[:-1]; b = s[1:]
            for side in (0, 1):
                m = a == side
                ex.append((b[m].sum() + 1) / (m.sum() + 2))
        else:
            ex += [0.5, 0.5]
    ex.append(float(hist[-1] == hist[-2]) if t > 1 else 0.0)
    return np.r_[base, np.array(ex, dtype=float)]


# ------------------------------------------------------------------------------------------------ walk-forward machinery
def build_matrix(h: np.ndarray, warm: int = 200):
    X = np.array([rich_features(h, t) for t in range(warm, len(h))])
    return X, h[warm:], np.arange(warm, len(h))


def wf_predict_proba(fit_predict: Callable, X, y, start_idx: int, end_idx: int, refit_every: int, train_window: int = 0):
    """Walk-forward: for target rows in [start_idx,end_idx) fit on rows < the block start only. Returns P(Big) per target row."""
    out = np.full(end_idx - start_idx, 0.5)
    for b in range(start_idx, end_idx, refit_every):
        e = min(end_idx, b + refit_every)
        lo = max(0, b - train_window) if train_window else 0
        out[b - start_idx:e - start_idx] = fit_predict(X[lo:b], y[lo:b], X[b:e])
    return out


def _sk(make):
    def fp(Xtr, ytr_digit, Xte):
        ytr = (ytr_digit >= 5).astype(int)
        if ytr.min() == ytr.max():
            return np.full(len(Xte), float(ytr[0]))
        m = make(); m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    return fp


def model_zoo() -> Dict[str, Callable]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    zoo = {
        "LogReg(C=0.02)": _sk(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=400))),
        "RandForest": _sk(lambda: RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, n_jobs=1, random_state=0)),
        "ExtraTrees": _sk(lambda: ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, n_jobs=1, random_state=0)),
        "HistGB": _sk(lambda: HistGradientBoostingClassifier(max_iter=40, learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=0)),
    }
    try:
        import xgboost as xgb
        zoo["XGBoost"] = _sk(lambda: xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=0, verbosity=0))
    except Exception:
        pass
    try:
        import lightgbm as lgb
        zoo["LightGBM"] = _sk(lambda: lgb.LGBMClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, random_state=0, verbose=-1))
    except Exception:
        pass
    try:
        import catboost as cb
        zoo["CatBoost"] = _sk(lambda: cb.CatBoostClassifier(iterations=80, depth=3, learning_rate=0.05, thread_count=1, random_seed=0, verbose=0))
    except Exception:
        pass
    return zoo


# simple (non-ML) predictors operating on the raw sequence: return P(Big) for rows [start,end)
def seq_models(h: np.ndarray, start: int, end: int) -> Dict[str, np.ndarray]:
    y = (h >= 5).astype(int)
    n = end - start
    res = {k: np.zeros(n) for k in ("Majority(train)", "CumFreq(Big)", "Recent50", "Repeat-last", "Opposite-last", "BS-Markov1", "BS-Markov2", "BS-Markov3",
                                    "Bayes Beta-Binomial", "Bayes Dirichlet digit", "Digit-first Markov1")}
    for i, t in enumerate(range(start, end)):
        past = y[:t]; ph = h[:t]
        res["Majority(train)"][i] = float(past.mean() >= 0.5)
        res["CumFreq(Big)"][i] = past.mean()
        res["Recent50"][i] = past[-50:].mean()
        res["Repeat-last"][i] = float(past[-1])
        res["Opposite-last"][i] = 1.0 - float(past[-1])
        for k, nm in ((1, "BS-Markov1"), (2, "BS-Markov2"), (3, "BS-Markov3")):
            if t > k + 5:
                idx = np.zeros(t - k, dtype=int)
                for j in range(k): idx = idx * 2 + past[j:t - k + j]
                ctx = 0
                for j in range(k): ctx = ctx * 2 + past[t - k + j]
                m = idx == ctx
                res[nm][i] = (past[k:][m].sum() + 1) / (m.sum() + 2)
            else:
                res[nm][i] = 0.5
        res["Bayes Beta-Binomial"][i] = (past.sum() + 1) / (t + 2)                       # posterior mean, uniform prior
        c = np.bincount(ph, minlength=10) + 1.0
        res["Bayes Dirichlet digit"][i] = c[5:].sum() / c.sum()
        C = np.ones((10, 10))
        C += np.bincount(ph[:-1] * 10 + ph[1:], minlength=100).reshape(10, 10)
        r = C[ph[-1]]; r = r / r.sum()
        res["Digit-first Markov1"][i] = r[5:].sum()
    return res



# ------------------------------------------------------------------------------------------------ power check
def planted_sequence(n: int, p_repeat: float, seed: int) -> np.ndarray:
    """Synthetic digits whose Big/Small side repeats the previous side with probability p_repeat (a REAL, known signal)."""
    rng = np.random.default_rng(seed)
    side = np.empty(n, dtype=int); side[0] = rng.integers(0, 2)
    for i in range(1, n):
        side[i] = side[i - 1] if rng.random() < p_repeat else 1 - side[i - 1]
    return np.where(side == 1, rng.integers(5, 10, n), rng.integers(0, 5, n))


def power_check(n: int = 5086, final_n: int = FINAL_N, seeds=(0, 1, 2)) -> dict:
    """Can THIS pipeline detect a real edge? Plant one, run the same DEV/FINAL protocol with the cheap models, report detection."""
    out = []
    for p_rep in (0.5, 0.52, 0.55, 0.60):
        accs, ps = [], []
        for sd in seeds:
            h = planted_sequence(n, p_rep, sd)
            y = (h >= 5).astype(int)
            start = n - final_n
            pb = seq_models(h, start, n)["BS-Markov1"]
            sc = score(pb, y[start:])
            accs.append(sc["direct"]); ps.append(sc["p_two_sided"])
        out.append({"planted_repeat_prob": p_rep, "mean_final_accuracy": float(np.mean(accs)), "mean_p_two_sided": float(np.mean(ps)),
                    "detected_at_5pct": int(sum(p_ < 0.05 for p_ in ps)), "seeds": len(seeds)})
    # minimum detectable accuracy with `final_n` predictions (80% power, two-sided 5%)
    se = math.sqrt(0.25 / final_n)
    mde = 0.5 + (1.96 + 0.84) * se
    return {"planted": out, "final_n": final_n, "min_detectable_accuracy_80pct_power": float(mde),
            "needed_for_80pct_accuracy_z": float((0.80 - 0.5) / se)}

# ------------------------------------------------------------------------------------------------ scoring
def score(pb: np.ndarray, y: np.ndarray, thr: float = 0.5) -> dict:
    pred = (pb > thr).astype(int)
    hit = pred == y
    n, k = len(y), int(hit.sum())
    lo, hi = wilson(k, n)
    p_two = float(stats.binomtest(k, n, 0.5).pvalue)
    p_one = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)
    p = np.clip(pb, 1e-4, 1 - 1e-4)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return {"n": n, "k": k, "direct": k / n, "reverse": 1 - k / n, "lo": lo, "hi": hi, "p_two_sided": p_two, "p_one_sided_better": p_one,
            "brier": float(((p - y) ** 2).mean()), "logloss": ll, "logloss_coin": math.log(2), "big_share_of_picks": float(pred.mean())}


def confidence_filter(pb: np.ndarray, y: np.ndarray, thr: float, margin: float = 0.0) -> dict:
    conf = np.abs(pb - 0.5)
    sel = conf > thr
    if sel.sum() == 0:
        return {"n": 0}
    hit = ((pb[sel] > 0.5).astype(int) == y[sel])
    n, k = int(sel.sum()), int(hit.sum())
    return {"n": n, "coverage": n / len(y), "direct": k / n, "reverse": 1 - k / n, "p_two_sided": float(stats.binomtest(k, n, 0.5).pvalue)}


def run(csv: str, out: str, refit_every: int = 100, warm: int = 200, final_n: int = FINAL_N) -> dict:
    t0 = time.time()
    df = pd.read_csv(csv, dtype={"period_id": str}).sort_values("period_id").reset_index(drop=True)
    h = df["number"].astype(int).values
    y = (h >= 5).astype(int)
    N = len(h)
    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data_n": N, "final_n": final_n, "warm": warm, "refit_every": refit_every}
    # --- data quality
    pid = df["period_id"].astype("int64").values
    d = np.diff(pid)
    res["data_quality"] = {"rounds": N, "period_id_jumps": int((d != 1).sum()), "rounds_missing_in_jumps": int(sum(x - 1 for x in d if 1 < x < 100)),
                           "days": sorted(set(df["period_id"].str[:8])), "game_code_values": sorted(set(df["period_id"].str[8:13]))[:5],
                           "note": "Timestamps in the CSV are ESTIMATED (Data.docx has none), so time-of-day features use the period-id round-of-day only."}
    res["randomness_all"] = randomness_report(h, y)
    res["randomness_dev_only"] = randomness_report(h[:N - final_n], y[:N - final_n])
    # time-of-day (round number within the UTC day -> minute of day)
    rnd = df["period_id"].str[-4:].astype(int).values
    bucket = (rnd // 120)                       # 2-hour buckets (1 round = 1 minute)
    tbl = np.array([[((y == s) & (bucket == b)).sum() for s in (0, 1)] for b in np.unique(bucket)])
    tbl = tbl[tbl.sum(1) > 30]
    res["time_of_day_bigshare_chi2_p"] = float(stats.chi2_contingency(tbl)[1]) if len(tbl) > 1 else None
    day = df["period_id"].str[:8].values
    dtbl = np.array([[((y == s) & (day == dd)).sum() for s in (0, 1)] for dd in np.unique(day)]); dtbl = dtbl[dtbl.sum(1) > 30]
    res["per_day_bigshare"] = {dd: float(y[day == dd].mean()) for dd in np.unique(day)}
    res["per_day_chi2_p"] = float(stats.chi2_contingency(dtbl)[1])

    # --- models
    X, yd, ridx = build_matrix(h, warm)                      # row j  <->  target round index warm+j
    n_rows = len(yd)
    final_start = n_rows - final_n
    dev_eval_start = 1000 - warm if 1000 > warm else 0       # first DEV predictions after 1000 rounds of history
    ybin = (yd >= 5).astype(int)
    probs_dev, probs_final = {}, {}
    zoo = model_zoo()
    for nm, fp in zoo.items():
        ts = time.time()
        pb = wf_predict_proba(fp, X, yd, dev_eval_start, n_rows, refit_every)       # one continuous walk-forward; DEV part and FINAL part never share fits
        probs_dev[nm], probs_final[nm] = pb[:final_start - dev_eval_start], pb[final_start - dev_eval_start:]
        print(f"  {nm}: {time.time() - ts:.0f}s", flush=True)
    sm = seq_models(h, warm + dev_eval_start, N)
    for nm, pb in sm.items():
        probs_dev[nm], probs_final[nm] = pb[:final_start - dev_eval_start], pb[final_start - dev_eval_start:]
    y_dev, y_fin = ybin[dev_eval_start:final_start], ybin[final_start:]
    res["dev_n"], res["final_n_actual"] = int(len(y_dev)), int(len(y_fin))

    # --- ensembles (equal weight, no tuned weights)
    ml_names = [k for k in zoo]
    for nm_e, names in (("Ensemble(all ML, mean P)", ml_names), ("Ensemble(trees only)", [k for k in ml_names if k != "LogReg(C=0.02)"])):
        probs_dev[nm_e] = np.mean([probs_dev[k] for k in names], axis=0)
        probs_final[nm_e] = np.mean([probs_final[k] for k in names], axis=0)
    # --- model selection on DEV ONLY (best DEV direct accuracy among genuine predictors; reverse-style baselines excluded)
    cand = [k for k in probs_dev if k not in ("Opposite-last", "Majority(train)")]
    dev_scores = {k: score(probs_dev[k], y_dev) for k in probs_dev}
    selected = max(cand, key=lambda k: dev_scores[k]["direct"])
    res["selection"] = {"rule": "highest DEV direct accuracy (FINAL never used)", "selected": selected, "dev_accuracy": dev_scores[selected]["direct"]}
    K = len(probs_dev)
    final_scores = {k: score(probs_final[k], y_fin) for k in probs_final}
    table = []
    for k in probs_dev:
        f, dv = final_scores[k], dev_scores[k]
        table.append({"model": k, "dev": dv, "final": f, "final_p_bonferroni_two_sided": min(1.0, f["p_two_sided"] * K),
                      "selected_on_dev": k == selected})
    res["model_table"] = table
    res["bonferroni_K"] = K
    res["selected_final"] = final_scores[selected]
    # --- random baseline (many coin-flip strategies: what does 'best of K' look like by pure luck?)
    rng = np.random.default_rng(1)
    best_of_k = [max((rng.integers(0, 2, len(y_fin)) == y_fin).mean() for _ in range(K)) for _ in range(500)]
    res["luck_benchmark"] = {"coin_flip_accuracy": float((rng.integers(0, 2, len(y_fin)) == y_fin).mean()),
                             "best_of_K_coin_flips_median": float(np.median(best_of_k)), "best_of_K_coin_flips_95pct": float(np.percentile(best_of_k, 95)), "K": K}
    # --- confidence filtering: threshold chosen on DEV, applied once to FINAL
    pbd, pbf = probs_dev[selected], probs_final[selected]
    best_thr, best_acc = None, 0
    for thr in (0.01, 0.02, 0.03, 0.05, 0.08):
        r = confidence_filter(pbd, y_dev, thr)
        if r.get("n", 0) >= 200 and r["direct"] > best_acc:
            best_thr, best_acc = thr, r["direct"]
    res["confidence_filter"] = {"model": selected, "threshold_chosen_on_dev": best_thr,
                                "dev": {str(t): confidence_filter(pbd, y_dev, t) for t in (0.01, 0.02, 0.03, 0.05, 0.08)},
                                "final_at_dev_threshold": confidence_filter(pbf, y_fin, best_thr) if best_thr else None,
                                "final_all_thresholds_for_information_only": {str(t): confidence_filter(pbf, y_fin, t) for t in (0.01, 0.02, 0.03, 0.05, 0.08)}}
    # --- direct vs reverse decided on DEV
    dir_dev = dev_scores[selected]["direct"]
    res["direct_vs_reverse"] = {"model": selected, "dev_direct": dir_dev, "chosen_on_dev": "reverse" if dir_dev < 0.5 else "direct",
                                "final_direct": final_scores[selected]["direct"], "final_reverse": final_scores[selected]["reverse"]}
    # --- calibration of selected model on FINAL
    conf = pbf
    edges = np.quantile(conf, [0, .2, .4, .6, .8, 1.0])
    cal = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf <= b)
        cal.append({"mean_pbig": float(conf[m].mean()), "observed_big": float(y_fin[m].mean()), "n": int(m.sum())})
    res["calibration_selected_final"] = cal
    # --- ceiling: can an oracle-free 'digit-first' beat BS direct? (digit top1 accuracy vs 10%)
    res["power_check"] = power_check(N, final_n)
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/wingo_docx_clean.csv")
    ap.add_argument("--out", default="research_report.json")
    ap.add_argument("--refit-every", type=int, default=100)
    a = ap.parse_args()
    r = run(a.csv, a.out, a.refit_every)
    print(json.dumps({k: r[k] for k in ("selection", "selected_final", "luck_benchmark", "direct_vs_reverse")}, indent=1, default=float))
