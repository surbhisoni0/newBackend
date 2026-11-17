"""
Wingo prediction experiment - leakage-free, chronological.

Usage:   python wingo_experiment.py path/to/results.csv [--quick]
Input:   a CSV with a `number` column (0-9), OLDEST round first. Run it once on your 1-minute data and once on your 30-second data.
Needs:   numpy pandas scipy scikit-learn lightgbm xgboost

What it does
  * Split IN TIME ORDER: first 60% = train, next 20% = validation (hyper-parameters and the confidence threshold are chosen ONLY here),
    last 20% = test (looked at ONCE). No shuffling, no future rows in any feature (every feature for round t uses rounds < t only).
  * Targets, analysed separately:  Big/Small (baseline 50%),  Colour = Green / Red / Violet (best constant guess 40%),  Number 0-9 (baseline 10%).
  * Models: majority baseline, logistic regression, random forest, extra trees, histogram GB, LightGBM and XGBoost (random hyper-parameter search),
    soft-voting ensemble of the three best on validation.
  * Confidence filtering: only act when the model's top probability is in its top 20% / 5% of the VALIDATION confidence range.
  * NULL CONTROL: the same pipeline on a SHUFFLED copy of your data (all temporal structure destroyed). If a "best model" on shuffled data also
    scores like your real data, the real score is luck.
Nothing here is tuned on the test block, and nothing is hard-coded.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
COLOUR = np.array([2, 0, 1, 0, 1, 2, 1, 0, 1, 0])          # 0=Green(1,3,7,9) 1=Red(2,4,6,8) 2=Violet(0,5)
BASE = {"bigsmall": 0.5, "colour": 0.4, "number": 0.1}
NCLS = {"bigsmall": 2, "colour": 3, "number": 10}


def features(n: np.ndarray, L: int = 20, start: int = 100):
    """Row for round t uses ONLY rounds before t. Returns X, t-index array."""
    N, sz = len(n), (n >= 5).astype(int)
    col = COLOUR[n]
    c_sz = np.r_[0, np.cumsum(sz)]
    c_dig = np.zeros((N + 1, 10))
    c_dig[1:] = np.cumsum(np.eye(10)[n], axis=0)
    last_seen = np.full(10, -1)
    since = np.zeros((N, 10))
    run = np.zeros(N)
    for t in range(N):                                        # state BEFORE seeing round t
        since[t] = t - last_seen
        run[t] = 0 if t == 0 else (run[t - 1] + 1 if (t >= 2 and sz[t - 1] == sz[t - 2]) else 1)
        last_seen[n[t]] = t
    rows, idx = [], []
    for t in range(start, N):
        f = [*n[t - L:t], *sz[t - L:t], *col[t - 5:t]]
        for w in (5, 10, 20, 50, 100):
            f.append((c_sz[t] - c_sz[t - w]) / w)
        for w in (20, 100):
            f.extend((c_dig[t] - c_dig[t - w]) / w)
        f.extend(since[t]); f.append(run[t])
        rows.append(f); idx.append(t)
    return np.array(rows, float), np.array(idx)


def targets(n, idx):
    return {"bigsmall": (n[idx] >= 5).astype(int), "colour": COLOUR[n[idx]], "number": n[idx]}


def ci(k, m):
    r = stats.binomtest(int(k), int(m), 0.5).proportion_ci(method="wilson")
    return 100 * r.low, 100 * r.high


def pick_params(rng, kind):
    p = dict(n_estimators=int(rng.integers(40, 250 if not QUICK else 120)), learning_rate=float(rng.choice([0.02, 0.05, 0.1])),
             subsample=float(rng.choice([0.6, 0.8, 1.0])), colsample=float(rng.choice([0.5, 0.8, 1.0])), reg=float(rng.choice([0.1, 1, 10, 50])),
             depth=int(rng.choice([2, 3, 4, 6])), min_child=int(rng.choice([20, 50, 100])))
    return p


def build(name, p, ncls):
    if name == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=2 ** p["depth"], max_depth=p["depth"] + 1,
                                  min_child_samples=p["min_child"], subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample"],
                                  reg_lambda=p["reg"], random_state=0, verbose=-1, n_jobs=1)
    return xgb.XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["depth"], min_child_weight=p["min_child"] / 10,
                             subsample=p["subsample"], colsample_bytree=p["colsample"], reg_lambda=p["reg"], random_state=0, n_jobs=1, verbosity=0,
                             tree_method="hist")


def run_target(X, y, name, seed=0, search=6, verbose=True):
    N = len(y)
    a, b = int(N * 0.6), int(N * 0.8)
    Xtr, ytr, Xva, yva, Xte, yte = X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]
    ncls = NCLS[name]
    rng = np.random.default_rng(seed)
    cand = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=300)),
        "random_forest": lambda: RandomForestClassifier(200 if not QUICK else 80, max_depth=5, min_samples_leaf=20, random_state=0, n_jobs=1),
        "extra_trees": lambda: ExtraTreesClassifier(200 if not QUICK else 80, max_depth=6, min_samples_leaf=20, random_state=0, n_jobs=1),
        "hist_gb": lambda: HistGradientBoostingClassifier(max_iter=60, learning_rate=0.05, max_depth=3, min_samples_leaf=40, random_state=0),
    }
    fitted = {}
    for k, mk in cand.items():
        m = mk().fit(Xtr, ytr); fitted[k] = (m, None)
    for lib in ("lightgbm", "xgboost"):                                    # random search; selected on VALIDATION accuracy only
        best = None
        for _ in range(search):
            p = pick_params(rng, lib)
            try:
                m = build(lib, p, ncls).fit(Xtr, ytr)
            except Exception:
                continue
            v = (m.predict(Xva) == yva).mean()
            if best is None or v > best[0]:
                best = (v, m, p)
        fitted[lib] = (best[1], best[2])
    res = {}
    for k, (m, _) in fitted.items():
        pva, pte = m.predict_proba(Xva), m.predict_proba(Xte)
        res[k] = {"val": (pva.argmax(1) == yva).mean(), "test_p": pte, "val_p": pva}
    top3 = sorted([k for k in res if k not in ("majority",)], key=lambda k: -res[k]["val"])[:3]
    pva = np.mean([res[k]["val_p"] for k in top3], axis=0); pte = np.mean([res[k]["test_p"] for k in top3], axis=0)
    res["ensemble(top3 by val)"] = {"val": (pva.argmax(1) == yva).mean(), "test_p": pte, "val_p": pva}
    maj = np.bincount(ytr).argmax()
    out = {"baseline_majority_test": (yte == maj).mean(), "n_test": len(yte), "models": {}}
    for k, r in res.items():
        hit = r["test_p"].argmax(1) == yte
        out["models"][k] = {"val": r["val"], "test": hit.mean(), "k": int(hit.sum()), "ci": ci(hit.sum(), len(hit))}
    best = max(res, key=lambda k: res[k]["val"])                          # the model a person would pick = best on validation
    out["picked"] = best
    conf_va, conf_te = res[best]["val_p"].max(1), res[best]["test_p"].max(1)
    hit = res[best]["test_p"].argmax(1) == yte
    out["filter"] = {}
    for q in (0.0, 0.8, 0.95):
        thr = np.quantile(conf_va, q)
        sel = conf_te >= thr
        out["filter"][q] = (int(sel.sum()), float(hit[sel].mean()) if sel.sum() else float("nan"), ci(hit[sel].sum(), sel.sum()) if sel.sum() else (0, 0))
    return out


def report(name, out):
    print(f"\n=== {name.upper()}  (chance baseline {100 * BASE[name]:.0f}%, majority-class on test {100 * out['baseline_majority_test']:.1f}%, n_test={out['n_test']}) ===")
    print(f"{'model':26s}{'val acc':>9s}{'TEST acc':>10s}   95% range")
    for k, m in out["models"].items():
        print(f"{k:26s}{100 * m['val']:8.2f}%{100 * m['test']:9.2f}%   [{m['ci'][0]:.1f}, {m['ci'][1]:.1f}]")
    print(f"model chosen on validation: {out['picked']}")
    for q, (cnt, acc, c) in out["filter"].items():
        lab = "all picks" if q == 0 else f"only top {int((1 - q) * 100)}% most confident"
        print(f"  {lab:32s} {cnt:4d} picks  accuracy {100 * acc:5.1f}%  [{c[0]:.1f}, {c[1]:.1f}]")
    print(f"  ({len(out['models'])} models were compared; with that many tries the best one looks better than chance by luck alone - judge by the 95% range and the shuffled-data control, not the maximum)")


def main():
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    df = pd.read_csv(path, dtype=str)
    n = df["number"].astype(int).values
    print(f"rounds: {len(n)} | quick={QUICK}")
    X, idx = features(n)
    Y = targets(n, idx)
    t0 = time.time()
    real = {}
    for name in ("bigsmall", "colour", "number"):
        real[name] = run_target(X, Y[name], name, search=4 if QUICK else 6)
        report(name, real[name]); print(f"  [{time.time() - t0:.0f}s]", flush=True)
    print("\n=== NULL CONTROL: same pipeline on SHUFFLED history (no pattern can exist) - Big/Small ===")
    accs = []
    for s in range(3 if not QUICK else 2):
        ns = np.random.default_rng(100 + s).permutation(n)
        Xs, ix = features(ns)
        o = run_target(Xs, targets(ns, ix)["bigsmall"], "bigsmall", seed=s, search=3)
        picked = o["models"][o["picked"]]
        accs.append(picked["test"])
        print(f"  shuffle {s + 1}: model chosen on validation = {o['picked']:22s} test accuracy {100 * picked['test']:.2f}%", flush=True)
    r = real["bigsmall"]; rp = r["models"][r["picked"]]["test"]
    print(f"\nReal-data Big/Small test accuracy of the validation-chosen model: {100 * rp:.2f}%  |  same pipeline on shuffled data: {', '.join(f'{100 * a:.2f}%' for a in accs)}")
    print("If those are in the same range, the pipeline finds nothing the shuffled (patternless) data does not also 'find'.")


if __name__ == "__main__":
    main()
