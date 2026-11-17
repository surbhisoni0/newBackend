"""Walk-forward ML lab for WinGo rounds.

Rules enforced here:
  * feature row for round t is built from digits[:t] ONLY (see test_no_leak)
  * models are fitted on rows < t_fit and used for rounds >= t_fit (expanding window, chronological)
  * every target is compared with its own constant-guess baseline
  * selection of thresholds / hyper-parameters happens on DEV only; FINAL is scored once
"""
import numpy as np, pandas as pd, warnings, time, json, sys
from scipy import stats
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost as xgb

COLOR_GROUP = np.array([2, 0, 1, 0, 1, 2, 1, 0, 1, 0])  # 0 green, 1 red, 2 violet


def load(path="data/wingo_docx_clean.csv"):
    d = pd.read_csv(path, dtype={"period_id": str})
    d["rnd"] = d.period_id.str[-4:].astype(int)           # round-of-day from period id
    d["gap"] = (d.period_id.astype("int64").diff().fillna(1) != 1).astype(int)
    return d


def features(h, rnd=None, gap=None):
    """Return X (n x F) where row t only uses h[:t]. Row 0..warm are NaN-filled with -1 history."""
    h = np.asarray(h, int); n = len(h)
    big = (h >= 5).astype(int); odd = h % 2; cg = COLOR_GROUP[h]
    cols, names = [], []

    def add(name, arr): cols.append(np.asarray(arr, float)); names.append(name)

    def lag(a, k):
        o = np.full(n, -1.0); o[k:] = a[:n - k]; return o
    for k in range(1, 21):
        add(f"d_lag{k}", lag(h, k))
    for k in range(1, 11):
        add(f"big_lag{k}", lag(big, k)); add(f"cg_lag{k}", lag(cg, k))
    cs = np.concatenate([[0], np.cumsum(big)])
    for w in (5, 10, 20, 50, 100, 200, 500):
        o = np.full(n, 0.5)
        for t in range(1, n):
            a = max(0, t - w); o[t] = (cs[t] - cs[a]) / (t - a)
        add(f"bigfrac_{w}", o)
    for dgt in range(10):
        ind = (h == dgt).astype(int); c = np.concatenate([[0], np.cumsum(ind)])
        for w in (30, 150):
            o = np.full(n, 0.1)
            for t in range(1, n):
                a = max(0, t - w); o[t] = (c[t] - c[a]) / (t - a)
            add(f"dfrac{dgt}_{w}", o)
        g = np.full(n, 200.0); last = -1
        for t in range(n):
            g[t] = min(200, t - last) if last >= 0 else 200
            if h[t] == dgt: last = t
        add(f"gap_digit{dgt}", g)

    def streak(a):
        o = np.zeros(n); r = 0
        for t in range(1, n):
            r = r + 1 if a[t - 1] == (a[t - 2] if t >= 2 else -9) else 1
            o[t] = r
        return o
    add("streak_big", streak(big)); add("streak_odd", streak(odd)); add("streak_cg", streak(cg)); add("streak_digit", streak(h))
    # expanding Markov probabilities: counts include only transitions completed before round t
    C = np.ones((10, 10)); CB = np.ones((4, 2)); pb = np.full(n, 0.5); pb2 = np.full(n, 0.5)
    for t in range(1, n):
        if t >= 2:
            C[h[t - 2], h[t - 1]] += 1
        if t >= 3:
            CB[big[t - 3] * 2 + big[t - 2], big[t - 1]] += 1
        pb[t] = C[h[t - 1], 5:].sum() / C[h[t - 1]].sum()
        if t >= 2:
            s2 = big[t - 2] * 2 + big[t - 1]; pb2[t] = CB[s2, 1] / CB[s2].sum()
    add("markov1_pbig", pb); add("markov2_pbig", pb2)
    for w in (5, 10, 20):
        m = np.zeros(n); s = np.zeros(n)
        for t in range(1, n):
            seg = h[max(0, t - w):t]; m[t] = seg.mean(); s[t] = seg.std()
        add(f"mean_{w}", m); add(f"std_{w}", s)
    if rnd is not None:
        r = np.asarray(rnd, float)
        add("rnd_sin", np.sin(2 * np.pi * r / 2880)); add("rnd_cos", np.cos(2 * np.pi * r / 2880))
        add("rnd_mod2", r % 2); add("rnd_mod10", r % 10)
    if gap is not None:
        add("gap_before", np.asarray(gap, float))
    return np.column_stack(cols), names


def make_model(kind, n_classes, seed=0, strong_reg=True):
    if kind == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300))
    if kind == "RandForest":
        return RandomForestClassifier(n_estimators=250, min_samples_leaf=25, max_features="sqrt", n_jobs=1, random_state=seed)
    if kind == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=250, min_samples_leaf=25, max_features="sqrt", n_jobs=1, random_state=seed)
    if kind == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=120, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.7, subsample_freq=1,
                                  colsample_bytree=0.5, reg_lambda=10, random_state=seed, verbose=-1, n_jobs=1)
    if kind == "XGBoost":
        return xgb.XGBClassifier(n_estimators=120, learning_rate=0.03, max_depth=3, min_child_weight=10, subsample=0.7, colsample_bytree=0.5,
                                 reg_lambda=10, random_state=seed, n_jobs=1, verbosity=0, tree_method="hist")
    raise ValueError(kind)


def walk_forward(X, y, n_classes, kinds, start, refit_every, seed=0, log=None):
    """Return dict kind -> (n_test x n_classes) probability matrix for rounds start..n-1; each block fitted only on rows < block start."""
    n = len(y); P = {k: np.zeros((n - start, n_classes)) for k in kinds}
    for a in range(start, n, refit_every):
        b = min(n, a + refit_every); tr = slice(20, a)           # skip first 20 rows whose lag features are padding
        for k in kinds:
            m = make_model(k, n_classes, seed); m.fit(X[tr], y[tr])
            pr = m.predict_proba(X[a:b]); full = np.zeros((b - a, n_classes))
            full[:, m.classes_.astype(int)] = pr; P[k][a - start:b - start] = full
        if log: log(f"  refit at {a}/{n}")
    return P


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def binom_p(k, n, p0):
    return stats.binomtest(int(k), int(n), p0).pvalue if n else np.nan


def score(P, y, base_acc, prior):
    pred = P.argmax(1); k = int((pred == y).sum()); n = len(y)
    ll = -np.mean(np.log(np.clip(P[np.arange(n), y], 1e-9, 1)))
    ll0 = -np.mean(np.log(prior[y]))
    lo, hi = wilson(k, n)
    return dict(n=n, acc=k / n, lo=lo, hi=hi, p=binom_p(k, n, base_acc), logloss=ll, logloss_prior=ll0)
