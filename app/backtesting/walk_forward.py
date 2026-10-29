"""Strict chronological walk-forward backtest. For each round t the engine sees ONLY rounds before t."""
import time
from typing import Callable, Optional
import numpy as np
from app.backtesting.colour_eval import colour_eval, size_eval
from app.prediction import size as sz
from app.prediction.colour import colour_probs
from app.prediction.engine import ALL_MODELS, PredictionEngine, consensus, decide
from app.utils.constants import BASELINES, COLOR_GROUP, METRICS
from app.utils.stats import rate_stats

N_BLOCKS = 4


def _model_hits(p: np.ndarray, actual: int):
    order = np.argsort(-p, kind="stable")
    size_pred = int(p[5:].sum() > p[:5].sum())
    grp = int(np.argmax([p[COLOR_GROUP == g].sum() for g in range(3)]))
    return (order[0] == actual, actual in order[:3], size_pred == int(actual >= 5), grp == COLOR_GROUP[actual])


def run_walk_forward(seq, start: int = 100, refit_every: int = 10, use_ml: bool = True,
                     progress: Optional[Callable[[int, int], None]] = None) -> dict:
    seq = np.asarray(seq, dtype=int)
    n_total = len(seq)
    if n_total <= start + 1:
        raise ValueError(f"need more than {start + 1} results to backtest")
    eng = PredictionEngine(refit_every=refit_every, use_ml=use_ml)
    names, hits, rows = None, {}, []
    logloss, cprobs, cgroups, sprobs, sgroups = {}, [], [], [], []
    for t in range(start, n_total):
        actual = int(seq[t])
        scores = eng.model_scores(seq[:t])
        if names is None:
            names = list(scores)
            hits = {k: {m: [] for m in METRICS} for k in names + ["Consensus"]}
            logloss = {k: [] for k in names}
        for k, p in scores.items():
            for m, h in zip(METRICS, _model_hits(p, actual)):
                hits[k][m].append(bool(h))
            pp = np.clip(p / p.sum(), 1e-4, 1)
            logloss[k].append(-float(np.log(pp[actual])))
        cprobs.append(colour_probs(scores)); cgroups.append(int(COLOR_GROUP[actual]))
        cons = consensus(scores); d = decide(cons)
        sp = sz.size_probs(scores); d["size"] = sz.size_pick(sp); sprobs.append(sp); sgroups.append(int(actual >= 5))
        ch = (d["top1"] == actual, actual in d["top3"], d["size"] == ("Big" if actual >= 5 else "Small"),
              d["color_group"] == COLOR_GROUP[actual])
        for m, h in zip(METRICS, ch):
            hits["Consensus"][m].append(bool(h))
        rows.append({"target_seq": t + 1, "actual": actual, "top1": d["top1"], "top3": d["top3"], "size": d["size"],
                     "color": d["color"], "top1_hit": bool(ch[0]), "top3_hit": bool(ch[1]),
                     "bs_hit": bool(ch[2]), "color_hit": bool(ch[3])})
        if progress and (t - start) % 25 == 0:
            progress(t - start, n_total - start)
    summary = summarize(hits, logloss, n_total, start, refit_every)
    summary["colour_eval"] = colour_eval(np.array(cprobs), cgroups)
    summary["size_eval"] = size_eval(np.array(sprobs), sgroups)
    return {"summary": summary, "rows": rows}


def summarize(hits: dict, logloss: dict, n_total: int, start: int, refit_every: int) -> dict:
    models = {}
    for k, mh in hits.items():
        models[k] = {m: rate_stats(mh[m], BASELINES[m], N_BLOCKS) for m in METRICS}
        if k in logloss:
            ll = np.array(logloss[k]); d = ll - np.log(10)
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            models[k]["logloss_vs_uniform"] = {"mean_diff": float(d.mean()), "lo": float(d.mean() - 1.96 * se),
                                                "hi": float(d.mean() + 1.96 * se)}
    cons = hits["Consensus"]
    n = len(cons["top1"])
    roll = {}
    w = 100
    for m in METRICS:
        a = np.array(cons[m], dtype=float)
        roll[m] = [float(a[i - w:i].mean()) for i in range(w, n + 1, 10)] if n >= w else []
    return {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data_n": int(n_total), "start": start,
            "refit_every": refit_every, "n_predictions": n, "n_blocks": N_BLOCKS,
            "K": len(models) * len(METRICS),        # number of (model, metric) tests => Bonferroni denominator
            "baselines": BASELINES, "models": models, "consensus": models["Consensus"],
            "rolling_window": w, "consensus_rolling": roll}
