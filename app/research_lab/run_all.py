"""Reproduce the full edge study:   python -m app.research_lab.run_all   (from Backend dir, ~3 min on 1 CPU)
Writes data/edge_lab_report.json. Needs lightgbm + xgboost (pip install lightgbm xgboost)."""
import json, pickle, numpy as np
from scipy import stats
from app.research_lab import edge_lab as lab

START, REFIT = 1500, 250
KINDS = {"bs": ["LogReg", "RandForest", "ExtraTrees", "LightGBM", "XGBoost"], "color": ["LogReg", "RandForest", "LightGBM", "XGBoost"],
         "number": ["LogReg", "RandForest", "LightGBM"]}
CHANCE = {"bs": 0.5, "color": 0.4, "number": 0.1}


def main(path="data/wingo_docx_clean.csv", out="data/edge_lab_report.json"):
    d = lab.load(path); h = d.number.values
    X, _ = lab.features(h, d.rnd.values, d.gap.values)
    y = {"bs": (h >= 5).astype(int), "color": lab.COLOR_GROUP[h], "number": h}; ncl = {"bs": 2, "color": 3, "number": 10}
    saved = {"start": START, "y": {}}
    rep = {"rounds": len(h), "walk_forward_start": START, "refit_every": REFIT, "targets": {}}
    for tgt in ("bs", "color", "number"):
        P = lab.walk_forward(X, y[tgt], ncl[tgt], KINDS[tgt], START, REFIT); saved[tgt] = dict(P); saved["y"][tgt] = y[tgt][START:]; P["Ensemble"] = np.mean(list(P.values()), axis=0)
        yt = y[tgt][START:]; prior = np.bincount(y[tgt][:START], minlength=ncl[tgt]) / START; half = len(yt) // 2; K = len(P)
        rep["targets"][tgt] = {}
        for k, p in P.items():
            s = lab.score(p, yt, CHANCE[tgt], prior)
            rep["targets"][tgt][k] = {**s, "p_bonferroni": min(1, s["p"] * K), "dev_acc": lab.score(p[:half], yt[:half], CHANCE[tgt], prior)["acc"],
                                      "final_acc": lab.score(p[half:], yt[half:], CHANCE[tgt], prior)["acc"]}
            print(tgt, k, round(s["acc"] * 100, 2), "p_bonf", round(min(1, s["p"] * K), 3), flush=True)
    pickle.dump(saved, open("main_preds.pkl", "wb"))   # input for confidence_filter.py
    json.dump(rep, open(out, "w"), indent=1, default=float)
    return rep


if __name__ == "__main__":
    main()
