"""Prediction engine.

Produces per-model scores, an UNWEIGHTED rank-average consensus, and Top-3 / Big-Small / Colour candidates.
Nothing here is a calibrated probability. Whether the output deserves any trust is decided by signal.py from
walk-forward validation, never by the model itself.
"""
import logging
import threading
import numpy as np
from app.prediction import baselines
from app.prediction import size as sz
from app.prediction.colour import as_dict, colour_probs
from app.prediction.features import feats
from app.prediction.ml_models import make_models
from app.utils.constants import COLOR_GROUP, COLOR_NAMES

log = logging.getLogger("wingo.engine")
ENGINE_VERSION = "1.0.0"
BASELINE_MODELS = ("CumFreq", "Recent50", "Markov1", "Markov2")
ML_MODELS = ("LogReg", "RandForest", "HistGB")
ALL_MODELS = BASELINE_MODELS + ML_MODELS


def rank_scores(p: np.ndarray) -> np.ndarray:
    """1.0 for the best-ranked number down to 0.0 for the worst (position based, not a probability)."""
    order = np.argsort(-np.asarray(p), kind="stable")
    r = np.empty(10); r[order] = np.arange(10)
    return 1.0 - r / 9.0


def consensus(scores: dict) -> np.ndarray:
    """Equal-weight mean of each model's rank scores. No tuned weights (none were justified by validation)."""
    return np.mean([rank_scores(p) for p in scores.values()], axis=0)


def decide(cons: np.ndarray) -> dict:
    order = [int(x) for x in np.argsort(-cons, kind="stable")]
    big, small = float(cons[5:].sum()), float(cons[:5].sum())
    groups = [float(cons[COLOR_GROUP == g].sum()) for g in range(3)]
    return {"order": order, "top1": order[0], "top3": order[:3],
            "size": "Big" if big > small else "Small",
            "color_group": int(np.argmax(groups)), "color": COLOR_NAMES[int(np.argmax(groups))]}


class PredictionEngine:
    def __init__(self, refit_every: int = 10, warm: int = 20, use_ml: bool = True, min_ml_history: int = 60,
                 background_refit: bool = False):
        """background_refit=True (live server only): once a first fit exists, a due refit runs in a worker thread on a snapshot of the data and the
        request keeps using the current models until the new ones are swapped in. Default False = refit inline (backtests, tests, scripts)."""
        self.refit_every, self.warm, self.use_ml, self.min_ml_history = refit_every, warm, use_ml, min_ml_history
        self.background_refit = background_refit
        self._models = None; self._fit_n = -1
        self._fit_lock = threading.Lock()                                  # one fit at a time; also guards the feature-row cache
        self._spawn_lock, self._refit_running = threading.Lock(), False
        self._cache_seq = np.zeros(0, dtype=int); self._cache_X = []      # feature-row cache (append-only data)

    def _rows(self, seq: np.ndarray, upto: int):
        """Feature rows for t = warm..upto-1; row t is built from seq[:t] only. Cached while the data prefix is unchanged."""
        m = min(len(self._cache_seq), len(seq))
        if m and not np.array_equal(self._cache_seq[:m], seq[:m]):
            m, self._cache_X = 0, []
        valid = min(len(self._cache_X), max(0, m - self.warm + 1))       # rows whose input prefix (t <= m) is unchanged
        self._cache_X = self._cache_X[:valid]
        for t in range(self.warm + valid, upto):
            self._cache_X.append(feats(seq[:t]))
        self._cache_seq = seq.copy()
        return np.array(self._cache_X[: upto - self.warm])

    def _fit(self, seq: np.ndarray):
        n = len(seq)
        X, y = self._rows(seq, n), seq[self.warm:n]
        models = make_models()
        for m in models.values():
            m.fit(X, y)
        self._models, self._fit_n = models, n

    def _due(self, n: int) -> bool:
        return self._models is None or n < self._fit_n or n - self._fit_n >= self.refit_every

    def ready(self, n: int) -> bool:
        """Live (background_refit) mode only: False while the FIRST fit is still missing, so callers can answer "warming up" instead of
        blocking on it. Always True in the default inline mode, where a missing fit is simply made on demand."""
        return (not self.background_refit) or (not self.use_ml) or n < self.min_ml_history or self._models is not None

    def warm_async(self, seq) -> None:
        """Start (or join) the first fit in a worker thread; never blocks the caller."""
        self._refit_in_background(np.asarray(seq, dtype=int))

    def wait_ready(self, timeout: float) -> bool:
        """Wait (bounded) for the first fit. Returns whether the models exist."""
        import time
        end = time.time() + max(0.0, timeout)
        while self._models is None and time.time() < end:
            time.sleep(0.05)
        return self._models is not None

    def _refit_in_background(self, h: np.ndarray) -> None:
        with self._spawn_lock:
            if self._refit_running:
                return
            self._refit_running = True
        snap = h.copy()

        def work():
            try:
                with self._fit_lock:
                    if self._due(len(snap)):
                        self._fit(snap)
            except Exception:                                              # noqa: BLE001 - the current models stay in use
                log.warning("background model refit failed", exc_info=True)
            finally:
                with self._spawn_lock:
                    self._refit_running = False
        threading.Thread(target=work, daemon=True, name="refit").start()

    def model_scores(self, seq) -> dict:
        h = np.asarray(seq, dtype=int)
        if len(h) < 2:
            raise ValueError("need at least 2 results")
        out = {"CumFreq": baselines.cum_freq(h), "Recent50": baselines.recent_freq(h, 50),
               "Markov1": baselines.markov1(h), "Markov2": baselines.markov2(h)}
        if self.use_ml and len(h) >= self.min_ml_history:
            models = self._models
            if self._due(len(h)):
                if self.background_refit and models is not None and len(h) >= self._fit_n:
                    self._refit_in_background(h)                           # keep answering with the current models meanwhile
                else:
                    with self._fit_lock:
                        if self._due(len(h)):                              # another thread may have fitted while we waited
                            self._fit(h)
                    models = self._models
            x = feats(h).reshape(1, -1)
            for k, m in models.items():
                pr = np.zeros(10)
                classes = m[-1].classes_ if hasattr(m, "steps") else m.classes_
                pr[classes] = m.predict_proba(x)[0]
                out[k] = pr
        return out

    def predict(self, seq) -> dict:
        scores = self.model_scores(seq)
        cons = consensus(scores)
        d = decide(cons)
        d["size"] = sz.size_pick(sz.size_probs(scores))
        return {
            "engine_version": ENGINE_VERSION, "n_history": int(len(seq)),
            "top1": d["top1"], "top3": d["top3"], "size": d["size"], "color": d["color"], "color_group": d["color_group"],
            "color_probs": as_dict(colour_probs(scores)),
            "size_probs": sz.size_dict(sz.size_probs(scores)),
            "ranking": [{"number": n, "score": round(float(cons[n]), 4)} for n in d["order"]],
            "model_tops": {k: [int(x) for x in np.argsort(-p, kind="stable")[:3]] for k, p in scores.items()},
            "model_scores": {k: [round(float(x), 4) for x in p] for k, p in scores.items()},
        }
