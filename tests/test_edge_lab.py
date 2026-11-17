"""Guards for the research lab: (1) no look-ahead in features, (2) the harness CAN detect a planted edge, (3) it does NOT invent one in noise."""
import numpy as np, pytest
pytest.importorskip("lightgbm"); pytest.importorskip("xgboost")
from app.research_lab import edge_lab as lab


def _fake(n=900, seed=0):
    r = np.random.default_rng(seed); return r.integers(0, 10, n), np.arange(n) % 2880, np.zeros(n, int)


def test_features_never_use_future_rounds():
    h, rnd, gap = _fake(); X, _ = lab.features(h, rnd, gap); r = np.random.default_rng(1)
    for t in (60, 200, 431, 700):
        h2 = h.copy(); h2[t:] = r.integers(0, 10, len(h) - t)
        X2, _ = lab.features(h2, rnd, gap)
        assert np.allclose(X[t], X2[t]), f"row {t} depends on round >= t"


def _run(big, n):
    h = np.where(big == 1, 7, 2); X, _ = lab.features(h)
    P = lab.walk_forward(X, big, 2, ["LogReg"], 400, 250); return ((P["LogReg"][:, 1] > 0.5) == big[400:]).mean(), n - 400


def test_detects_planted_repeat_edge():
    r = np.random.default_rng(5); n = 1800; big = np.zeros(n, int)
    for t in range(1, n): big[t] = big[t - 1] if r.random() < 0.4 else r.integers(2)      # true ceiling 70%
    acc, m = _run(big, n); assert acc > 0.64


def test_no_edge_found_in_pure_noise():
    r = np.random.default_rng(6); n = 1800; acc, m = _run(r.integers(0, 2, n), n)
    assert abs(acc - 0.5) < 3.5 * 0.5 / np.sqrt(m)
