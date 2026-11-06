"""Tests for the research harness: it must (a) never use future data, (b) find a planted signal, (c) not invent one in noise."""
import numpy as np
import pytest
from app.backtesting import research as R


def _rand(n, seed=0):
    return np.random.default_rng(seed).integers(0, 10, n)


def test_seq_models_use_only_past():
    """Changing any round at or after t must not change the prediction for t."""
    h = _rand(600)
    base = R.seq_models(h, 400, 450)
    h2 = h.copy(); h2[430:] = (h2[430:] + 3) % 10                     # rewrite the 'future' relative to round 430
    alt = R.seq_models(h2, 400, 450)
    for name in base:
        assert np.allclose(base[name][:30], alt[name][:30]), name      # rounds 400..429 only see h[:t] with t <= 429


def test_rich_features_use_only_past():
    h = _rand(400)
    a = R.rich_features(h, 300)
    h2 = h.copy(); h2[300:] = 0
    assert np.allclose(a, R.rich_features(h2, 300))


def test_walk_forward_never_trains_on_target_block():
    seen = {}
    def fp(Xtr, ytr, Xte):
        seen["max_train_row"] = len(Xtr)
        return np.full(len(Xte), 0.5)
    X = np.arange(300).reshape(-1, 1).astype(float); y = np.zeros(300, dtype=int)
    R.wf_predict_proba(fp, X, y, 100, 200, 50)
    assert seen["max_train_row"] == 150                               # last block starts at 150 and trains on rows < 150 only


def test_planted_signal_is_detected():
    h = R.planted_sequence(4000, 0.62, seed=3); y = (h >= 5).astype(int)
    pb = R.seq_models(h, 3000, 4000)["BS-Markov1"]
    s = R.score(pb, y[3000:])
    assert s["direct"] > 0.55 and s["p_two_sided"] < 0.01


def test_noise_is_not_reported_as_signal():
    hits = 0
    for sd in range(20):
        h = _rand(2500, sd); y = (h >= 5).astype(int)
        pb = R.seq_models(h, 1500, 2500)["BS-Markov1"]
        hits += R.score(pb, y[1500:])["p_two_sided"] < 0.01
    assert hits <= 2                                                   # ~1% false positives expected


def test_reverse_is_exact_complement():
    y = (_rand(500) >= 5).astype(int); pb = np.random.default_rng(1).random(500)
    s = R.score(pb, y)
    assert s["direct"] + s["reverse"] == pytest.approx(1.0)


def test_power_check_shape():
    pc = R.power_check(n=1500, final_n=300, seeds=(0,))
    assert len(pc["planted"]) == 4 and pc["needed_for_80pct_accuracy_z"] > 10


def test_research_endpoint_available(monkeypatch):
    import os
    os.environ.update(DATABASE_URL="memory://", FEED_ENABLED="false", AUTO_SEED="false", AUTO_BACKTEST="false", SYNC_ENABLED="false", ADMIN_API_KEY="test-key")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        r = c.get("/api/research/report")
        assert r.status_code == 200 and "available" in r.json()
