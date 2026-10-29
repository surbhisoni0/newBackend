# Live regression tests against running Wingo server (http://localhost:8099): SEC-001 fail-closed guard, streaks, daily summary
import os
from collections import Counter
import pytest
import requests

BASE = os.environ.get("WINGO_URL", "http://localhost:8099")
COLOR_GROUP = [2, 0, 1, 0, 1, 2, 1, 0, 1, 0]
NAMES = ["Green", "Red", "Violet"]


@pytest.fixture(scope="module")
def all_rows():
    rows, off = [], 0
    while True:
        r = requests.get(f"{BASE}/api/results", params={"limit": 500, "offset": off}).json()
        rows += r["results"]
        if len(r["results"]) < 500:
            break
        off += 500
    return rows  # newest first


@pytest.fixture(scope="module")
def report():
    r = requests.get(f"{BASE}/api/stats/report")
    assert r.status_code == 200
    return r.json()


# --- Security: writes disabled when ADMIN_API_KEY blank -> 503, no data changes
WRITES = [
    ("/api/results", {"json": {"number": 3}}),
    ("/api/data/import", {"json": {"rows": [{"number": 3}]}}),
    ("/api/data/import/csv", {"files": {"file": ("a.csv", b"number\n3\n", "text/csv")}}),
    ("/api/data/import/paste", {"json": {"text": "3 4 5"}}),
    ("/api/sync/run", {}),
    ("/api/backtest/run", {}),
]


@pytest.mark.parametrize("path,kw", WRITES)
@pytest.mark.parametrize("key", [None, "anything", ""])
def test_write_blocked_503(path, kw, key):
    if os.environ.get("EXPECT_KEY"):
        pytest.skip("server running with key")
    h = {"X-API-Key": key} if key is not None else {}
    r = requests.post(f"{BASE}{path}", headers=h, **kw)
    assert r.status_code == 503, (path, r.status_code, r.text)
    assert "ADMIN_API_KEY" in r.json()["detail"]


def test_total_unchanged_and_status():
    s = requests.get(f"{BASE}/api/status").json()
    assert s["results"] == 5086
    assert requests.get(f"{BASE}/api/results?limit=1").json()["total"] == 5086
    if not os.environ.get("EXPECT_KEY"):
        assert s["writes_enabled"] is False


def test_gets_work():
    for p in ["/api/health", "/api/status", "/api/results/latest", "/api/clock", "/api/sync/status", "/api/data/stats",
              "/api/prediction/next", "/api/prediction/history", "/api/backtest", "/api/model-performance", "/api/stats/report"]:
        r = requests.get(f"{BASE}{p}")
        assert r.status_code == 200, (p, r.status_code, r.text[:200])


# --- With key configured (run with EXPECT_KEY=testkey123)
def test_keyed_auth():
    key = os.environ.get("EXPECT_KEY")
    if not key:
        pytest.skip("no key")
    assert requests.post(f"{BASE}/api/sync/run").status_code == 401
    assert requests.post(f"{BASE}/api/sync/run", headers={"X-API-Key": "wrong"}).status_code == 401
    assert requests.post(f"{BASE}/api/results", json={"number": 1}, headers={"X-API-Key": "wrong"}).status_code == 401
    r = requests.post(f"{BASE}/api/sync/run", headers={"X-API-Key": key})
    assert r.status_code not in (401, 503), r.text
    assert requests.get(f"{BASE}/api/status").json()["writes_enabled"] is True


# --- Streaks
def _run(labels):
    n = 0
    for x in labels:
        if x != labels[0]:
            break
        n += 1
    return labels[0], n


def test_streaks(report, all_rows):
    st = report["streaks"]
    assert st["threshold"] == 5
    nums = [r["number"] for r in all_rows]
    v, n = _run(["Big" if x >= 5 else "Small" for x in nums])
    assert st["size"] == {"value": v, "length": n, "alert": n >= 5}
    v, n = _run(["Green" if x % 2 else "Red" for x in nums])
    assert st["colour"] == {"value": v, "length": n, "alert": n >= 5}
    assert st["colour"]["value"] == "Green" and st["colour"]["length"] == 6 and st["colour"]["alert"] is True


# --- Daily summary
def test_daily(report, all_rows):
    d = report["daily"]
    assert [x["date"] for x in d] == ["2026-10-06", "2026-09-28", "2026-09-27", "2026-09-26", "2026-09-25", "2026-09-24"]
    assert sum(x["rounds"] for x in d) == 5086
    by = {}
    for r in all_rows:
        p = r["period_id"]
        by.setdefault(f"{p[:4]}-{p[4:6]}-{p[6:8]}", []).append(r["number"])
    for x in d:
        nums = by[x["date"]]
        c = Counter(nums)
        assert x["rounds"] == len(nums)
        hot = sorted(range(10), key=lambda k: (-c[k], k))[:3]
        cold = sorted(range(10), key=lambda k: (c[k], k))[:3]
        assert [h["number"] for h in x["hot"]] == hot and [h["count"] for h in x["hot"]] == [c[k] for k in hot]
        assert [h["number"] for h in x["cold"]] == cold and [h["count"] for h in x["cold"]] == [c[k] for k in cold]
        cc = Counter(COLOR_GROUP[n] for n in nums)
        assert x["colours"] == {NAMES[i]: cc[i] for i in range(3)}
        assert x["size"]["Big"] == sum(1 for n in nums if n >= 5)


def test_accuracy_and_hotcold(report):
    a = report["accuracy"]
    assert a["available"] is True
    assert set(report["hot_cold"].keys()) == {"50", "100"}
    assert report["total_results"] == 5086
