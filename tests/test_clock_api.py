"""HTTP-level checks for /api/clock and the timing block on /api/prediction/next (real wall clock, in-memory database)."""
import os, sys, time, unittest
from datetime import datetime, timezone
os.environ.update(DATABASE_URL="memory://", FEED_ENABLED="false", AUTO_SEED="false", AUTO_BACKTEST="false", SYNC_ENABLED="false", ADMIN_API_KEY="test-key", ASSUMED_SCHEDULE="false")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import importlib
    from fastapi.testclient import TestClient
    import app.config
    importlib.reload(app.config)                                       # pick up this test's env even if another test loaded config first
    import app.main
    importlib.reload(app.main)
    from app.main import app
    HAVE = True
except Exception:                                                        # noqa: BLE001
    HAVE = False


def iso(e): return datetime.fromtimestamp(e, timezone.utc).isoformat()


@unittest.skipUnless(HAVE, "fastapi/httpx not installed")
class TestClockApi(unittest.TestCase):
    def test_clock_endpoint_end_to_end(self):
        with TestClient(app) as c:
            empty = c.get("/api/clock").json()
            self.assertEqual(empty["countdown"]["state"], "no_data"); self.assertEqual(c.get("/api/clock").headers["cache-control"], "no-store")
            end_of_latest = time.time() - 12                                   # newest confirmed round ended 12 s ago
            rows = [{"number": (i * 7 + i // 3) % 10, "period_id": str(700001 + i), "timestamp": iso(end_of_latest - (129 - i) * 60)} for i in range(130)]
            key = app.state.settings.admin_api_key          # several test modules set this env var at import time; use what the app loaded
            r = c.post("/api/data/import", json={"rows": rows}, headers={"X-API-Key": key} if key else {}); self.assertEqual(r.status_code, 200, r.text)
            ck = c.get("/api/clock").json()
            self.assertTrue(ck["schedule"]["verified"]); self.assertEqual(ck["next_round"]["period_id"], "700131")
            self.assertEqual(ck["countdown"]["state"], "open"); self.assertTrue(40 < ck["countdown"]["seconds_remaining"] < 50, ck["countdown"])
            self.assertLess(abs(ck["server_epoch_ms"] / 1000 - time.time()), 5)
            nxt = c.get("/api/prediction/next"); body = nxt.json()
            self.assertEqual(nxt.headers["cache-control"], "no-store")
            self.assertEqual((body["target_period"], body["estimate_state"], body["target_id_kind"]), ("700131", "open", "period_id"))
            self.assertEqual(body["timing"]["next_round"]["period_id"], body["target_period"])
            again = c.get("/api/prediction/next").json(); self.assertEqual((again["id"], again["created_at"]), (body["id"], body["created_at"]))


if __name__ == "__main__":
    unittest.main()
