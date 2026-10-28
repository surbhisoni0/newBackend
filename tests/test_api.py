"""HTTP-level tests. They need fastapi + httpx (pip install -r backend/requirements.txt) and are skipped otherwise.
They run against an in-memory database with auto-seed off, never against any website."""
import os, sys, unittest
os.environ.update(DATABASE_URL="memory://", AUTO_SEED="false", AUTO_BACKTEST="false", SYNC_ENABLED="false", ADMIN_API_KEY="test-key")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from fastapi.testclient import TestClient
    from app.main import app
    HAVE = True
except Exception:                                                        # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "fastapi/httpx not installed")
class TestApi(unittest.TestCase):
    def test_flow(self):
        with TestClient(app) as c:
            self.assertEqual(c.get("/api/health").json()["status"], "ok")
            self.assertEqual(c.get("/api/results/latest").json(), {"available": False})
            self.assertEqual(c.get("/api/sync/status").json()["state"], "disabled")
            self.assertEqual(c.post("/api/data/import/paste", content="5\n2\n").status_code, 401)       # key required for writes
            lines = "\n".join(str((i * 7 + i // 3) % 10) for i in range(130))
            r = c.post("/api/data/import/paste", content=lines, headers={"X-API-Key": "test-key"}); self.assertEqual(r.status_code, 200)
            latest = c.get("/api/results/latest").json(); self.assertTrue(latest["available"]); self.assertEqual(latest["seq"], 130)
            nxt = c.get("/api/prediction/next").json(); self.assertTrue(nxt["available"]); self.assertEqual(nxt["target_seq"], 131)
            self.assertAlmostEqual(sum(nxt["color_probs"].values()), 1.0, places=2)
            self.assertEqual(c.post("/api/sync/run").status_code, 401)
            self.assertIn("colour", c.get("/api/data/stats").json())
            self.assertEqual(c.post("/api/data/import/paste", content="12", headers={"X-API-Key": "test-key"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
