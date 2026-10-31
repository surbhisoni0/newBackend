"""ClockServer: the clock is served from its own pool (immune to a saturated shared pool) and falls back to the last good snapshot."""
import asyncio, os, sys, threading, time, unittest
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.clock_cache import ClockServer


class Svc:
    def __init__(self): self.t, self.fail, self.delay = 1000.0, False, 0
    def now_fn(self): return self.t
    def clock(self):
        if self.delay: time.sleep(self.delay)
        if self.fail: raise RuntimeError("db down")
        return {"server_epoch_ms": int(self.t * 1000), "server_time": "x", "countdown": {"state": "open", "seconds_remaining": 20.0}}


class T(unittest.TestCase):
    def test_fresh_each_call(self):
        async def go():
            c, s = ClockServer(), Svc()
            a = await c.get("m", s); s.t += 5; b = await c.get("m", s)
            self.assertEqual((a["server_epoch_ms"], b["server_epoch_ms"]), (1000000, 1005000)); self.assertNotIn("stale", b)
        asyncio.run(go())

    def test_not_starved_by_saturated_shared_pool(self):
        async def go():
            shared, gate = ThreadPoolExecutor(max_workers=2), threading.Event()
            loop = asyncio.get_running_loop()
            for _ in range(2): loop.run_in_executor(shared, gate.wait)          # shared pool fully busy (heavy endpoints)
            queued = loop.run_in_executor(shared, lambda: "clock-in-shared-pool")  # what the old sync route would have done
            t0 = time.time(); d = await ClockServer().get("m", Svc())
            self.assertLess(time.time() - t0, 1.0); self.assertEqual(d["countdown"]["state"], "open")
            self.assertFalse(queued.done(), "the shared pool really was blocked")
            gate.set(); await queued
        asyncio.run(go())

    def test_fallback_when_compute_fails_or_times_out(self):
        async def go():
            c, s = ClockServer(timeout=0.3), Svc()
            await c.get("m", s); s.t += 4; s.fail = True
            d = await c.get("m", s)
            self.assertTrue(d["stale"]); self.assertEqual(d["server_epoch_ms"], 1004000); self.assertAlmostEqual(d["countdown"]["seconds_remaining"], 16.0)
            s.fail, s.delay = False, 1.0; s.t += 1
            d = await c.get("m", s); self.assertTrue(d["stale"], "timeout also falls back")
            with self.assertRaises(Exception): await ClockServer().get("other", _Broken())
        asyncio.run(go())


class _Broken(Svc):
    def __init__(self): super().__init__(); self.fail = True


if __name__ == "__main__":
    unittest.main()
