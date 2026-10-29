import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.sync.service import SyncService, plan_ingest
from app.sync.sources import HttpJsonSource, RoundRecord, SourceError, parse_records, robots_allows

R = RoundRecord


class FakeSource:
    configured, name, last_invalid = True, "fake", 0
    def __init__(self, records=None, error=None): self.records, self.error = records or [], error
    def fetch(self):
        if self.error: raise SourceError(self.error)
        return list(self.records)


def make_service(with_periods=True, source=None, **kw):
    svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), min_history=100, min_validation_n=300)
    svc.repo.init()
    svc.import_rows([{"number": (i * 7 + i // 3) % 10, "period_id": str(5001 + i) if with_periods else None, "timestamp": None} for i in range(130)])
    return svc, SyncService(svc, source or FakeSource(), enabled=True, **kw)


class TestParsing(unittest.TestCase):
    def test_valid_and_invalid_items_are_separated(self):
        payload = {"data": {"list": [{"id": "9001", "n": 5}, {"id": "9002", "n": "7"}, {"id": "x", "n": 1}, {"id": "9004", "n": 12},
                                     {"id": "9005", "n": 2.5}, {"id": "9006"}]}}
        good, bad = parse_records(payload, "data.list", "id", "n")
        self.assertEqual([(r.period_id, r.number) for r in good], [("9001", 5), ("9002", 7)])
        self.assertEqual(bad, 4)

    def test_wrong_path_is_an_error(self):
        with self.assertRaises(SourceError): parse_records({"a": []}, "data.list", "id", "n")

    def test_robots(self):
        self.assertFalse(robots_allows("User-agent: *\nDisallow: /", "https://x.test/api/r", "Bot"))
        self.assertTrue(robots_allows("User-agent: *\nDisallow: /private", "https://x.test/api/r", "Bot"))

    def test_http_source_refuses_when_robots_disallow(self):
        class S(HttpJsonSource):
            def _http_get(self, url): return (200, "User-agent: *\nDisallow: /") if url.endswith("robots.txt") else (200, "[]")
        with self.assertRaises(SourceError) as cm: S("https://x.test/api/r", "", "id", "n").fetch()
        self.assertIn("robots", str(cm.exception))

    def test_http_source_not_authorised_and_unconfigured(self):
        class S(HttpJsonSource):
            def _http_get(self, url): return (404, "") if url.endswith("robots.txt") else (403, "")
        with self.assertRaises(SourceError): S("https://x.test/api/r", "", "id", "n").fetch()
        self.assertFalse(HttpJsonSource().configured)


class TestPlan(unittest.TestCase):
    def test_orders_dedupes_and_ignores_old(self):
        recs = [R("5133", 1), R("5131", 2), R("5131", 2), R("5130", 3), R("5132", 4)]
        p = plan_ingest(recs, "5130", lambda x: False)
        self.assertEqual([r.period_id for r in p["ingest"]], ["5131", "5132", "5133"])
        self.assertEqual((p["duplicates"], p["ignored_old"], p["gap"]), (1, 1, None))

    def test_gap_stops_ingestion_unless_allowed(self):
        recs = [R("5131", 1), R("5132", 2), R("5135", 3)]
        p = plan_ingest(recs, "5130", lambda x: False)
        self.assertEqual([r.period_id for r in p["ingest"]], ["5131", "5132"])
        self.assertEqual(p["gap"], {"expected": "5133", "first_available": "5135"})
        self.assertEqual(len(plan_ingest(recs, "5130", lambda x: False, allow_gaps=True)["ingest"]), 3)


class TestSyncLoop(unittest.TestCase):
    def test_new_round_resolves_prediction_and_predicts_next(self):
        svc, sync = make_service()
        pending = svc.next_prediction(); self.assertEqual(pending["target_period"], "5131")
        sync.source = FakeSource([R("5131", 4), R("5130", 9)])
        st = sync.run_once()
        self.assertEqual((st["state"], st["last_ingested"], st["expected_next_round"]), ("ok", 1, "5132"))
        got = svc.repo.get_prediction_by_target(131)
        self.assertEqual(got["status"], "resolved"); self.assertEqual(got["outcome"]["actual"], 4)
        nxt = svc.next_prediction(); self.assertEqual((nxt["target_period"], nxt["target_seq"]), ("5132", 132))
        self.assertEqual(nxt["n_history"], 131)                     # the new prediction used only rounds up to 5131
        self.assertAlmostEqual(sum(nxt["color_probs"].values()), 1.0, places=2)

    def test_repeat_sync_creates_no_duplicates(self):
        svc, sync = make_service(source=FakeSource([R("5131", 4)]))
        sync.run_once(); n = len(svc.repo.get_numbers())
        st = sync.run_once()
        self.assertEqual((len(svc.repo.get_numbers()), st["last_ingested"]), (n, 0))
        targets = [r["target_seq"] for r in svc.repo.list_predictions(100, 0)[0]]
        self.assertEqual(len(targets), len(set(targets)))          # never two predictions for the same round

    def test_gap_is_blocked_and_reported(self):
        svc, sync = make_service(source=FakeSource([R("5133", 4)]))
        st = sync.run_once()
        self.assertEqual(st["state"], "gap_blocked"); self.assertEqual(len(svc.repo.get_numbers()), 130)

    def test_source_failure_is_reported_not_hidden(self):
        svc, sync = make_service(source=FakeSource(error="HTTP 500"))
        st = sync.run_once(); st = sync.run_once()
        self.assertEqual((st["state"], st["consecutive_failures"]), ("error", 2)); self.assertEqual(len(svc.repo.get_numbers()), 130)

    def test_history_without_round_ids_needs_an_anchor(self):
        svc, sync = make_service(with_periods=False, source=FakeSource([R("9001", 4)]))
        self.assertEqual(sync.run_once()["state"], "needs_anchor"); self.assertEqual(len(svc.repo.get_numbers()), 130)
        sync.start_after = "9000"
        self.assertEqual(sync.run_once()["state"], "ok"); self.assertEqual(len(svc.repo.get_numbers()), 131)

    def test_unconfigured_source(self):
        svc, sync = make_service(source=HttpJsonSource())
        self.assertEqual(sync.run_once()["state"], "not_configured")


if __name__ == "__main__":
    unittest.main()
