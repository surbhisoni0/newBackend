"""Timing honesty, provenance, per-mode intervals, the strict 30s adapter, mode isolation and scoring integrity.
Time is always injected; no network; in-memory (or fake) databases only."""
import json, os, sys, unittest
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import pymongo_stub  # noqa: F401
from fake_mongo import FakeClient
from app.database.factory import create_repository
from app.database.memory_repo import InMemoryRepository
from app.database.mongo_repo import MongoRepository
from app.modes import DEFAULT_MODE, MODES, get_mode, namespace_of
from app.prediction.engine import PredictionEngine
from app.services.errors import Conflict
from app.services.predictor_service import PredictorService
from app.sync.service import SyncService
from app.sync.sources import RoundRecord, SourceError, Wingo30ApiSource, parse_wingo30
from app.timing import ClockConfig, consecutive_run, countdown, expected_next_period, verify_schedule

T0 = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc).timestamp()        # the newest round ENDS here


def iso(e): return datetime.fromtimestamp(e, timezone.utc).isoformat()


def rows_nf(n, P=60, newest_ts=T0, newest_id=900100, src="live", ts_fn=None):
    """n stored rows, newest first, ids consecutive, timestamps spaced P seconds."""
    return [{"period_id": str(newest_id - k), "timestamp": (ts_fn(k) if ts_fn else iso(newest_ts - k * P)), "ts_source": src} for k in range(n)]


def make(mode="1m", n=130, now=T0 + 5, src="live", with_ts=True, assumed=False, P=None):
    P = P or get_mode(mode).period_seconds
    cell = [now]
    svc = PredictorService(InMemoryRepository(namespace_of(mode)), PredictionEngine(use_ml=False), 100, 300,
                           clock_cfg=ClockConfig(P, assumed=assumed), now_fn=lambda: cell[0], mode=mode)
    svc.repo.init()
    svc.import_rows([{"number": (i * 7 + i // 3) % 10, "period_id": str(900001 + i), "timestamp": iso(T0 - (n - 1 - i) * P) if with_ts else None}
                     for i in range(n)], source=src)
    return svc, cell


# ------------------------------------------------------------------------------------------------ 1/2: verification needs genuine evidence
class TestEvidence(unittest.TestCase):
    def test_genuine_consecutive_live_timestamps_verify(self):
        s = verify_schedule(rows_nf(10), ClockConfig(), T0 + 5)
        self.assertTrue(s["verified"]); self.assertEqual((s["samples"], s["trusted_samples"]), (10, 10))
        self.assertAlmostEqual(s["measured_interval_seconds"], 60.0)

    def test_perfectly_spaced_timestamps_are_NOT_evidence_unless_their_origin_is_trusted(self):
        for src in ("seed", "import", "manual", None):
            s = verify_schedule(rows_nf(30, src=src), ClockConfig(), T0 + 5)
            self.assertFalse(s["verified"], src); self.assertIn("not evidence", s["reason"])

    def test_attested_origin_counts_but_is_labelled(self):
        s = verify_schedule(rows_nf(10, src="attested"), ClockConfig(), T0 + 5)
        self.assertTrue(s["verified"]); self.assertEqual(s["timestamp_origin"], "attested")

    def test_legacy_rows_without_recorded_origin_never_verify(self):
        rows = rows_nf(30); [r.pop("ts_source") for r in rows]
        s = verify_schedule(rows, ClockConfig(), T0 + 5)
        self.assertFalse(s["verified"]); self.assertIn("before origins were recorded", s["reason"])

    def test_one_untrusted_row_at_the_newest_end_breaks_the_run(self):
        rows = rows_nf(12); rows[0]["ts_source"] = "manual"                          # a hand-typed newest round
        self.assertFalse(verify_schedule(rows, ClockConfig(), T0 + 5)["verified"])

    def test_the_old_generated_timestamp_pattern_is_rejected_end_to_end(self):
        """The previous docx_to_csv.py stamped ids with 'last full minute at conversion time, minus 60s per id'. Loaded as seed history
        (or from a database written by the old code) it must not produce a verified schedule or a countdown."""
        old_style = [{"number": i % 10, "period_id": str(20260924100051515 + i), "timestamp": iso(T0 - 3600 * 24 - (5000 - i) * 60)} for i in range(5000)]
        for src in ("seed", None):
            svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), clock_cfg=ClockConfig(), now_fn=lambda: T0 + 5)
            svc.repo.init()
            if src: svc.import_rows(old_style, source=src)
            else:
                from app.services.data_service import clean_rows
                svc.repo.add_results(clean_rows(old_style, None))                          # what the old code stored: no ts_source at all
            ck = svc.clock()
            self.assertEqual((ck["timing_status"], ck["schedule"]["verified"]), ("unverified", False))
            self.assertIsNone(ck["countdown"]["round_ends_at"])
            self.assertIsNone(ck["next_round"]["period_id"])                               # no invented 'next round' from months-old history
            self.assertFalse(ck["latest_confirmed"]["timestamp_trusted"])

    def test_the_shipped_seed_csv_carries_no_timestamps(self):
        import csv
        path = os.path.join(os.path.dirname(__file__), "..", "data", "wingo_docx_clean.csv")
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        self.assertGreater(len(rows), 5000)
        self.assertNotIn("timestamp", rows[0]); self.assertTrue(all(len(r["period_id"]) == 17 for r in rows[:100]))


# ------------------------------------------------------------------------------------------------ 3: invalid / future / ordering / duplicates
class TestHygiene(unittest.TestCase):
    def test_future_timestamps_are_refused_not_counted_down(self):
        s = verify_schedule(rows_nf(10, newest_ts=T0 + 3600), ClockConfig(), T0)
        self.assertFalse(s["verified"]); self.assertEqual(s["issues"]["future"], 10); self.assertIn("future", s["reason"])
        svc, _ = make(now=T0 - 7200)                                                       # server clock far behind the data
        self.assertEqual(svc.clock()["timing_status"], "unverified")

    def test_timezone_handling_is_explicit(self):
        naive = lambda k: (datetime.fromtimestamp(T0 - k * 60, timezone.utc).astimezone(__import__("zoneinfo").ZoneInfo("Asia/Kolkata"))).strftime("%Y-%m-%dT%H:%M:%S")
        rows = rows_nf(10, ts_fn=naive)                                                    # IST wall-clock with no offset
        self.assertTrue(verify_schedule(rows, ClockConfig(source_tz="Asia/Kolkata"), T0 + 5)["verified"])
        wrong = verify_schedule(rows, ClockConfig(source_tz="UTC"), T0 + 5)                # same strings read as UTC => 5.5 h in the future
        self.assertFalse(wrong["verified"]); self.assertIn("future", wrong["reason"])

    def test_out_of_order_and_duplicated_timestamps(self):
        shuffled = rows_nf(10, ts_fn=lambda k: iso(T0 - (k if k % 2 else 9 - k) * 60))
        s = verify_schedule(shuffled, ClockConfig(), T0 + 5); self.assertFalse(s["verified"]); self.assertGreater(s["issues"]["out_of_order"], 0)
        dup = verify_schedule(rows_nf(10, ts_fn=lambda k: iso(T0)), ClockConfig(), T0 + 5)
        self.assertFalse(dup["verified"]); self.assertEqual(dup["issues"]["duplicate_timestamps"], 9)

    def test_unparseable_timestamp_stops_the_run(self):
        rows = rows_nf(10); rows[1]["timestamp"] = "yesterday-ish"
        s = verify_schedule(rows, ClockConfig(), T0 + 5); self.assertEqual(s["samples"], 1); self.assertFalse(s["verified"])

    def test_inconsistent_interval_is_reported_as_wrong_mode(self):
        s = verify_schedule(rows_nf(10, P=30), ClockConfig(period_seconds=60), T0 + 5)
        self.assertFalse(s["verified"]); self.assertIn("30s apart", s["reason"]); self.assertIn("wrong game mode", s["reason"])

    def test_duplicate_and_out_of_order_ids_are_rejected_on_store(self):
        svc, _ = make()
        svc.add_result(1, "900131", iso(T0 + 60), source="live")
        with self.assertRaises(Conflict): svc.add_result(1, "900131", iso(T0 + 60), source="live")
        with self.assertRaises(Conflict): svc.add_result(1, "900050", iso(T0 - 600), source="live")

    def test_expected_id_needs_a_consecutive_run(self):
        rows = rows_nf(10)
        self.assertEqual(expected_next_period(rows, 5), "900101"); self.assertEqual(consecutive_run(rows), 10)
        rows[2]["period_id"] = "800000"
        self.assertIsNone(expected_next_period(rows, 5))                                    # broken sequence: no guess
        self.assertIsNone(expected_next_period(rows_nf(3), 5))                              # too few rows: no guess


# ------------------------------------------------------------------------------------------------ 4-7: countdown honesty
class TestCountdownHonesty(unittest.TestCase):
    def test_round_end_is_absolute_and_stable_across_refresh_and_polling(self):
        svc, cell = make()
        ends = set()
        for dt in (1, 2, 7, 13, 29, 41, 58):
            cell[0] = T0 + dt; ck = svc.clock(); ends.add(ck["countdown"]["round_ends_at"])
            self.assertEqual(ck["timing_status"], "verified")
            self.assertAlmostEqual(ck["countdown"]["seconds_remaining"], 60 - dt, places=2)
        self.assertEqual(len(ends), 1)                                                      # same boundary every time: nothing restarted

    def test_zero_is_awaiting_result_never_a_fabricated_next_round(self):
        svc, cell = make(); before = svc.clock()["next_round"]
        cell[0] = T0 + 61; ck = svc.clock()
        self.assertEqual((ck["timing_status"], ck["countdown"]["seconds_remaining"]), ("awaiting_result", 0.0))
        self.assertEqual(ck["next_round"], before)                                          # the period did not advance
        self.assertEqual(ck["latest_confirmed"]["period_id"], "900130"); self.assertEqual(svc.repo.last_result()["seq"], 130)

    def test_late_result_is_reported_as_late(self):
        svc, cell = make(); cell[0] = T0 + 60 + 45
        ck = svc.clock(); self.assertEqual(ck["timing_status"], "late"); self.assertIn("no new period is assumed", ck["countdown"]["reason"])
        cell[0] = T0 + 60 * 9; ck = svc.clock(); self.assertEqual((ck["timing_status"], ck["countdown"]["rounds_ended_since_confirmed"]), ("late", 9))

    def test_assumed_countdown_never_appears_verified(self):
        svc, _ = make(src="seed", assumed=True)
        ck = svc.clock()
        self.assertEqual((ck["timing_status"], ck["countdown"]["state"], ck["schedule"]["verified"]), ("assumed", "assumed", False))
        self.assertNotEqual(ck["countdown"]["state"], "open"); self.assertIn("not verified", ck["countdown"]["reason"].lower())
        self.assertEqual(ck["latest_confirmed"]["period_id"], "900130")                    # an assumed timer never advances a period

    def test_no_data(self):
        svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False)); svc.repo.init()
        self.assertEqual(svc.clock()["timing_status"], "no_data")

    def test_every_status_maps_from_a_state(self):
        svc, cell = make()
        seen = {svc.clock()["timing_status"]}
        for dt in (62, 200):
            cell[0] = T0 + dt; seen.add(svc.clock()["timing_status"])
        self.assertEqual(seen, {"verified", "awaiting_result", "late"})


# ------------------------------------------------------------------------------------------------ 9: four round durations
class TestModes(unittest.TestCase):
    def test_each_duration_uses_its_own_interval(self):
        for key, P in (("30s", 30), ("1m", 60), ("3m", 180), ("5m", 300)):
            self.assertEqual(MODES[key].period_seconds, P)
            svc, cell = make(mode=key, now=T0 + P / 3)
            ck = svc.clock()
            self.assertEqual((ck["mode"], ck["period_seconds"], ck["timing_status"]), (key, P, "verified"), key)
            self.assertAlmostEqual(ck["countdown"]["seconds_remaining"], P - P / 3, places=2)
            cell[0] = T0 + P + 1; self.assertEqual(svc.clock()["timing_status"], "awaiting_result")
            cell[0] = T0 + 3 * P; self.assertEqual(svc.clock()["timing_status"], "late")

    def test_data_of_one_mode_does_not_verify_another(self):
        for data, config in (("30s", 60), ("1m", 30), ("3m", 60), ("5m", 180)):
            svc, _ = make(mode="1m", now=T0 + 5, P=get_mode(data).period_seconds)
            svc.clock_cfg = ClockConfig(config)
            ck = svc.clock(); self.assertFalse(ck["schedule"]["verified"], (data, config)); self.assertIn("wrong game mode", ck["schedule"]["reason"])

    def test_modes_are_isolated_in_memory_and_in_mongo(self):
        a, b = create_repository("memory://", namespace_of("30s")), create_repository("memory://", namespace_of("1m"))
        a.add_results([{"raw": {}, "number": 3, "period_id": "1", "timestamp": None, "ts_source": "live"}])
        self.assertEqual((a.last_result()["seq"], b.last_result()), (1, None))
        client = FakeClient(); m30, m1 = MongoRepository("mongodb://f/w", client, namespace_of("30s")), MongoRepository("mongodb://f/w", client, namespace_of("1m"))
        for r in (m30, m1): r.init()
        m30.add_results([{"raw": {}, "number": 3, "period_id": "7", "timestamp": "2026-10-29T10:00:00+00:00", "ts_source": "live"}])
        self.assertEqual((m30.last_result()["seq"], m30.last_result()["ts_source"], m1.last_result()), (1, "live", None))
        m1.add_results([{"raw": {}, "number": 4, "period_id": "7", "timestamp": None, "ts_source": "seed"}])      # SAME period id, other mode: allowed
        self.assertEqual(m1.last_result()["number"], 4)
        self.assertIn("results", client.db._c); self.assertIn("results__30s", client.db._c)                      # legacy name kept for the default mode

    def test_unknown_mode_and_defaults(self):
        self.assertEqual(DEFAULT_MODE, "1m"); self.assertEqual(namespace_of("1m"), ""); self.assertEqual(namespace_of("30s"), "30s")
        with self.assertRaises(KeyError): get_mode("2m")

    def test_period_seconds_are_configurable(self):
        from app.config import Settings
        os.environ["PERIOD_SECONDS_3M"] = "181"
        try: self.assertEqual(Settings().period_seconds_for("3m"), 181); self.assertEqual(Settings().period_seconds_for("5m"), 300)
        finally: del os.environ["PERIOD_SECONDS_3M"]

    def test_a_documented_source_timestamp_convention_overrides_the_global_setting(self):
        from app.config import Settings
        cfg = Settings().clock_config("30s", Wingo30ApiSource("k"))
        self.assertEqual((cfg.period_seconds, cfg.time_means, cfg.source_tz), (30, "end", "UTC"))


# ------------------------------------------------------------------------------------------------ the strict 30s adapter
def w30(period="20260903301284", number=7, ms=None, tm=None, **kw):
    base = {"period": period, "number": number, "size": "Big" if number >= 5 else "Small",
            "colors": ["Red", "Violet"] if number == 0 else ["Green", "Violet"] if number == 5 else ["Green"] if number % 2 else ["Red"],
            "blockTimestamp": ms, "time": tm}
    base.update(kw); return base


def page(items, has_next=False, ok=True): return {"ok": ok, "data": items, "pagination": {"page": 1, "limit": 100, "hasNextPage": has_next}}


class TestWingo30Adapter(unittest.TestCase):
    def test_consistent_records_are_accepted_and_normalised_to_utc_iso(self):
        recs, bad, _ = parse_wingo30(page([w30(ms=int(T0 * 1000), tm=iso(T0).replace("+00:00", ".000Z"))]))
        self.assertEqual((len(recs), bad), (1, 0)); self.assertEqual(recs[0].timestamp, "2026-10-29T10:00:00Z")

    def test_the_vendors_own_published_example_is_refused_because_its_two_timestamps_disagree(self):
        """On https://wingo30.com/developer the sample has blockTimestamp 1788414600000 (= 05:50:00Z) but time '2026-09-03T04:10:00.000Z'
        (1 h 40 min apart). A feed like that cannot be trusted for timing, so such records are dropped, not 'fixed'."""
        recs, bad, why = parse_wingo30(page([w30(ms=1788414600000, tm="2026-09-03T04:10:00.000Z")]))
        self.assertEqual((recs, bad), ([], 1)); self.assertIn("disagree", list(why)[0])

    def test_inconsistent_or_missing_data_is_dropped_never_repaired(self):
        good = w30(ms=int(T0 * 1000))
        bad = [w30(period="abc", ms=int(T0 * 1000)), w30(number=11, ms=int(T0 * 1000)), w30(ms=None, tm=None), w30(ms=int(T0 * 1000), size="Small"),
               w30(ms=int(T0 * 1000), colors=["Red"]), "junk", w30(ms=True)]
        recs, n_bad, _ = parse_wingo30(page([good] + bad)); self.assertEqual((len(recs), n_bad), (1, 7))

    def test_provider_failures_are_errors(self):
        for payload in (page([], ok=False), [], {"ok": True, "data": "x"}):
            with self.assertRaises(SourceError): parse_wingo30(payload)

    def _src(self, responses, **kw):
        calls = []
        class S(Wingo30ApiSource):
            def _http_get(self, url):
                calls.append(url); r = responses.pop(0); self.last_headers = r[2] if len(r) > 2 else {}
                return r[0], r[1]
        s = S("secret-key", "https://example.test", retries=0, **kw); s.calls = calls; return s

    def test_fetch_uses_bearer_auth_pagination_and_never_exposes_the_key(self):
        p1 = page([w30(period="20260903301285", ms=int(T0 * 1000))], has_next=True); p2 = page([w30(period="20260903301284", ms=int((T0 - 30) * 1000))])
        s = self._src([(200, json.dumps(p1)), (200, json.dumps(p2))])
        recs = s.fetch(max_pages=3)
        self.assertEqual(sorted(r.period_id for r in recs), ["20260903301284", "20260903301285"])
        self.assertIn("page=1", s.calls[0]); self.assertIn("page=2", s.calls[1]); self.assertIn("/api/developer/30-sec-game-history", s.calls[0])
        self.assertEqual(s.headers["Authorization"], "Bearer secret-key"); self.assertEqual(s.name, "example.test"); self.assertNotIn("secret-key", s.name)

    def test_http_is_refused_and_missing_key_is_not_configured(self):
        self.assertFalse(Wingo30ApiSource("").configured)
        with self.assertRaises(SourceError) as cm: Wingo30ApiSource("k", "http://example.test").fetch()
        self.assertIn("HTTPS", str(cm.exception))

    def test_unauthorised_rate_limited_and_not_json(self):
        with self.assertRaises(SourceError) as cm: self._src([(401, "")]).fetch()
        self.assertIn("not authorised", str(cm.exception))
        with self.assertRaises(SourceError) as cm: self._src([(429, "", {"retry-after": "42"})]).fetch()
        self.assertEqual(cm.exception.retry_after, 42.0)
        with self.assertRaises(SourceError) as cm: self._src([(200, "<html>")]).fetch()
        self.assertIn("not JSON", str(cm.exception))

    def test_all_invalid_response_explains_the_most_common_problem(self):
        with self.assertRaises(SourceError) as cm: self._src([(200, json.dumps(page([w30(ms=1788414600000, tm="2026-09-03T04:10:00.000Z")] * 3)))]).fetch()
        self.assertIn("3 invalid", str(cm.exception)); self.assertIn("disagree", str(cm.exception))


# ------------------------------------------------------------------------------------------------ 10/11: sync behaviour and diagnostics
class FakeSource:
    configured, name, last_invalid, provider, supports_paging = True, "fake", 0, "fake", False
    def __init__(self, recs=None, err=None): self.recs, self.err = recs or [], err
    def fetch(self, max_pages=1):
        if self.err: raise self.err
        return list(self.recs)


def live_service(mode="30s", n=20, now=None):
    P = get_mode(mode).period_seconds
    cell = [T0 + 5 if now is None else now]
    svc = PredictorService(InMemoryRepository(namespace_of(mode)), PredictionEngine(use_ml=False), 10, 300, clock_cfg=ClockConfig(P), now_fn=lambda: cell[0], mode=mode)
    svc.repo.init()
    return svc, cell


class TestSyncIntegrity(unittest.TestCase):
    def recs(self, n, P=30, first=900001):
        return [RoundRecord(str(first + i), (i * 7 + i // 3) % 10, iso(T0 - (n - 1 - i) * P)) for i in range(n)]

    def test_live_sync_establishes_a_verified_schedule_from_real_rows_only(self):
        svc, _ = live_service()
        self.assertEqual(svc.clock()["timing_status"], "no_data")
        sync = SyncService(svc, FakeSource(self.recs(12)), enabled=True, mode="30s"); sync.run_once()
        ck = svc.clock(); self.assertEqual((ck["timing_status"], ck["schedule"]["timestamp_origin"], ck["next_round"]["period_id"]), ("verified", "live", "900013"))
        self.assertTrue(ck["latest_confirmed"]["timestamp_trusted"]); self.assertFalse(ck["next_round"]["confirmed"])

    def test_repeated_sync_never_duplicates_periods_results_or_predictions(self):
        svc, cell = live_service(); recs = self.recs(12)
        sync = SyncService(svc, FakeSource(recs), enabled=True, mode="30s"); sync.run_once()
        n = (len(svc.repo.results), len(svc.repo.preds))
        for _ in range(4): sync.run_once()
        self.assertEqual((len(svc.repo.results), len(svc.repo.preds)), n); self.assertEqual(sync.s["ingested_total"], 12)
        self.assertEqual(len({r["period_id"] for r in svc.repo.results}), 12)

    def test_new_round_advances_the_period_and_scores_the_matching_estimate_exactly_once(self):
        svc, cell = live_service(n=0); svc.min_history = 10
        recs = self.recs(12); sync = SyncService(svc, FakeSource(recs), enabled=True, mode="30s"); sync.run_once()
        est = svc.next_prediction(); self.assertEqual(est["target_period"], "900013")
        nxt = RoundRecord("900013", 4, iso(T0 + 30)); cell[0] = T0 + 32
        sync.source = FakeSource(recs + [nxt])
        for _ in range(3): sync.run_once()                                                  # the same new round arrives on 3 polls
        outs = svc.repo.all_outcomes()
        self.assertEqual(len(outs), 1); self.assertEqual((outs[0]["target_period"], outs[0]["outcome"]["actual"]), ("900013", 4))
        self.assertEqual(svc.next_prediction()["target_period"], "900014"); self.assertEqual(svc.clock()["latest_confirmed"]["period_id"], "900013")

    def test_estimate_is_voided_not_scored_when_the_next_result_is_for_another_period(self):
        svc, cell = live_service(n=0); svc.min_history = 10
        svc.import_rows([{"number": i % 10, "period_id": str(900001 + i), "timestamp": iso(T0 - (11 - i) * 30)} for i in range(12)], source="live")
        est = svc.next_prediction(); self.assertEqual(est["target_period"], "900013")
        out = svc.add_result(6, "900020", iso(T0 + 240), source="live")                    # rounds 900013..900019 were never stored
        self.assertTrue(out["prediction_voided_target_mismatch"]); self.assertIsNone(out["resolved_prediction"])
        self.assertEqual(svc.repo.all_outcomes(), []); self.assertEqual(svc.voided_mismatch, 1)       # nothing was scored against the wrong round

    def test_a_pending_estimate_is_not_scored_by_a_result_with_a_matching_period_only_once(self):
        svc, cell = live_service(n=0); svc.min_history = 10
        svc.import_rows([{"number": i % 10, "period_id": str(900001 + i), "timestamp": iso(T0 - (11 - i) * 30)} for i in range(12)], source="live")
        svc.next_prediction(); svc.add_result(2, "900013", iso(T0 + 30), source="live")
        with self.assertRaises(Conflict): svc.add_result(2, "900013", iso(T0 + 30), source="live")
        self.assertEqual(len(svc.repo.all_outcomes()), 1)

    def test_api_errors_stale_data_and_gaps_give_accurate_status(self):
        svc, cell = live_service(); sync = SyncService(svc, FakeSource(self.recs(12)), enabled=True, mode="30s"); sync.run_once()
        sync.source = FakeSource(err=SourceError("rate limited (HTTP 429)", retry_after=90)); sync.run_once()
        st = sync.status(); self.assertEqual((st["state"], st["retry_after_seconds"]), ("error", 90)); self.assertGreaterEqual(sync.next_delay(), 89)
        self.assertEqual(st["latest_confirmed"]["period_id"], "900012")                    # the last good data is kept
        sync.source = FakeSource(self.recs(12)); sync._retry_until = 0; cell[0] = T0 + 30 * 9; sync.run_once()
        st = sync.status(); self.assertEqual(st["state"], "source_stale"); self.assertIn("published no new round", st["last_error"])
        self.assertEqual(svc.repo.last_result()["period_id"], "900012")                    # nothing fabricated to keep the dashboard moving
        sync.source = FakeSource([RoundRecord("900030", 1, iso(T0 + 600))]); sync.run_once()
        self.assertEqual(sync.status()["state"], "gap_blocked")

    def test_paging_backfills_a_gap_only_when_the_source_supports_it(self):
        svc, cell = live_service(); recs = self.recs(20)
        class Paged(FakeSource):
            supports_paging = True
            def fetch(self, max_pages=1): return list(recs if max_pages > 1 else recs[10:])
        SyncService(svc, FakeSource(recs[:6]), enabled=True, mode="30s").run_once()
        sync = SyncService(svc, Paged(), enabled=True, mode="30s", backfill_pages=3); sync.run_once()
        self.assertEqual(svc.repo.last_result()["period_id"], "900020"); self.assertEqual(len(svc.repo.results), 20)

    def test_diagnostics_never_contain_the_api_key(self):
        svc, _ = live_service(); sync = SyncService(svc, Wingo30ApiSource("super-secret-key"), enabled=True, mode="30s")
        self.assertNotIn("super-secret-key", json.dumps(sync.status())); self.assertEqual(sync.status()["provider"], "wingo30-api")


if __name__ == "__main__":
    unittest.main()
