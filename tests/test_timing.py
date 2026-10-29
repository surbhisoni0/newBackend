"""Timing + period-keyed workflow. Time is injected (never the real clock), no network, in-memory database."""
import os, sys, unittest
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.errors import Conflict
from app.services.predictor_service import PredictorService
from app.sync.service import SyncService
from app.sync.sources import RoundRecord, SourceError, parse_records
from app.timing import ClockConfig, countdown, next_period_id, parse_ts, verify_schedule

T0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc).timestamp()
CFG = ClockConfig()


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def rows_newest_first(n, newest_id=700100, newest_ts=T0, step=60.0, jitter=None):
    out = []
    for k in range(n):
        j = jitter[k] if jitter else 0.0
        out.append({"period_id": str(newest_id - k), "timestamp": iso(newest_ts - k * step + j)})
    return out


class TestScheduleVerification(unittest.TestCase):
    def test_no_timestamps_means_no_countdown(self):
        s = verify_schedule([{"period_id": str(700000 + i), "timestamp": None} for i in range(10)], CFG)
        self.assertFalse(s["verified"]); self.assertIn("timestamps", s["reason"])
        self.assertEqual(countdown(s, T0, CFG)["state"], "unverified")

    def test_no_period_ids_means_no_countdown(self):
        s = verify_schedule([{"period_id": None, "timestamp": iso(T0 - 60 * i)} for i in range(10)], CFG)
        self.assertFalse(s["verified"])

    def test_exact_spacing_is_verified(self):
        s = verify_schedule(rows_newest_first(10), CFG)
        self.assertTrue(s["verified"]); self.assertEqual(s["samples"], 10); self.assertAlmostEqual(s["spread_seconds"], 0, places=3)

    def test_small_jitter_is_tolerated(self):
        ok = verify_schedule(rows_newest_first(8, jitter=[0, 1.5, -2, 1, -1.5, 2, 0, 1]), CFG)
        self.assertTrue(ok["verified"]); self.assertEqual(ok["outliers"], 0)

    def test_one_late_timestamp_does_not_veto_the_schedule_but_is_reported(self):
        s = verify_schedule(rows_newest_first(20, jitter=[0, 0, 0, 0, 0, 0, 0, 25.0] + [0] * 12), CFG)     # a single late-recorded round
        self.assertTrue(s["verified"]); self.assertEqual(s["outliers"], 1)
        newest_is_late = verify_schedule(rows_newest_first(20, jitter=[25.0] + [0] * 19), CFG)               # even the NEWEST row being off
        self.assertTrue(newest_is_late["verified"]); self.assertAlmostEqual(newest_is_late["anchor_epoch"], T0, places=3)

    def test_many_disagreeing_timestamps_are_rejected(self):
        bad = verify_schedule(rows_newest_first(8, jitter=[0, 1, -2, 30, -1, 25, 0, 1]), CFG)                # 6/8 = 75% < 80%
        self.assertFalse(bad["verified"]); self.assertEqual(bad["outliers"], 2); self.assertIn("agree within", bad["reason"])
        scattered = verify_schedule(rows_newest_first(10, jitter=[0, 40, -50, 70, -30, 20, 60, -45, 33, -66]), CFG)
        self.assertFalse(scattered["verified"])                                                              # e.g. manual-entry arrival times

    def test_wrong_period_length_is_rejected(self):
        self.assertFalse(verify_schedule(rows_newest_first(10, step=90.0), CFG)["verified"])      # a 90s feed is not a 60s game

    def test_gap_in_period_ids_ends_the_verified_run(self):
        rows = rows_newest_first(10)
        rows[3]["period_id"] = str(int(rows[3]["period_id"]) - 5)                                  # ids jump: only 3 consecutive left
        s = verify_schedule(rows, CFG)
        self.assertEqual(s["samples"], 3); self.assertFalse(s["verified"])

    def test_naive_timestamps_use_the_configured_zone(self):
        naive_ist = "2026-10-23T15:30:00"                                                         # = 10:00:00 UTC
        self.assertEqual(parse_ts(naive_ist, "Asia/Kolkata"), T0)
        self.assertEqual(parse_ts("2026-10-23T10:00:00Z", "Asia/Kolkata"), T0)                    # explicit offset wins
        self.assertEqual(parse_ts(int(T0), "UTC"), T0); self.assertEqual(parse_ts(int(T0 * 1000), "UTC"), T0)
        self.assertIsNone(parse_ts("not a time")); self.assertIsNone(parse_ts(None))

    def test_bad_config_is_refused(self):
        for kw in ({"period_seconds": 1}, {"time_means": "middle"}, {"source_tz": "Mars/Base"}, {"min_samples": 1}):
            with self.assertRaises(ValueError): ClockConfig(**kw)

    def test_next_period_id_keeps_leading_zeros_and_rejects_text(self):
        self.assertEqual(next_period_id("0099"), "0100"); self.assertEqual(next_period_id("20261023100051234"), "20261023100051235")
        self.assertIsNone(next_period_id(None)); self.assertIsNone(next_period_id("abc"))


class TestCountdown(unittest.TestCase):
    def sched(self, cfg=CFG):
        return verify_schedule(rows_newest_first(10), cfg)

    def test_countdown_is_computed_from_timestamps_not_a_counter(self):
        s = self.sched()
        a, b = countdown(s, T0 + 10, CFG), countdown(s, T0 + 45, CFG)
        self.assertEqual((a["state"], round(a["seconds_remaining"], 3)), ("open", 50.0))
        self.assertEqual(round(b["seconds_remaining"], 3), 15.0)                                   # same input -> same output, any call order
        self.assertEqual(countdown(s, T0 + 10, CFG)["seconds_remaining"], a["seconds_remaining"])

    def test_zero_is_awaiting_confirmation_not_a_new_period(self):
        s = self.sched()
        c = countdown(s, T0 + 62, CFG)
        self.assertEqual(c["state"], "awaiting_result"); self.assertEqual(c["seconds_remaining"], 0.0)
        self.assertAlmostEqual(c["overdue_seconds"], 2.0, places=3); self.assertEqual(c["rounds_ended_since_confirmed"], 1)

    def test_never_negative_and_becomes_late_after_grace(self):
        s = self.sched()
        c = countdown(s, T0 + 60 + CFG.late_grace_seconds + 5, CFG)
        self.assertEqual(c["state"], "late"); self.assertGreaterEqual(c["seconds_remaining"], 0)
        far = countdown(s, T0 + 60 * 7 + 3, CFG)
        self.assertEqual(far["state"], "late"); self.assertEqual(far["rounds_ended_since_confirmed"], 7); self.assertEqual(far["seconds_remaining"], 0.0)

    def test_timestamp_in_the_future_disables_countdown(self):
        c = countdown(self.sched(), T0 - 3600, CFG)                                                # e.g. wrong timezone assumption
        self.assertEqual(c["state"], "unverified"); self.assertIn("future", c["reason"])

    def test_start_semantics_shift_by_one_period(self):
        cfg = ClockConfig(time_means="start"); s = verify_schedule(rows_newest_first(10), cfg)     # timestamp = START of the newest round
        c = countdown(s, T0 + 70, cfg)                                                              # it ended at T0+60; next ends at T0+120
        self.assertEqual((c["state"], round(c["seconds_remaining"], 3)), ("open", 50.0))
        self.assertEqual(countdown(s, T0 + 125, cfg)["state"], "awaiting_result")
        # a confirmed result for a round that (by this reading) has not ended yet is impossible -> refuse to guess
        self.assertEqual(countdown(s, T0 + 10, cfg)["state"], "unverified")


def make(with_clock=True, n=130, now=None, source=None):
    """n confirmed rounds, ids 700001.., one per minute, last one ends at T0. Returns (service, now_cell)."""
    cell = [T0 + 5 if now is None else now]
    svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), 100, 300, clock_cfg=CFG, now_fn=lambda: cell[0])
    svc.repo.init()
    base = 700000
    svc.import_rows([{"number": (i * 7 + i // 3) % 10, "period_id": str(base + i + 1),
                      "timestamp": iso(T0 - (n - 1 - i) * 60) if with_clock else None} for i in range(n)])
    return svc, cell


class TestPeriodKeyedWorkflow(unittest.TestCase):
    def test_startup_with_no_data(self):
        svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), 100, 300); svc.repo.init()
        c = svc.clock(); self.assertEqual(c["countdown"]["state"], "no_data"); self.assertIsNone(c["next_round"])
        self.assertFalse(svc.next_prediction()["available"])

    def test_estimate_is_bound_to_the_exact_next_period(self):
        svc, _ = make()
        p = svc.next_prediction()
        self.assertEqual(p["target_period"], "700131"); self.assertEqual(p["target_label"], "700131"); self.assertEqual(p["target_id_kind"], "period_id")
        self.assertEqual(p["timing"]["next_round"]["period_id"], "700131")                         # clock and estimate agree
        self.assertEqual(p["estimate_state"], "open"); self.assertTrue(p["timing"]["schedule"]["verified"])
        self.assertAlmostEqual(p["timing"]["countdown"]["seconds_remaining"], 55.0, places=2)

    def test_without_period_ids_the_label_is_honestly_a_sequence_number(self):
        svc, _ = make(with_clock=False)
        svc2 = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), 100, 300); svc2.repo.init()
        svc2.import_rows([{"number": (i * 7 + i // 3) % 10, "period_id": None, "timestamp": None} for i in range(130)])
        p = svc2.next_prediction()
        self.assertIsNone(p["target_period"]); self.assertEqual(p["target_label"], "#131"); self.assertEqual(p["target_id_kind"], "sequence")
        self.assertEqual(p["estimate_state"], "timing_unverified"); self.assertEqual(p["timing"]["countdown"]["state"], "unverified")

    def test_repeated_requests_and_refresh_never_regenerate(self):
        svc, cell = make()
        first = svc.next_prediction()
        for dt in (1, 2, 20, 30):                                                                   # "page refresh" at various moments in the round
            cell[0] = T0 + dt; again = svc.next_prediction()
            self.assertEqual((again["id"], again["created_at"], again["size"], again["top1"]), (first["id"], first["created_at"], first["size"], first["top1"]))
        self.assertEqual(sum(1 for p in svc.repo.preds if p["target_seq"] == 131), 1)

    def test_transition_resolves_old_estimate_and_creates_one_new_for_next_period(self):
        svc, cell = make()
        before = svc.next_prediction()
        cell[0] = T0 + 61
        out = svc.add_result(4, "700131", iso(T0 + 60))
        self.assertEqual(out["resolved_prediction"]["target_period"], "700131"); self.assertEqual(out["resolved_prediction"]["outcome"]["actual"], 4)
        self.assertEqual(out["next_prediction"]["target_period"], "700132")
        self.assertNotEqual(out["next_prediction"]["id"], before["id"])
        self.assertEqual(svc.clock()["countdown"]["state"], "open")                                 # a confirmed result, not the timer, advances the round
        self.assertEqual(sum(1 for p in svc.repo.preds if p["target_seq"] == 132 and p["status"] != "void"), 1)

    def test_duplicate_and_out_of_order_results_are_rejected(self):
        svc, _ = make()
        svc.add_result(4, "700131", iso(T0 + 60))
        with self.assertRaises(Conflict): svc.add_result(4, "700131", iso(T0 + 60))               # duplicate period
        with self.assertRaises(Conflict): svc.add_result(1, "700100", iso(T0 - 600))              # older than latest
        self.assertEqual(svc.repo.last_result()["seq"], 131)

    def test_late_result_estimate_is_kept_flagged_then_scored_not_replaced(self):
        svc, cell = make()
        est = svc.next_prediction(); cell[0] = T0 + 50
        self.assertEqual(svc.next_prediction()["estimate_state"], "open")
        cell[0] = T0 + 61                                                                           # countdown reached zero, no result yet
        p = svc.next_prediction()
        self.assertEqual(p["estimate_state"], "round_ended"); self.assertEqual(p["timing"]["countdown"]["state"], "awaiting_result")
        self.assertEqual((p["id"], p["created_at"], p["size"]), (est["id"], est["created_at"], est["size"]))   # unchanged, not re-rolled
        cell[0] = T0 + 400
        p = svc.next_prediction()
        self.assertEqual(p["timing"]["countdown"]["state"], "late")
        # Several cycles have passed with no verified result: the card now shows a recalculated, UNSTORED projection for the round that is
        # running. The stored pick is untouched and still waits to be scored against ITS OWN round (asserted here and below).
        self.assertEqual((p["kind"], p["estimate_state"], p["id"]), ("projection", "projection", None))
        stored = svc.repo.get_prediction_by_target(131)
        self.assertEqual((stored["id"], stored["status"], stored["size"]), (est["id"], "pending", est["size"]))
        self.assertEqual(sum(1 for r in svc.repo.preds if r["status"] != "void"), 1)                  # the projection was never written to the database
        out = svc.add_result(7, "700131", iso(T0 + 60))                                             # the late result finally arrives
        self.assertEqual(out["resolved_prediction"]["id"], est["id"]); self.assertEqual(out["resolved_prediction"]["outcome"]["actual"], 7)
        self.assertEqual(out["resolved_prediction"]["size"], est["size"])                           # the original pick was scored, not swapped
        self.assertEqual(svc.clock()["countdown"]["state"], "late")                                 # and the clock says more rounds are still unconfirmed

    def test_estimates_and_verified_results_are_stored_separately(self):
        svc, _ = make(); est = svc.next_prediction()
        self.assertTrue(all("size" not in r and "top1" not in r for r in svc.repo.results))         # results rows carry no estimate fields
        svc.add_result(2, "700131", iso(T0 + 60))
        h = [r for r in svc.history(50, 0)["predictions"] if r["round"] == "700131"][0]
        self.assertEqual((h["size"], h["actual"]), (est["size"], 2)); self.assertEqual(svc.repo.results[-1]["number"], 2)
        self.assertEqual(svc.repo.get_prediction_by_target(131)["top1"], est["top1"])               # estimate row unchanged after scoring


class TestSyncRobustness(unittest.TestCase):
    class Src:
        configured, name, last_invalid = True, "fake", 0
        def __init__(self, recs=None, err=None): self.recs, self.err = recs or [], err
        def fetch(self):
            if self.err: raise SourceError(self.err)
            return list(self.recs)

    def test_provider_outage_keeps_last_data_and_invents_nothing(self):
        svc, cell = make(); est = svc.next_prediction(); n_pred = len(svc.repo.preds)
        sync = SyncService(svc, self.Src(err="network error: down"), enabled=True)
        sync.run_once(); sync.run_once()
        self.assertEqual(sync.s["state"], "error"); self.assertEqual(sync.s["consecutive_failures"], 2)
        self.assertEqual(len(svc.repo.preds), n_pred); self.assertEqual(svc.repo.last_result()["seq"], 130)   # no fake round, no fake estimate
        self.assertEqual(svc.next_prediction()["id"], est["id"])
        self.assertGreater(sync.next_delay(), sync.interval)                                         # backs off instead of hammering

    def test_same_batch_twice_creates_no_duplicates(self):
        svc, cell = make()
        recs = [RoundRecord("700131", 3, iso(T0 + 60)), RoundRecord("700132", 8, iso(T0 + 120))]
        sync = SyncService(svc, self.Src(recs), enabled=True)
        cell[0] = T0 + 125
        sync.run_once(); n_results, n_preds = len(svc.repo.results), len(svc.repo.preds)
        sync.run_once(); sync.run_once()
        self.assertEqual((len(svc.repo.results), len(svc.repo.preds)), (n_results, n_preds)); self.assertEqual(sync.s["ingested_total"], 2)
        self.assertEqual(svc.repo.last_result()["period_id"], "700132"); self.assertEqual(svc.next_prediction()["target_period"], "700133")

    def test_invalid_items_are_dropped_not_repaired(self):
        good, bad = parse_records([{"p": "700131", "n": 3, "t": 1761213600}, {"p": "x", "n": 1, "t": 1}, {"p": "700133", "n": 11, "t": 2}], "", "p", "n", "t")
        self.assertEqual([(r.period_id, r.number) for r in good], [("700131", 3)]); self.assertEqual(bad, 2)

    def test_gap_blocks_ingestion_and_is_reported(self):
        svc, cell = make()
        sync = SyncService(svc, self.Src([RoundRecord("700135", 1, iso(T0 + 300))]), enabled=True)
        sync.run_once()
        self.assertEqual(sync.s["state"], "gap_blocked"); self.assertEqual(svc.repo.last_result()["seq"], 130)

    def test_epoch_timestamps_from_a_json_source_are_accepted_end_to_end(self):
        svc, cell = make(); cell[0] = T0 + 65
        recs, _ = parse_records([{"p": "700131", "n": 6, "t": int(T0 + 60)}], "", "p", "n", "t")
        SyncService(svc, self.Src(recs), enabled=True).run_once()
        self.assertEqual(svc.repo.last_result()["period_id"], "700131")
        self.assertTrue(svc.clock()["schedule"]["verified"])

    def test_fast_polling_only_while_a_result_is_overdue(self):
        svc, cell = make(); sync = SyncService(svc, self.Src(), enabled=True, interval=30, fast_interval=5)
        cell[0] = T0 + 10; self.assertEqual(sync.next_delay(), 30)
        cell[0] = T0 + 62; self.assertEqual(sync.next_delay(), 5)
        cell[0] = T0 + 600; self.assertEqual(sync.next_delay(), 30)                                  # long overdue: do not hammer the provider


if __name__ == "__main__":
    unittest.main()
