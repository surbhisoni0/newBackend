"""Timing verification: timestamps, schedule proof, feed open-round validation, sync failures, countdown rollover, period ids.

No test here talks to the network. The feed is a scripted stand-in that follows the contract documented in
app/sync/sources.py (state 0 = open / 1 = settled, startTime/endTime as IST wall time without offset, `Date` header for the clock).
Nothing in these tests is real feed data.
"""
import os, sys, time, types, unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.database.memory_repo import InMemoryRepository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService
from app.sync.service import SyncService
from app.sync.sources import IST, SourceError, WinGoFeedSource
from app.timing import (ClockConfig, advance_period_id, next_period_id, parse_ts, period_id_date, timestamp_matches_id,
                        validate_open_round, verification, verify_schedule)

UTC = timezone.utc
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def utc(*a):
    return datetime(*a, tzinfo=UTC).timestamp()


def pid(day, code, n):
    return f"{day}{code}{n:04d}"


# ------------------------------------------------------------------------------------------------------------- timestamps
class TestParseTimestamp(unittest.TestCase):
    def test_valid_forms(self):
        t = utc(2026, 10, 29, 9, 1, 0)
        self.assertEqual(parse_ts("2026-10-29T09:01:00Z"), t)
        self.assertEqual(parse_ts("2026-10-29T14:31:00+05:30"), t)
        self.assertEqual(parse_ts("2026-10-29 14:31:00", "Asia/Kolkata"), t)         # naive -> the zone that is passed in
        self.assertEqual(parse_ts("2026-10-29T09:01:00", "UTC"), t)
        self.assertEqual(parse_ts(str(int(t))), t)                                    # epoch seconds
        self.assertEqual(parse_ts(str(int(t * 1000))), t)                             # epoch milliseconds

    def test_invalid_values_are_none_never_a_guess(self):
        for bad in (None, "", "   ", "garbage", "2026-13-45T00:00:00Z", "nan", "inf", "12:34", "10/29/2026 1:05 PM", "0", "-5"):
            self.assertIsNone(parse_ts(bad), bad)

    def test_out_of_range_epochs_and_years_are_rejected(self):
        self.assertIsNone(parse_ts("946684799"))                                     # 1999-12-31
        self.assertIsNone(parse_ts("99999999999999"))                                # absurd milliseconds
        self.assertIsNone(parse_ts("1999-06-01T00:00:00Z"))
        self.assertIsNone(parse_ts("2101-01-01T00:00:00Z"))
        self.assertIsNone(parse_ts("0001-01-01T00:00:00Z"))                          # astimezone would overflow
        self.assertIsNotNone(parse_ts("2000-01-01T00:00:00Z"))

    def test_period_id_date_and_consistency(self):
        self.assertEqual(str(period_id_date("20261006100011183")), "2026-10-06")
        for odd in ("700131", "2026100610001118", "20261340100011183", "20261006999991183", None, "abc"):
            self.assertIsNone(period_id_date(odd), odd)
        ts = utc(2026, 10, 6, 12)
        self.assertTrue(timestamp_matches_id("20261006100011183", ts))
        self.assertTrue(timestamp_matches_id("20261006100011183", utc(2026, 10, 7, 1)))          # IST/UTC day-boundary margin
        self.assertFalse(timestamp_matches_id("20261006100011183", utc(2026, 10, 28, 12)))       # 22 days off: cannot be this round
        self.assertTrue(timestamp_matches_id("700131", utc(2026, 10, 28)))                       # unknown id shape: not judged


# ------------------------------------------------------------------------------------------------------------- stored schedule
def stored_rows(n=8, end=utc(2026, 10, 29, 9, 0), day="20261029", code="10001", first=500, step=60, tz_suffix="Z"):
    """newest first, one round per `step` seconds, timestamps = draw (end) times."""
    return [{"period_id": pid(day, code, first + n - 1 - i), "number": i % 10,
             "timestamp": datetime.fromtimestamp(end - i * step, UTC).strftime("%Y-%m-%dT%H:%M:%S") + tz_suffix} for i in range(n)]


class TestStoredSchedule(unittest.TestCase):
    cfg = ClockConfig()

    def test_verified_when_timestamps_follow_the_schedule(self):
        v = verify_schedule(stored_rows(), self.cfg)
        self.assertTrue(v["verified"]); self.assertIsNone(v["code"]); self.assertEqual(v["samples"], 8)

    def test_missing_timestamps(self):
        rows = [{"period_id": r["period_id"], "number": 1, "timestamp": None} for r in stored_rows()]
        v = verify_schedule(rows, self.cfg)
        self.assertFalse(v["verified"]); self.assertEqual(v["code"], "no_timestamps"); self.assertEqual(v["samples"], 0)

    def test_unparseable_timestamp_ends_the_run(self):
        rows = stored_rows(); rows[0]["timestamp"] = "garbage"
        self.assertEqual(verify_schedule(rows, self.cfg)["code"], "no_timestamps")

    def test_missing_period_ids(self):
        rows = [{**r, "period_id": None} for r in stored_rows()]
        self.assertEqual(verify_schedule(rows, self.cfg)["code"], "no_period_ids")

    def test_too_few_rows(self):
        v = verify_schedule(stored_rows(3), self.cfg)
        self.assertFalse(v["verified"]); self.assertEqual(v["code"], "insufficient_rows")

    def test_gap_in_period_ids(self):
        rows = stored_rows(8); rows[3]["period_id"] = str(int(rows[3]["period_id"]) - 1)           # newest run is only 3 long
        v = verify_schedule(rows, self.cfg)
        self.assertFalse(v["verified"]); self.assertEqual(v["code"], "period_gap")

    def test_timestamps_that_contradict_their_ids_are_not_evidence(self):
        """This is exactly what the old seed CSV looked like: ids from 2026-10-06, 'times' invented on 2026-10-28."""
        rows = stored_rows(8, end=utc(2026, 10, 28, 12, 12), day="20261006", first=1176)
        v = verify_schedule(rows, self.cfg)
        self.assertFalse(v["verified"]); self.assertEqual(v["code"], "timestamps_contradict_ids")
        self.assertIn("2026-10-06", v["reason"])

    def test_timestamps_not_one_period_apart(self):
        v = verify_schedule(stored_rows(8, step=45), self.cfg)
        self.assertFalse(v["verified"]); self.assertEqual(v["code"], "timestamps_off_schedule")

    def test_wrong_timezone_is_reported_not_trusted(self):
        """IST wall time read as UTC lands 5.5 h in the future: the countdown must be refused with an actionable code."""
        from app.timing import countdown
        rows = stored_rows(8, tz_suffix="")                                                           # naive strings
        ist_naive = [{**r, "timestamp": datetime.fromtimestamp(parse_ts(r["timestamp"]), IST).strftime("%Y-%m-%dT%H:%M:%S")} for r in rows]
        now = utc(2026, 10, 29, 9, 0, 30)
        wrong = countdown(verify_schedule(ist_naive, ClockConfig(source_tz="UTC")), now, ClockConfig(source_tz="UTC"))
        self.assertEqual((wrong["state"], wrong["code"]), ("unverified", "future_timestamp"))
        right_cfg = ClockConfig(source_tz="Asia/Kolkata")
        right = countdown(verify_schedule(ist_naive, right_cfg), now, right_cfg)
        self.assertEqual(right["state"], "open")                                                      # newest round ended 09:00:00; 30 s into the next one
        self.assertAlmostEqual(right["seconds_remaining"], 30.0, places=1)


# ------------------------------------------------------------------------------------------------------------- feed open round
def good_round(P=60, fetched=1000.0, offset=0.0, end=1030.0):
    return {"period_id": "20261029100010541", "start": end - P, "end": end, "interval_seconds": P, "observed_feed_time": fetched + offset}


class TestValidateOpenRound(unittest.TestCase):
    def check(self, op, **kw):
        a = dict(fetched_at=1000.0, clock_offset=0.0, period_seconds=60, now=1005.0); a.update(kw)
        return validate_open_round(op, **a)

    def test_valid(self):
        r = self.check(good_round()); self.assertTrue(r["ok"]); self.assertIsNone(r["code"])

    def test_no_open_round(self):
        for op in (None, {}):
            self.assertEqual(self.check(op)["code"], "feed_no_open_round")

    def test_unreadable_fields(self):
        for patch in ({"period_id": None}, {"period_id": "abc"}, {"start": None}, {"end": float("nan")}, {"end": "x"}):
            self.assertEqual(self.check({**good_round(), **patch})["code"], "feed_open_round_invalid", patch)

    def test_interval_must_match_the_mode(self):
        self.assertEqual(self.check({**good_round(), "end": 1090.0})["code"], "feed_interval_mismatch")          # 120 s round in a 60 s mode
        self.assertEqual(self.check({**good_round(), "end": 940.0})["code"], "feed_interval_mismatch")           # end before start
        self.assertEqual(self.check({**good_round(), "interval_seconds": 180})["code"], "feed_interval_mismatch")
        self.assertTrue(self.check({**good_round(), "interval_seconds": None})["ok"])                             # absent is fine; end-start is the proof

    def test_stale_read(self):
        self.assertTrue(self.check(good_round(), now=1000 + 135)["ok"])
        r = self.check(good_round(), now=1000 + 136); self.assertEqual(r["code"], "feed_stale")
        self.assertEqual(self.check(good_round(), fetched_at=None)["code"], "feed_not_fetched_yet")

    def test_read_from_the_future(self):
        self.assertEqual(self.check(good_round(), now=900.0)["code"], "feed_open_round_invalid")

    def test_clock_must_be_measured(self):
        self.assertEqual(self.check(good_round(), clock_offset=None)["code"], "feed_clock_unmeasured")
        self.assertEqual(self.check({**good_round(), "observed_feed_time": None})["code"], "feed_clock_unmeasured")

    def test_feed_clock_must_lie_inside_the_round_it_calls_open(self):
        """Catches a wrong timezone (5.5 h out), a frozen feed and a cached response - none of which may count as verified."""
        wrong_tz = {**good_round(), "start": 1030.0 - 60 + 19800, "end": 1030.0 + 19800}
        self.assertEqual(self.check(wrong_tz)["code"], "feed_round_not_current")
        self.assertEqual(self.check({**good_round(), "observed_feed_time": 1500.0})["code"], "feed_round_not_current")
        self.assertTrue(self.check({**good_round(), "observed_feed_time": 1031.9})["ok"])                         # inside the 2 s tolerance
        self.assertEqual(self.check({**good_round(), "observed_feed_time": 1033.0})["code"], "feed_round_not_current")

    def test_verification_object_rejects_unknown_status(self):
        with self.assertRaises(ValueError): verification("probably_fine")


# ------------------------------------------------------------------------------------------------------------- scripted feed
class SimClock:
    def __init__(self, t): self.t = t
    def __call__(self): return self.t


class ScriptedFeed(WinGoFeedSource):
    """WinGoFeedSource with the network replaced by a script. Everything after `_post` (parsing, open-round logic) is the real code."""
    def __init__(self, clock, P=60, code="10001", **kw):
        super().__init__("https://feed.invalid/api", {60: 1, 30: 30, 180: 10, 300: 3}[P], P, backfill_rows=0, **kw)
        self.clock, self.P, self.code = clock, P, code
        self.error, self.omit_open, self.offset, self.header_ok, self.ist_shift = None, False, 0.0, True, 0.0

    def _check_robots(self): pass

    def items(self):
        t, P = self.clock.t + self.offset, self.P
        end = (int(t // P) + 1) * P
        day0 = datetime.fromtimestamp(end - P, UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        n = int((end - day0.timestamp()) // P)                                       # round of the UTC day the open round belongs to
        day = day0.strftime("%Y%m%d")
        wall = lambda e: datetime.fromtimestamp(e + self.ist_shift, IST).strftime("%Y-%m-%d %H:%M:%S")
        open_ = {"issueNumber": pid(day, self.code, n), "number": "", "state": 0, "startTime": wall(end - P), "endTime": wall(end), "intervalM": P / 60}
        done = [{"issueNumber": pid(day, self.code, n - i), "number": str((n - i) % 10), "state": 1,
                 "startTime": wall(end - (i + 1) * P), "endTime": wall(end - i * P), "intervalM": P / 60} for i in range(1, 9) if n - i >= 1]
        return ([] if self.omit_open else [open_]) + done

    def _post(self, day, page, size):
        if self.error: raise self.error
        self.fetch_mid = self.clock.t
        self.clock_offset = self.offset if self.header_ok else None
        return {"data": self.items()}


def make_service(clock, feed, source_tz="UTC", P=60):
    svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), clock_cfg=ClockConfig(period_seconds=P, source_tz=source_tz),
                           now_fn=clock)
    svc.live_source = feed
    sync = SyncService(svc, feed, True, 10, allow_gaps=True, fast_interval=3)
    svc.sync_service = sync
    return svc, sync


def run_sync(clock, sync):
    with mock.patch("time.time", clock):
        return sync.run_once()


class TestFeedSource(unittest.TestCase):
    def setUp(self):
        self.clock = SimClock(utc(2026, 10, 29, 9, 0, 30))

    def fetch(self, feed):
        with mock.patch("time.time", self.clock):
            return feed.fetch()

    def test_settled_rows_carry_an_explicit_ist_offset_and_open_round_is_never_stored(self):
        feed = ScriptedFeed(self.clock); recs = self.fetch(feed)
        self.assertEqual(len(recs), 8)
        self.assertTrue(all(r.number is not None and r.period_id.isdigit() for r in recs))
        self.assertTrue(all(r.timestamp.endswith("+05:30") for r in recs))
        self.assertEqual(parse_ts(recs[0].timestamp), utc(2026, 10, 29, 9, 0))                    # newest settled round ended 09:00:00Z
        self.assertNotIn(feed.open_round["period_id"], [r.period_id for r in recs])

    def test_unparseable_end_time_is_dropped_not_repaired(self):
        feed = ScriptedFeed(self.clock)
        items = feed.items(); items[1]["endTime"] = "yesterday-ish"
        recs = feed._settled(items)
        self.assertEqual(len(recs), 8); self.assertIsNone(recs[0].timestamp)                       # the result is kept, the time is honestly unknown
        self.assertEqual(feed.invalid_timestamps, 1)
        self.assertTrue(all(r.timestamp for r in recs[1:]))

    def test_open_round_fields(self):
        feed = ScriptedFeed(self.clock); self.fetch(feed); op = feed.open_round
        self.assertEqual((op["period_id"], op["end"] - op["start"], op["interval_seconds"]), ("20261029100010541", 60.0, 60))
        self.assertEqual(op["end"], utc(2026, 10, 29, 9, 1))
        self.assertEqual(op["observed_feed_time"], self.clock.t)

    def test_open_round_is_forgotten_when_the_feed_stops_reporting_one(self):
        feed = ScriptedFeed(self.clock); self.fetch(feed); self.assertIsNotNone(feed.open_round)
        self.clock.t += 120; feed.omit_open = True; self.fetch(feed)
        self.assertIsNone(feed.open_round)                                                          # previously it was kept and rolled forward

    def test_several_open_rounds_the_earliest_ending_wins(self):
        feed = ScriptedFeed(self.clock); items = feed.items()
        future = {**items[0], "issueNumber": "20261029100010542", "startTime": items[0]["endTime"], "endTime": "2026-10-29 14:32:00"}
        feed._remember_open([future] + items); self.assertEqual(feed.open_round["period_id"], "20261029100010541")

    def test_malformed_open_round_is_ignored(self):
        feed = ScriptedFeed(self.clock)
        feed._remember_open([{"state": 0, "issueNumber": "abc", "startTime": "2026-10-29 14:30:00", "endTime": "2026-10-29 14:31:00"},
                             {"state": 0, "issueNumber": "20261029100010541", "startTime": "?", "endTime": "2026-10-29 14:31:00"}])
        self.assertIsNone(feed.open_round)

    def test_error_codes(self):
        class R:                                                                                  # minimal httpx.Response stand-in
            def __init__(s, status=200, headers=None, body=None, bad_json=False): s.status_code, s.headers, s._b, s._bad = status, headers or {}, body, bad_json
            def json(s):
                if s._bad: raise ValueError("no json")
                return s._b
        def run(resp=None, exc=None):
            fake = types.SimpleNamespace(post=lambda *a, **k: (_ for _ in ()).throw(exc) if exc else resp)
            feed = ScriptedFeed(self.clock)
            with mock.patch.dict(sys.modules, {"httpx": fake}), mock.patch("time.time", self.clock):
                try: WinGoFeedSource._post(feed, "2026-10-29", 1, 50); return feed, None
                except SourceError as e: return feed, e.code
        self.assertEqual(run(R(403))[1], "not_authorised"); self.assertEqual(run(R(401))[1], "not_authorised")
        self.assertEqual(run(R(429))[1], "rate_limited"); self.assertEqual(run(R(503))[1], "http_error")
        self.assertEqual(run(R(200, bad_json=True))[1], "bad_response")
        self.assertEqual(run(exc=OSError("connection refused"))[1], "network_error")

    def test_date_header_gives_the_clock_offset_and_bad_or_missing_header_gives_none(self):
        class R:
            status_code = 200
            def __init__(s, h): s.headers = h
            def json(s): return {"data": []}
        def offset(headers):
            feed = ScriptedFeed(self.clock); fake = types.SimpleNamespace(post=lambda *a, **k: R(headers))
            with mock.patch.dict(sys.modules, {"httpx": fake}), mock.patch("time.time", self.clock):
                WinGoFeedSource._post(feed, "2026-10-29", 1, 50)
            return feed.clock_offset, feed
        off, feed = offset({"date": "Thu, 29 Oct 2026 09:00:32 GMT"}); self.assertEqual(off, 2.0)     # feed clock 2 s ahead of ours
        self.assertEqual(feed.fetch_mid, self.clock.t)
        self.assertEqual(offset({"x-upstream-date": "Thu, 29 Oct 2026 09:00:29 GMT", "date": "Thu, 29 Oct 2026 09:00:32 GMT"})[0], -1.0)  # upstream header wins
        self.assertIsNone(offset({})[0]); self.assertIsNone(offset({"date": "not a date"})[0])


# ------------------------------------------------------------------------------------------------------------- sync + clock together
class TestClockWithFeed(unittest.TestCase):
    def setUp(self):
        self.clock = SimClock(utc(2026, 10, 29, 9, 0, 30)); self.feed = ScriptedFeed(self.clock)
        self.svc, self.sync = make_service(self.clock, self.feed)

    def cd(self):
        return self.svc.clock(self.clock.t)["countdown"]

    def test_healthy_feed_is_verified_with_the_feeds_own_round(self):
        self.assertEqual(run_sync(self.clock, self.sync)["state"], "ok")
        c = self.cd()
        self.assertEqual((c["state"], c["verification"]["status"], c["verification"]["method"]), ("open", "verified", "feed_open_round"))
        self.assertEqual((c["current_period"], c["next_period"]), ("20261029100010541", "20261029100010542"))
        self.assertAlmostEqual(c["seconds_remaining"], 30.0, places=1)
        self.assertEqual(c["round_ends_at"], "2026-10-29T09:01:00Z")

    def test_stored_feed_timestamps_do_not_depend_on_SOURCE_TIMEZONE(self):
        """Regression: IST wall times were stored without an offset and read with SOURCE_TIMEZONE (default UTC) -> 'in the future'."""
        run_sync(self.clock, self.sync)
        self.assertTrue(self.svc.repo.last_result()["timestamp"].endswith("+05:30"))
        self.feed.error = SourceError("HTTP 503", "http_error"); self.clock.t += 200            # live feed unusable -> stored-timestamp path
        for tz in ("UTC", "Asia/Kolkata", "America/New_York"):
            self.svc.clock_cfg = ClockConfig(source_tz=tz)
            c = self.svc.clock(self.clock.t)
            self.assertNotEqual(c["countdown"].get("code"), "future_timestamp", tz)
            self.assertTrue(c["schedule"]["verified"], tz)

    def test_sync_failure_is_reported_with_its_cause_and_never_shown_verified(self):
        run_sync(self.clock, self.sync)
        self.feed.error = SourceError("not authorised (HTTP 403)", "not_authorised")
        self.clock.t += 400                                                                      # long enough that nothing fresh remains
        st = run_sync(self.clock, self.sync)
        self.assertEqual((st["state"], st["last_error_code"], st["consecutive_failures"]), ("error", "not_authorised", 1))
        self.svc.repo = InMemoryRepository()                                                      # no stored fallback either
        c = self.cd()
        self.assertEqual(c["state"], "no_data")
        self.assertEqual(c["verification"]["status"], "no_data"); self.assertEqual(c["verification"]["code"], "not_authorised")
        self.assertIn("403", c["verification"]["reason"])

    def test_unverified_explains_both_sources_when_both_fail(self):
        run_sync(self.clock, self.sync)
        self.feed.error = SourceError("network error: boom", "network_error"); self.clock.t += 400; run_sync(self.clock, self.sync)
        rows = self.svc.repo.list_results(10, 0)[0]
        self.svc.repo.results[:] = [{**r, "timestamp": None} for r in sorted(rows, key=lambda r: r["seq"])] + []   # stored times lost
        c = self.cd()
        self.assertEqual(c["state"], "unverified"); self.assertIsNone(c["seconds_remaining"]); self.assertIsNone(c["round_ends_at"])
        v = c["verification"]
        self.assertEqual((v["status"], v["code"]), ("unverified", "network_error"))
        self.assertEqual({x["source"] for x in v["causes"]}, {"feed", "stored_timestamps"})
        self.assertIn("boom", c["reason"]); self.assertIn("timestamp", c["reason"])

    def test_recovers_by_itself_when_the_feed_comes_back(self):
        self.feed.error = SourceError("HTTP 503", "http_error")
        self.assertEqual(run_sync(self.clock, self.sync)["state"], "error")
        self.assertEqual(self.cd()["verification"]["status"], "no_data")
        self.feed.error = None; self.clock.t += 10
        self.assertEqual(run_sync(self.clock, self.sync)["state"], "ok")
        self.assertEqual(self.cd()["verification"]["status"], "verified")

    def test_a_single_failed_poll_does_not_flip_a_fresh_verified_countdown(self):
        run_sync(self.clock, self.sync)
        self.feed.error = SourceError("HTTP 502", "http_error"); self.clock.t += 10; run_sync(self.clock, self.sync)
        self.assertEqual(self.cd()["verification"]["status"], "verified")                       # read is 10 s old: still inside the bound
        self.clock.t += 200
        self.assertNotEqual(self.cd()["verification"]["status"], "verified")                    # 210 s old: no longer vouched for

    def test_feed_with_no_open_round_is_not_verified(self):
        self.feed.omit_open = True; run_sync(self.clock, self.sync)
        v = self.cd()["verification"]
        self.assertNotEqual(v["status"], "verified"); self.assertEqual(v["code"], "feed_no_open_round")

    def test_missing_date_header_cannot_be_verified(self):
        self.feed.header_ok = False; run_sync(self.clock, self.sync)
        self.assertEqual(self.cd()["verification"]["code"], "feed_clock_unmeasured")

    def test_wrong_timezone_in_the_feed_is_caught_by_the_feeds_own_clock(self):
        self.feed.ist_shift = 5.5 * 3600                                                         # feed times 5.5 h off what its clock says
        run_sync(self.clock, self.sync)
        v = self.cd()["verification"]
        self.assertNotEqual(v["status"], "verified"); self.assertEqual(v["code"], "feed_round_not_current")

    def test_feed_clock_offset_is_applied(self):
        self.feed.offset = 2.0; run_sync(self.clock, self.sync)                                  # feed clock runs 2 s ahead of ours
        c = self.cd()
        self.assertEqual(c["verification"]["status"], "verified")
        self.assertAlmostEqual(c["seconds_remaining"], 28.0, places=1)                            # 30 s to the feed's boundary, minus 2 s
        self.assertEqual(c["round_ends_at"], "2026-10-29T09:00:58Z")                              # expressed on OUR clock

    def test_wrong_mode_interval_is_not_verified(self):
        feed30 = ScriptedFeed(self.clock, P=30)                                                  # a 30 s feed read by a 60 s mode
        svc, sync = make_service(self.clock, feed30, P=60)
        run_sync(self.clock, sync); c = svc.clock(self.clock.t)["countdown"]
        self.assertNotEqual(c["verification"]["status"], "verified"); self.assertEqual(c["verification"]["code"], "feed_interval_mismatch")


# ------------------------------------------------------------------------------------------------------------- rollover
class TestCountdownRollover(unittest.TestCase):
    def live(self, P, code, start, ahead):
        """Feed read at `start`; the clock is then advanced by `ahead` seconds WITHOUT another read -> boundary extrapolation."""
        clock = SimClock(start); feed = ScriptedFeed(clock, P=P, code=code)
        svc, sync = make_service(clock, feed, P=P); run_sync(clock, sync)
        clock.t = start + ahead
        return svc.clock(clock.t)["countdown"], feed

    def test_period_changes_exactly_at_the_boundary(self):
        start = utc(2026, 10, 29, 9, 0, 55)                                                      # 5 s before 09:01:00
        for ahead, cur, left in ((4.0, "20261029100010541", 1.0), (5.0, "20261029100010542", 60.0), (5.01, "20261029100010542", 59.99),
                                 (35.0, "20261029100010542", 30.0), (64.99, "20261029100010542", 0.01), (65.0, "20261029100010543", 60.0)):
            c, _ = self.live(60, "10001", start, ahead)
            self.assertEqual(c["current_period"], cur, ahead); self.assertAlmostEqual(c["seconds_remaining"], left, places=2, msg=ahead)
            self.assertEqual(c["next_period"], next_period_id(cur))

    def test_countdown_is_continuous_and_never_negative(self):
        start, prev, wraps = utc(2026, 10, 29, 9, 0, 30), None, 0
        clock = SimClock(start); feed = ScriptedFeed(clock); svc, sync = make_service(clock, feed); run_sync(clock, sync)
        for i in range(0, 110):                                                                   # 0..109 s after the read, 1 s steps (still < 2P+15)
            clock.t = start + i; c = svc.clock(clock.t)["countdown"]
            self.assertEqual(c["verification"]["status"], "verified", i)
            self.assertTrue(0 <= c["seconds_remaining"] <= 60, c["seconds_remaining"])
            if prev is not None:
                if c["seconds_remaining"] > prev: wraps += 1
                else: self.assertAlmostEqual(prev - c["seconds_remaining"], 1.0, places=6)
            prev = c["seconds_remaining"]
        self.assertEqual(wraps, 2)                                                                # boundaries at 09:01:00 (i=30) and 09:02:00 (i=90)
        clock.t = start + 135; self.assertEqual(svc.clock(clock.t)["countdown"]["verification"]["status"], "verified")
        clock.t = start + 136; self.assertNotEqual(svc.clock(clock.t)["countdown"]["verification"]["status"], "verified")   # extrapolation limit

    def test_pending_results_are_listed_never_invented(self):
        c, _ = self.live(60, "10001", utc(2026, 10, 29, 9, 0, 55), 70.0)                          # two boundaries passed since the read
        self.assertEqual(c["id_source"], "feed+schedule"); self.assertEqual(c["extrapolated_periods"], 2)
        self.assertEqual(c["awaiting_results"], ["20261029100010541", "20261029100010542"][:len(c["awaiting_results"])] if c["awaiting_results"] else [])
        self.assertEqual(c["current_period"], "20261029100010543")

    def test_all_four_modes(self):
        cases = ((30, "10005", 2880), (60, "10001", 1440), (180, "10102", 480), (300, "10003", 288))
        for P, code, per_day in cases:
            start = utc(2026, 10, 29, 9, 0, 0) + P - 5 + (0 if P < 300 else 0)
            start = (int(utc(2026, 10, 29, 9, 0, 0) // P) * P) + P - 5                           # 5 s before a boundary of this mode
            c, feed = self.live(P, code, start, 5.0)
            end = int((start + 5) // P) * P
            n = int((end - utc(2026, 10, 29, 0, 0, 0)) // P) + 1
            self.assertEqual(c["current_period"], pid("20261029", code, n), (P, n)); self.assertAlmostEqual(c["seconds_remaining"], P, places=2)
            self.assertEqual(c["period_seconds"], P)

    def test_utc_day_change_rolls_the_date_and_restarts_rounds(self):
        for P, code, per_day in ((30, "10005", 2880), (60, "10001", 1440), (180, "10102", 480), (300, "10003", 288)):
            start = utc(2026, 10, 29, 23, 59, 55) if P <= 60 else utc(2026, 10, 29, 23, 59, 55) - (P - 60)   # a read just before midnight
            start = (int(utc(2026, 10, 30, 0, 0, 0) // P) * P) - 5
            c0, _ = self.live(P, code, start, 0.0)
            self.assertEqual(c0["current_period"], pid("20261029", code, per_day), P)             # last round of the 29th
            c1, _ = self.live(P, code, start, 5.0)
            self.assertEqual(c1["current_period"], pid("20261030", code, 1), P)                   # first round of the 30th, not ...1441
            self.assertEqual(c1["next_period"], pid("20261030", code, 2))

    def test_next_and_advance_period_id_across_day_month_and_year(self):
        self.assertEqual(next_period_id("20261231100011440"), "20270101100010001")
        self.assertEqual(next_period_id("20240228100011440"), "20240229100010001")               # leap year
        self.assertEqual(next_period_id("20260228100011440"), "20260301100010001")
        self.assertEqual(advance_period_id("20261029100011439", 3), "20261030100010002")
        self.assertEqual(next_period_id("700131"), "700132"); self.assertIsNone(next_period_id("x")); self.assertIsNone(next_period_id(None))

    def test_awaiting_and_late_states_from_a_verified_stored_schedule(self):
        """Without a live feed, the verified stored schedule never advances a period by itself."""
        svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), clock_cfg=ClockConfig(source_tz="UTC"))
        end = utc(2026, 10, 29, 9, 0)
        rows = [{"number": i % 10, "period_id": pid("20261029", "10001", 500 + i), "timestamp": datetime.fromtimestamp(end - (7 - i) * 60, UTC).isoformat()} for i in range(8)]
        svc.import_rows(rows)
        c = lambda dt: svc.clock(end + dt)["countdown"]
        self.assertEqual((c(30)["state"], round(c(30)["seconds_remaining"])), ("open", 30))
        self.assertEqual(c(30)["verification"]["method"], "stored_timestamps")
        self.assertEqual(c(61)["state"], "awaiting_result"); self.assertEqual(c(61)["seconds_remaining"], 0.0)
        self.assertEqual(c(61 + 25)["state"], "late")
        self.assertEqual(svc.clock(end + 61)["next_round"]["period_id"], pid("20261029", "10001", 508))     # unchanged until a result is confirmed


# ------------------------------------------------------------------------------------------------------------- seed data / config / api
class TestSeedAndConfig(unittest.TestCase):
    def test_shipped_seed_contains_no_invented_timestamps(self):
        import csv
        rows = list(csv.DictReader(open(os.path.join(DATA, "wingo_docx_clean.csv"), encoding="utf-8-sig")))
        self.assertGreater(len(rows), 5000)
        self.assertTrue(all((r["timestamp"] or "").strip() == "" for r in rows))

    def test_seed_only_database_is_honestly_unverified(self):
        svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), clock_cfg=ClockConfig())
        svc.import_csv(open(os.path.join(DATA, "wingo_docx_clean.csv"), encoding="utf-8-sig").read())
        c = svc.clock(utc(2026, 10, 29, 9))
        self.assertFalse(c["schedule"]["verified"]); self.assertEqual(c["countdown"]["state"], "unverified")
        self.assertEqual(c["countdown"]["verification"]["status"], "unverified"); self.assertIsNone(c["countdown"]["seconds_remaining"])
        self.assertEqual(c["countdown"]["verification"]["code"], "feed_disabled")

    def test_old_fabricated_seed_timestamps_already_in_a_database_are_not_trusted(self):
        svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), clock_cfg=ClockConfig())
        end = utc(2026, 10, 28, 12, 12)
        svc.import_rows([{"number": i % 10, "period_id": pid("20261006", "10001", 1176 + i), "timestamp": datetime.fromtimestamp(end - (7 - i) * 60, UTC).isoformat()} for i in range(8)])
        c = svc.clock(utc(2026, 10, 28, 12, 12, 30))                                                # the invented times would look perfect
        self.assertFalse(c["schedule"]["verified"]); self.assertEqual(c["schedule"]["code"], "timestamps_contradict_ids")
        self.assertEqual(c["countdown"]["state"], "unverified")

    def test_assumed_schedule_is_labelled_assumed_never_verified(self):
        svc = PredictorService(InMemoryRepository(), PredictionEngine(use_ml=False), clock_cfg=ClockConfig(assumed=True))
        c = svc.clock(utc(2026, 10, 29, 9, 0, 20))["countdown"]
        self.assertEqual((c["state"], c["verification"]["status"]), ("assumed", "assumed"))
        svc.import_rows([{"number": 1, "period_id": None, "timestamp": None}])
        c = svc.clock(utc(2026, 10, 29, 9, 0, 20))["countdown"]
        self.assertEqual((c["state"], c["verification"]["status"]), ("assumed", "assumed"))

    def test_config_flags_and_quotes(self):
        import app.config as cfg
        with mock.patch.dict(os.environ, {"FEED_ENABLED": ' "True" ', "SYNC_ENABLED": "1", "ASSUMED_SCHEDULE": "yes", "AUTO_SEED": "off", "SYNC_RESPECT_ROBOTS": "'false'"}):
            self.assertTrue(cfg._flag("FEED_ENABLED")); self.assertTrue(cfg._flag("SYNC_ENABLED")); self.assertTrue(cfg._flag("ASSUMED_SCHEDULE"))
            self.assertFalse(cfg._flag("AUTO_SEED", "true")); self.assertFalse(cfg._flag("SYNC_RESPECT_ROBOTS", "true")); self.assertFalse(cfg._flag("UNSET_FLAG"))
        self.assertEqual(cfg._unquote('"abc"'), "abc"); self.assertEqual(cfg._unquote("'a b'"), "a b"); self.assertEqual(cfg._unquote('"abc'), '"abc')


class TestClockApi(unittest.TestCase):
    """The /api/clock payload the page consumes (real FastAPI app, in-memory database, scripted feed)."""
    @classmethod
    def setUpClass(cls):
        env = dict(DATABASE_URL="memory://", FEED_ENABLED="false", AUTO_SEED="false", AUTO_BACKTEST="false", SYNC_ENABLED="false", ADMIN_API_KEY="k", ASSUMED_SCHEDULE="false")
        cls._env = mock.patch.dict(os.environ, env); cls._env.start()
        try:
            import importlib
            from fastapi.testclient import TestClient
            import app.config, app.main
            importlib.reload(app.config); importlib.reload(app.main)
            cls.app, cls.TestClient, cls.have = app.main.app, TestClient, True
        except Exception:                                                                          # noqa: BLE001
            cls.have = False

    @classmethod
    def tearDownClass(cls): cls._env.stop()

    def setUp(self):
        if not self.have: self.skipTest("fastapi/httpx not installed")

    def test_api_reports_unverified_with_a_reason_when_no_source_exists(self):
        with self.TestClient(self.app) as c:
            body = c.get("/api/clock").json(); cd = body["countdown"]
            self.assertEqual(cd["state"], "no_data"); self.assertEqual(cd["verification"]["status"], "no_data")
            self.assertEqual(cd["verification"]["code"], "feed_disabled"); self.assertIn("FEED_ENABLED", cd["verification"]["reason"])
            for mode in ("30s", "1min", "3min", "5min"):
                self.assertIn("verification", c.get(f"/api/clock?mode={mode}").json()["countdown"], mode)

    def test_api_serves_a_verified_live_round_and_withdraws_it_when_the_feed_dies(self):
        with self.TestClient(self.app) as c:
            clock = SimClock(utc(2026, 10, 29, 9, 0, 30)); feed = ScriptedFeed(clock); svc = self.app.state.services["1min"]
            svc.now_fn, svc.live_source = clock, feed
            svc.sync_service = SyncService(svc, feed, True, 10, allow_gaps=True); run_sync(clock, svc.sync_service)
            cd = c.get("/api/clock").json()["countdown"]
            self.assertEqual((cd["source"], cd["verification"]["status"], cd["current_period"]), ("feed", "verified", "20261029100010541"))
            self.assertEqual(c.get("/api/clock").headers["cache-control"], "no-store")
            feed.error = SourceError("HTTP 403", "not_authorised"); clock.t += 600; run_sync(clock, svc.sync_service)
            cd = c.get("/api/clock").json()["countdown"]
            self.assertNotEqual(cd["verification"]["status"], "verified"); self.assertNotIn("current_period", cd)
            self.assertEqual(cd["verification"]["code"], "not_authorised")
            self.app.state.syncs["1min"] = svc.sync_service                                           # the route reads the app's own SyncService
            sy = c.get("/api/sync/status").json(); self.assertEqual(sy["last_error_code"], "not_authorised")
            nxt = c.get("/api/prediction/next").json()
            self.assertTrue(nxt.get("estimate_state") in (None, "timing_unverified") or not nxt.get("available"))


if __name__ == "__main__":
    unittest.main()
