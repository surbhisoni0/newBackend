# Per-cycle Big/Small refresh (30 s / 1 min / 2 min)

## Problem
The displayed Big/Small never changed between cycles. `PredictorService._next_prediction()` keys the stored pick on
`last verified result + 1`, so while no new verified result was stored the same pending row (same id, round, time and value) was returned
forever. Nothing marked the cycle boundary, and the page could not tell the pick was for a round that had already ended.
(HTTP caching was ruled out: `/api/*` already sends `Cache-Control: no-store`, and the browser fetches with `cache: "no-store"`.)

## What now happens, once per completed cycle
1. A stored **pick** is made when a verified result arrives (unchanged). It is scored only against its own round's verified result.
2. For `SETTLE_SECONDS` (5 s) after a cycle boundary the app waits for the ended round's verified result. Normal case: it arrives, a new pick is made from it.
3. If it has not arrived, the app builds one **projection** for the round now running, from all verified history (`PredictorService._projection`).
   It is computed once per cycle, held steady for the whole countdown, labelled "Projection, not a result and not scored",
   and is never written to the database or counted in live accuracy.
4. `/api/prediction/next` now also returns `kind` ("pick" | "projection"), `cycle`, `unverified_cycles`, `settling` and `refresh_in_ms`
   (when the answer can next change without a new result). The page re-reads exactly once at that moment; nothing runs per second.
Cycle boundaries are taken from the countdown's own `round_ends_at`, so the pick rolls over at the instant the on-screen timer does, in every mode.

## What this does NOT do
- It does not make the Big/Small value change by itself. The engine is deterministic (seeded models): the same verified history gives the same value.
  With no new verified result the value can legitimately stay the same; only the round, calculation time and label move. Nothing is randomised.
- It does not guarantee accuracy. Backtests still show no reliable edge over a coin flip.
- 2 min has no results feed, so it shows "Need at least 100 results" until you add 2-minute rounds (Add rounds tab / CSV import with `mode=2min`).

## Files
`app/services/predictor_service.py` (cycle logic), `app/database/mongo_repo.py` (persist `made_at`), `tests/test_cycle_refresh.py` (new),
`tests/test_timing.py` and `tests/test_mongo_repo.py` (one assertion updated / one test added). Frontend: `app.js`, `test/cycle.test.js` (new).
Redeploy BOTH the backend and the frontend; the frontend alone cannot fix this.

---

# WinGo fix — root cause, change, and verification

## Summary
No source-code changes were required. The uploaded backend already implements, correctly:
- server-authoritative timing (`app/timing.py`, `/api/clock`) — the browser derives the countdown
  from the server's absolute `round_ends_at` + a measured clock offset, so it does NOT reset on refresh;
- period-specific predictions bound to one target round (`predictor_service._next_prediction`);
- duplicate prevention — one prediction per `target_seq`; repeated `/api/prediction/next` returns the
  SAME stored pending prediction instead of creating a new one;
- evaluate-on-result — adding a result resolves the pending prediction, scores it, then generates the
  next prediction (`add_result` -> `_resolve` -> `_next_prediction`).

## Root cause of the reported symptoms
The app was running with **no timing enabled and no live/period data**:
- the seed data (and the provided PDF) have **no period IDs and no timestamps**, so the schedule can
  never be *verified* -> the countdown showed `--:--`;
- `ASSUMED_SCHEDULE` was off, so there was no fallback ticking timer either;
- live sync was off and no authorized results API exists, so no new rounds were ever confirmed ->
  the single pending prediction (target `#803`) appeared "stuck/duplicated" (expected behaviour, not a bug).

## The change (configuration only) — `backend/.env`
```
ASSUMED_SCHEDULE=true        # enables the built-in 1-minute countdown (epoch-aligned, refresh-stable)
SOURCE_TIMEZONE=Asia/Kolkata # your timestamps' zone
AUTO_SEED=true               # loads the 802-round historical CSV into an empty DB
AUTO_BACKTEST=true           # optional: walk-forward accuracy evaluation
PERIOD_SECONDS=60
```
Apply these to the backend (Render) environment. Nothing else changes.

## Verified here (port 8099, local MongoDB, 802 seeded rounds)
- Countdown ticks to the next full minute; `round_ends_at` identical across repeated `/api/clock`
  reads 5s apart (40.2s -> 35.2s remaining, same `10:45:00Z`) => does not reset on refresh.
- `/api/prediction/next` called twice returned the identical prediction (id=1, target 803) => no duplicate.
- POST `/api/results {"number":7}` resolved prediction for 803 (actual=7, color HIT, big/small MISS),
  advanced to target 804, and generated a NEW prediction (id=2, different top3) => period progression +
  evaluation + fresh per-period prediction.
- `/api/prediction/history` shows 803 resolved (MISS) and 804 pending.

## Limitation (as requested, stated clearly)
A live countdown that matches the real game clock, auto-advancing real period numbers, and auto-fetched
live results ALL require a live results API that returns the newest round as it is drawn, with intact
period IDs and real timestamps. The provided `Data_Report.pdf` is a historical export whose period IDs are
Excel-corrupted (`2.03E+16`) and which has no timestamps, so it cannot drive live timing. The countdown is
therefore shown honestly as "Assumed schedule / NOT VERIFIED". To make it fully live, provide the authorized
endpoint URL + JSON field names (period id, number, time) and set `SYNC_ENABLED=true` with the `SYNC_*` vars.
