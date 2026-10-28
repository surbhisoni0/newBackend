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
