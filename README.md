# WinGo Big/Small Lab

A statistics dashboard for 1-minute WinGo rounds. It shows a server-controlled countdown, period IDs, an experimental Big/Small and colour estimate, a walk-forward accuracy report, hot/cold numbers, streak alerts and a daily summary.

> These are statistical estimates only. Testing on 4,986 past rounds shows them to be no better than guessing. The app places no bets.

## Folder structure
```
wingo-project/
├── Backend-main/          FastAPI backend (also serves the frontend at "/")
│   ├── app/               API, prediction engine, timing, services, database
│   ├── data/              Seed CSVs + docx_to_csv.py converter
│   ├── tests/             pytest suite
│   ├── requirements.txt   Python dependencies
│   └── .env.example       Configuration template (no secrets)
├── frontend/              Static HTML / vanilla JS / CSS (no build step)
│   ├── test/, e2e/        Node tests
│   └── vercel.json        Optional Vercel config (rewrites /api)
├── assets/Data.docx       Original round history you uploaded
├── start_backend.sh       Starts the app on port 8099
├── CHANGES.md             Diagnosis and change notes
└── README.md
```

## Requirements
- Python 3.10+
- MongoDB 5+ running locally, or a hosted MongoDB such as Atlas
- (Optional) Node 18+ to run the frontend tests

## Setup
```bash
cd wingo-project/Backend-main
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Edit `.env`:
- `DATABASE_URL`: your MongoDB URL, e.g. `mongodb://localhost:27017/wingo`
- `ADMIN_API_KEY`: set a long random value. **Saving rounds is switched off while this is blank.** Type the same value in the "Admin key" box on the Add rounds tab.
- `ALLOWED_ORIGINS`: the browser origin, e.g. `http://localhost:8099`
- Keep comments on their own lines: the loader does not strip `# comments` that come after a value.

## Run
```bash
cd wingo-project
./start_backend.sh
```
Then open http://localhost:8099. On first start, when the database is empty, the backend imports `data/wingo_docx_clean.csv` (5,086 rounds).

Run the walk-forward accuracy test once, so the Stats tab can show accuracy. It takes about 25 minutes:
```bash
curl -X POST http://localhost:8099/api/backtest/run -H "X-API-Key: <your ADMIN_API_KEY>"
```

## Re-importing Data.docx
```bash
cd Backend-main
python3 data/docx_to_csv.py ../assets/Data.docx data/wingo_docx_clean.csv
```
Data.docx has no times, so the script **estimates** them. The newest round is given the current minute, and earlier rounds count back 1 minute per period. To re-seed, drop the `wingo` database and restart.

## Live results and timing (WinGo feed, ON by default)
The app reads the same results feed that WingoRadar uses, for all four modes: 30s, 1 min, 3 min and 5 min.
- `FEED_ENABLED=true` and `FEED_URL=https://wingoradar.com/api/getwinresult`. You confirmed you have permission to use this feed.
- The current period and its end time come from the feed's OPEN round, corrected by the feed's clock (from its `Date` header). At each boundary the timer rolls straight into the next period, the same way WingoRadar does.
- Period ID = `YYYYMMDD` (UTC date) + 5-digit game code (30s `10005`, 1min `10001`, 3min `10102`, 5min `10003`) + 4-digit round of the UTC day.
- Only settled rounds (state 1 with a number) are stored. Ended rounds without a result are listed as "awaiting", never guessed.
- 3min: the feed currently publishes no winning numbers, so 3min shows timer and period only, with no predictions.
- Each mode has its own database: 1min = `wingo` (includes the Data.docx history), others = `wingo_30s`, `wingo_3min`, `wingo_5min`.
- On first start each mode loads up to `FEED_BACKFILL_ROWS` (1000) recent rounds.

## How the Big/Small estimate updates
Once per timer cycle (30 s, 1 min or 2 min). A verified result for the round that just ended produces a new stored pick, scored later against its own round.
If no verified result arrives within 5 s, one clearly labelled projection is calculated for the round now running. It is never stored or scored.
Same verified data gives the same value (the models are seeded), so the value changes only when the data does. See CHANGES.md.

## Tests
```bash
cd Backend-main && python3 -m pytest -q
cd ../frontend && node --test test/
```

## Key API endpoints
| Method | Path | Notes |
|---|---|---|
| GET | /api/modes | supported game modes |
| GET | /api/clock?mode=1min | current/next period, time left, latest confirmed, awaiting results |
| GET | /api/prediction/next | estimate for the next period (one per period) |
| GET | /api/stats/report | accuracy, hot/cold 50/100, streaks, daily summary |
| GET | /api/results | stored rounds |
| POST | /api/results, /api/data/import/* | need `X-API-Key` |

## Not included (on purpose)
- `.env` files and all secret values (admin key, sync token)
- `__pycache__`, `.pytest_cache`, the duplicate `Backend-main/app/app/` copy, old `b.zip`/`f.zip`
- `WINGO-PRECITOR-main/`: an identical copy of `frontend/`
