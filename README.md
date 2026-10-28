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

## Live results (optional, OFF by default)
Set `SYNC_ENABLED=true`, `SYNC_SOURCE_URL`, `SYNC_LIST_PATH`, `SYNC_PERIOD_FIELD`, `SYNC_NUMBER_FIELD` and `SYNC_TIME_FIELD` in `.env`. Only use a source you are authorized to use. A token in `SYNC_AUTH_HEADER` is sent as the `Authorization` header.

## Tests
```bash
cd Backend-main && python3 -m pytest -q
cd ../frontend && node --test test/
```

## Key API endpoints
| Method | Path | Notes |
|---|---|---|
| GET | /api/clock | schedule state, countdown, next period |
| GET | /api/prediction/next | estimate for the next period (one per period) |
| GET | /api/stats/report | accuracy, hot/cold 50/100, streaks, daily summary |
| GET | /api/results | stored rounds |
| POST | /api/results, /api/data/import/* | need `X-API-Key` |

## Not included (on purpose)
- `.env` files and all secret values (admin key, sync token)
- `__pycache__`, `.pytest_cache`, the duplicate `Backend-main/app/app/` copy, old `b.zip`/`f.zip`
- `WINGO-PRECITOR-main/`: an identical copy of `frontend/`
