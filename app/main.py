"""FastAPI entry point:  uvicorn app.main:app --reload   (run from backend/)"""
import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from app.config import get_settings
from app.database.factory import create_repository
from app.prediction.engine import ENGINE_VERSION, PredictionEngine
from app.services.errors import ServiceError
from app.services.predictor_service import PredictorService
from app.sync.service import SyncService
from app.modes import DEFAULT_MODE, MODES, mode_db_url
from app.sync.sources import HttpJsonSource, WinGoFeedSource

logging.basicConfig(level=get_settings().log_level, format="%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s")
log = logging.getLogger("wingo")
ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT.parent / "frontend"


def _find_seed(name: str):
    p = Path(name)
    for c in (p, ROOT / p, ROOT / "data" / p.name, ROOT.parent / "data" / p.name):     # works from any working directory
        if c.is_file():
            return c
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.settings, app.state.services, app.state.syncs, tasks = s, {}, {}, []
    for mode, m in MODES.items():
        service, sync = _build_mode(s, mode, m)
        app.state.services[mode], app.state.syncs[mode] = service, sync
        if sync.enabled:
            tasks.append(asyncio.create_task(sync.run_forever()))     # no browser needed: runs on the server
    app.state.service, app.state.sync = app.state.services[DEFAULT_MODE], app.state.syncs[DEFAULT_MODE]
    yield
    for t in tasks:
        t.cancel()


def _build_mode(s, mode: str, m: dict):
    repo = create_repository(mode_db_url(s.database_url, mode)); repo.init()
    service = PredictorService(repo, PredictionEngine(refit_every=s.refit_every), s.min_history, s.min_validation_n,
                               clock_cfg=s.clock_config(m["seconds"], force_assumed=not m["type_id"]))     # no source for this mode: labelled, assumed timer
    seed = _find_seed(s.seed_csv)
    if mode == DEFAULT_MODE and s.auto_seed and repo.last_result() is None:
        if seed:
            log.info("seeding from %s", seed)
            service.import_csv(seed.read_text(encoding="utf-8"))
        else:
            log.warning("AUTO_SEED is on but %s was not found; the database stays empty", s.seed_csv)
    last = repo.last_result()
    if s.auto_backtest and last and last["seq"] > s.min_history + 50 and repo.latest_backtest() is None:
        threading.Thread(target=service.run_backtest, daemon=True).start()
    sync_headers = json.loads(s.sync_headers_json) if s.sync_headers_json else {}
    if s.sync_auth_header and "Authorization" not in sync_headers:     # securely-stored token -> Authorization header
        sync_headers["Authorization"] = s.sync_auth_header
    if s.feed_enabled and s.feed_url and m["type_id"]:
        source = WinGoFeedSource(s.feed_url, m["type_id"], m["seconds"], backfill_rows=s.feed_backfill_rows,
                                 respect_robots=s.sync_respect_robots, timeout=s.feed_timeout)
        service.live_source = source
        sync = SyncService(service, source, True, 10, stale_after=max(120, 4 * m["seconds"]), allow_gaps=True, fast_interval=3)
    elif mode == DEFAULT_MODE:
        source = HttpJsonSource(s.sync_source_url, s.sync_list_path, s.sync_period_field, s.sync_number_field, s.sync_time_field,
                                sync_headers, respect_robots=s.sync_respect_robots)
        sync = SyncService(service, source, s.sync_enabled, s.sync_interval, stale_after=s.sync_stale_after, allow_gaps=s.sync_allow_gaps,
                           start_after=s.sync_start_after_period, fast_interval=s.sync_fast_interval)
    else:
        sync = SyncService(service, HttpJsonSource(), False)
    return service, sync


app = FastAPI(title="WinGo Analysis Platform", version=ENGINE_VERSION, lifespan=lifespan,
              description="Research/analysis tool. Statistical candidates only - no guarantees, no betting functionality.")
app.add_middleware(CORSMiddleware, allow_origins=list(get_settings().allowed_origins), allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-API-Key"])


@app.middleware("http")
async def no_store_for_api(request: Request, call_next):
    """Live data must never be served from a browser/CDN/proxy cache as if it were current."""
    resp = await call_next(request)
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError):
    return JSONResponse(status_code=exc.status, content={"error": str(exc)})


app.include_router(router)
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
