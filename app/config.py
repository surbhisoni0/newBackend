"""Settings come from environment variables / backend/.env. No secrets live in code."""
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    p = Path(__file__).resolve().parents[1] / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "mongodb://localhost:27017/wingo")
    admin_api_key: str = os.getenv("ADMIN_API_KEY", "")
    allowed_origins: tuple = tuple(o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",") if o.strip())
    auto_seed: bool = os.getenv("AUTO_SEED", "true").lower() == "true"
    seed_csv: str = os.getenv("SEED_CSV", "data/wingo_transcribed.csv")
    auto_backtest: bool = os.getenv("AUTO_BACKTEST", "false").lower() == "true"     # the walk-forward test takes minutes of CPU: off unless asked for
    refit_every: int = int(os.getenv("MODEL_REFIT_EVERY", "10"))
    min_history: int = int(os.getenv("MIN_HISTORY", "100"))
    min_validation_n: int = int(os.getenv("MIN_VALIDATION_N", "300"))
    # live sync: OFF unless you configure a data source you are allowed to use
    ui_poll_seconds: float = max(10.0, float(os.getenv("UI_POLL_SECONDS", "30")))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    sync_stale_after: float = float(os.getenv("SYNC_STALE_AFTER_SECONDS", "300"))
    sync_enabled: bool = os.getenv("SYNC_ENABLED", "false").lower() == "true"
    sync_interval: float = float(os.getenv("SYNC_INTERVAL_SECONDS", "30"))
    sync_source_url: str = os.getenv("SYNC_SOURCE_URL", "")
    sync_list_path: str = os.getenv("SYNC_LIST_PATH", "")
    sync_period_field: str = os.getenv("SYNC_PERIOD_FIELD", "")
    sync_number_field: str = os.getenv("SYNC_NUMBER_FIELD", "")
    sync_time_field: str = os.getenv("SYNC_TIME_FIELD", "")
    sync_headers_json: str = os.getenv("SYNC_HEADERS_JSON", "")
    sync_auth_header: str = os.getenv("SYNC_AUTH_HEADER", "")             # securely-stored token; sent as Authorization header
    sync_respect_robots: bool = os.getenv("SYNC_RESPECT_ROBOTS", "true").lower() == "true"
    sync_allow_gaps: bool = os.getenv("SYNC_ALLOW_GAPS", "false").lower() == "true"
    sync_start_after_period: str = os.getenv("SYNC_START_AFTER_PERIOD", "")
    sync_fast_interval: float = max(3.0, float(os.getenv("SYNC_FAST_INTERVAL_SECONDS", "5")))   # only while a result is overdue
    # period clock: the schedule is VERIFIED from stored timestamps, never assumed (see app/timing.py)
    period_seconds: int = int(os.getenv("PERIOD_SECONDS", "60"))
    source_timezone: str = os.getenv("SOURCE_TIMEZONE", "UTC")             # zone of timestamps that carry no UTC offset
    timestamp_means: str = os.getenv("TIMESTAMP_MEANS", "end").lower()     # "end" = draw time, "start" = round start
    clock_tolerance_seconds: float = float(os.getenv("CLOCK_TOLERANCE_SECONDS", "5"))
    clock_min_samples: int = int(os.getenv("CLOCK_MIN_SAMPLES", "5"))
    late_grace_seconds: float = float(os.getenv("LATE_GRACE_SECONDS", "20"))
    assumed_schedule: bool = os.getenv("ASSUMED_SCHEDULE", "false").lower() == "true"     # opt-in fallback countdown, always labelled "assumed"
    assumed_offset_seconds: float = float(os.getenv("ASSUMED_OFFSET_SECONDS", "0"))
    feed_enabled: bool = os.getenv("FEED_ENABLED", "false").lower() == "true"     # WinGo results feed (user confirmed permission)
    feed_url: str = os.getenv("FEED_URL", "")
    feed_backfill_rows: int = int(os.getenv("FEED_BACKFILL_ROWS", "1000"))
    feed_timeout: float = float(os.getenv("FEED_TIMEOUT", "25"))      # seconds per request; a slow upstream must not stall sync forever

    def clock_config(self, period_seconds=None, force_assumed=False):
        from app.timing import ClockConfig
        # The live feed reports IST times without an offset (see WinGoFeedSource), so rows it stored earlier are IST text:
        # read them as IST whatever SOURCE_TIMEZONE says, otherwise they look ~5.5 h in the future -> "Timing unverified".
        tz = "Asia/Kolkata" if (self.feed_enabled and self.feed_url) else self.source_timezone
        return ClockConfig(period_seconds or self.period_seconds, tz, self.timestamp_means, self.clock_tolerance_seconds,
                           self.clock_min_samples, self.late_grace_seconds, self.assumed_schedule or force_assumed, self.assumed_offset_seconds)


def get_settings() -> Settings:
    return Settings()
