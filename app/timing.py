"""Period clock: decides whether the game's one-minute schedule can be TRUSTED, and what state the current round is in.

Nothing here assumes a schedule. A schedule counts as verified only when the stored, confirmed rounds themselves prove it:
several consecutive period ids whose timestamps are spaced exactly one period apart (within a tolerance). Without that
evidence (no timestamps, gaps, jitter, a timestamp in the future) there is no countdown and the API says why.

The module is pure (no I/O, `now` is always passed in) so every state can be unit-tested deterministically.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MAX_SAMPLES = 30
MIN_AGREEING_SHARE = 0.8                                           # share of samples that must sit on the schedule


@dataclass(frozen=True)
class ClockConfig:
    period_seconds: int = 60
    source_tz: str = "UTC"            # zone assumed for timestamps that carry no UTC offset
    time_means: str = "end"           # what a stored timestamp marks: "end" (draw time) or "start" of its round
    tolerance_seconds: float = 5.0    # allowed jitter between timestamps and the ideal schedule
    min_samples: int = 5              # consecutive rounds needed before the schedule counts as verified
    late_grace_seconds: float = 20.0  # after the boundary: "awaiting result" for this long, then "late"
    assumed: bool = False             # opt-in: when the schedule cannot be VERIFIED, count down to an ASSUMED boundary instead
    assumed_offset_seconds: float = 0.0   # rounds are assumed to end at (k * period + offset) seconds of the UTC epoch

    def __post_init__(self):
        if self.period_seconds < 5:
            raise ValueError("period_seconds must be >= 5")
        if self.time_means not in ("end", "start"):
            raise ValueError("time_means must be 'end' or 'start'")
        if self.min_samples < 2:
            raise ValueError("min_samples must be >= 2")
        try:
            ZoneInfo(self.source_tz)
        except (ZoneInfoNotFoundError, ValueError, KeyError) as e:
            raise ValueError(f"unknown timezone {self.source_tz!r}") from e


def next_period_id(period_id: Optional[str]) -> Optional[str]:
    """Successor of a numeric period id, keeping leading zeros. None if the id is missing or not numeric."""
    if not period_id or not str(period_id).isdigit():
        return None
    s = str(period_id)
    per_day = _ROUNDS_PER_DAY.get(s[8:13]) if len(s) == 17 else None
    if per_day and int(s[13:]) >= per_day:                          # last round of the UTC day -> next date, round 0001
        try:
            d = datetime.strptime(s[:8], "%Y%m%d") + timedelta(days=1)
            return f"{d:%Y%m%d}{s[8:13]}0001"
        except ValueError:
            pass
    return str(int(s) + 1).zfill(len(s))


def advance_period_id(period_id: Optional[str], k: int) -> Optional[str]:
    for _ in range(max(0, k)):
        period_id = next_period_id(period_id)
    return period_id


# WinGo ids: YYYYMMDD (UTC date) + 5-digit game code + 4-digit round of the UTC day (verified on the live feed)
_ROUNDS_PER_DAY = {"10005": 2880, "10001": 1440, "10102": 480, "10003": 288}


def parse_ts(value, source_tz: str = "UTC") -> Optional[float]:
    """Stored timestamp -> UTC epoch seconds. Accepts ISO-8601 (with or without offset) and epoch seconds/milliseconds."""
    if value is None or str(value).strip() == "":
        return None
    s = str(value).strip()
    try:
        if s.replace(".", "", 1).isdigit() and len(s.split(".")[0]) >= 9:     # >= 9 digits: an epoch, not a compact date
            x = float(s)
            return x / 1000.0 if x > 1e11 else x          # > 1e11 can only be milliseconds (that is year 5138 in seconds)
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(source_tz))
    return dt.astimezone(timezone.utc).timestamp()


def verify_schedule(rows_newest_first: list, cfg: ClockConfig) -> dict:
    """rows: [{'period_id','timestamp',...}] newest first. Returns the schedule verdict.

    For every sample i that is k_i rounds older than the newest, the newest round's timestamp is estimated as
    ts_i + k_i * period. The schedule is verified when enough of those estimates (>= min_samples and >= 80%) agree with
    their median within the tolerance; the rest are reported as outliers instead of vetoing the schedule.
    """
    P = float(cfg.period_seconds)
    out = {"verified": False, "period_seconds": cfg.period_seconds, "time_means": cfg.time_means, "source_tz": cfg.source_tz,
           "samples": 0, "spread_seconds": None, "outliers": 0, "anchor_period_id": None, "anchor_epoch": None, "reason": None}
    samples = []
    for r in rows_newest_first[:MAX_SAMPLES]:
        pid, ts = r.get("period_id"), parse_ts(r.get("timestamp"), cfg.source_tz)
        if not pid or not str(pid).isdigit() or ts is None:
            break                                                    # the verified run must be unbroken from the newest round
        if samples and int(samples[-1][0]) - int(pid) != 1:
            break                                                    # a gap in period ids ends the run
        samples.append((str(pid), ts))
    out["samples"] = len(samples)
    if len(samples) < cfg.min_samples:
        out["reason"] = (f"Need {cfg.min_samples} consecutive rounds with period ids AND timestamps to verify the schedule; "
                         f"have {len(samples)}.")
        return out
    newest_id = samples[0][0]
    estimates = [ts + k * P for k, (_, ts) in enumerate(samples)]
    centre = median(estimates)                                      # robust: one late/jittered timestamp cannot move it
    dev = [abs(e - centre) for e in estimates]
    inliers = [d for d in dev if d <= cfg.tolerance_seconds]
    out.update(spread_seconds=round(max(inliers), 3) if inliers else None, outliers=len(dev) - len(inliers),
               anchor_period_id=newest_id, anchor_epoch=centre)
    if len(inliers) < cfg.min_samples or len(inliers) / len(dev) < MIN_AGREEING_SHARE:
        out["reason"] = (f"Timestamps do not follow a {cfg.period_seconds}s schedule: only {len(inliers)} of the last {len(dev)} rounds "
                         f"agree within {cfg.tolerance_seconds:.0f}s (need {cfg.min_samples}+ and {MIN_AGREEING_SHARE:.0%}).")
        return out
    out["verified"] = True
    return out


def countdown(schedule: dict, now: float, cfg: ClockConfig) -> dict:
    """State of the first round that has no confirmed result yet (the one after the latest confirmed round)."""
    base = {"state": "unverified", "round_ends_at": None, "seconds_remaining": None, "overdue_seconds": None,
            "rounds_ended_since_confirmed": None, "reason": schedule.get("reason")}
    if not schedule["verified"]:
        return base
    P = float(cfg.period_seconds)
    latest_end = schedule["anchor_epoch"] + (P if cfg.time_means == "start" else 0.0)
    elapsed = now - latest_end
    if elapsed < -cfg.tolerance_seconds:
        base["reason"] = (f"The newest confirmed round is timestamped {-elapsed:.0f}s in the future: timezone or clock mismatch "
                          f"(check SOURCE_TIMEZONE / TIMESTAMP_MEANS). Countdown disabled.")
        return base
    elapsed = max(0.0, elapsed)
    ended = int(elapsed // P)                                       # rounds that have finished since the newest confirmed one
    ends_at = latest_end + P                                        # end of the first unconfirmed round
    base["rounds_ended_since_confirmed"] = ended
    base["reason"] = None
    if ended == 0:
        base.update(state="open", round_ends_at=ends_at, seconds_remaining=max(0.0, ends_at - now))
        return base
    overdue = now - ends_at
    base.update(round_ends_at=ends_at, seconds_remaining=0.0, overdue_seconds=overdue)
    base["state"] = "awaiting_result" if (ended == 1 and overdue <= cfg.late_grace_seconds) else "late"
    return base


def assumed_countdown(now: float, cfg: ClockConfig) -> dict:
    """OPT-IN fallback (ASSUMED_SCHEDULE=true). Nothing verified this schedule: it only says rounds are assumed to end every
    `period_seconds`, aligned to the UTC epoch plus `assumed_offset_seconds`. The state is 'assumed', never 'open', so the UI can
    never present it as verified. It does not know about results and never advances a period id."""
    P, off = float(cfg.period_seconds), float(cfg.assumed_offset_seconds)
    ends_at = (int((now - off) // P) + 1) * P + off
    return {"state": "assumed", "round_ends_at": ends_at, "seconds_remaining": max(0.0, ends_at - now), "period_seconds": cfg.period_seconds,
            "overdue_seconds": None, "rounds_ended_since_confirmed": None,
            "reason": "Assumed schedule (not verified): the timer counts to every full period on the clock, not to your data source."}


def iso(epoch: Optional[float]) -> Optional[str]:
    return None if epoch is None else datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")
