"""Result sources for live synchronisation.

Nothing in this file knows any particular website. The URL and field names come from YOUR configuration (backend/.env), and
only for a data source you are allowed to use (an official API, your own service, or a feed the operator permits).
The HTTP source reads robots.txt first and refuses to fetch anything that file disallows. It never logs in, never rotates
IPs and never works around blocks.
"""
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

log = logging.getLogger("wingo.sync")


class SourceError(Exception):
    """The source could not be read, or it returned something we refuse to trust."""


@dataclass(frozen=True)
class RoundRecord:
    period_id: str
    number: int
    timestamp: Optional[str] = None


def robots_allows(robots_txt: str, url: str, agent: str) -> bool:
    rp = RobotFileParser()
    rp.parse(robots_txt.splitlines())
    return rp.can_fetch(agent, url)


def _dig(obj, path: str):
    for part in filter(None, path.split(".")):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
            obj = obj[int(part)]
        else:
            raise SourceError(f"{path!r} not found in the response (stopped at {part!r})")
    return obj


def _digit(value) -> int:
    if isinstance(value, bool):
        raise ValueError("bool")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("fractional")
        value = int(value)
    n = int(str(value).strip())
    if not 0 <= n <= 9:
        raise ValueError("out of range")
    return n


def parse_records(payload, list_path: str, period_field: str, number_field: str, time_field: str = ""):
    """Validate every item. Returns (records, number_of_invalid_items). Invalid items are dropped, never repaired."""
    items = _dig(payload, list_path) if list_path else payload
    if not isinstance(items, list):
        raise SourceError("expected a list of rounds in the response")
    good, bad = [], 0
    for it in items:
        try:
            period = str(it[period_field]).strip()
            if not period.isdigit():
                raise ValueError("period id must be digits")
            ts = str(it[time_field]) if time_field and it.get(time_field) is not None else None
            good.append(RoundRecord(period, _digit(it[number_field]), ts))
        except (KeyError, TypeError, ValueError, AttributeError):
            bad += 1
    return good, bad


class HttpJsonSource:
    def __init__(self, url="", list_path="", period_field="", number_field="", time_field="", headers=None,
                 timeout=10.0, retries=2, respect_robots=True, agent="WinGoAnalysisBot"):
        self.url, self.list_path, self.period_field, self.number_field = url, list_path, period_field, number_field
        self.time_field, self.headers, self.timeout, self.retries = time_field, dict(headers or {}), timeout, retries
        self.respect_robots, self.agent = respect_robots, agent
        self.last_invalid = 0
        self._robots_ok_until = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.url and self.period_field and self.number_field)

    @property
    def name(self) -> str:
        return urlsplit(self.url).netloc or "not configured"

    def _http_get(self, url: str):
        import httpx                                           # imported lazily so tests run without it
        r = httpx.get(url, headers={"User-Agent": self.agent, "Accept": "application/json", **self.headers},
                      timeout=self.timeout, follow_redirects=False)
        return r.status_code, r.text

    def _check_robots(self):
        if not self.respect_robots or time.time() < self._robots_ok_until:
            return
        parts = urlsplit(self.url)
        try:
            status, text = self._http_get(f"{parts.scheme}://{parts.netloc}/robots.txt")
        except Exception as e:                                 # noqa: BLE001
            raise SourceError(f"could not read robots.txt: {e}") from e
        if status == 200 and not robots_allows(text, self.url, self.agent):
            raise SourceError("robots.txt of this site disallows automated access to that URL; not fetching")
        if status >= 500:
            raise SourceError(f"robots.txt unavailable (HTTP {status}); not fetching")
        self._robots_ok_until = time.time() + 3600             # 4xx = no robots rules (RFC 9309)

    def fetch(self, backfill: bool = False):
        if not self.configured:
            raise SourceError("no data source configured")
        self._check_robots()
        last = "unknown error"
        for attempt in range(self.retries + 1):
            try:
                status, text = self._http_get(self.url)
            except Exception as e:                             # noqa: BLE001
                last = f"network error: {e}"
            else:
                if status == 200:
                    try:
                        payload = json.loads(text)
                    except ValueError as e:
                        raise SourceError(f"response is not JSON: {e}") from e
                    records, self.last_invalid = parse_records(payload, self.list_path, self.period_field, self.number_field, self.time_field)
                    if not records:
                        raise SourceError(f"no valid rounds in the response ({self.last_invalid} invalid items)")
                    return records
                if status in (401, 403):
                    raise SourceError(f"not authorised (HTTP {status}); this source needs access you have not configured")
                if status == 429:
                    raise SourceError("rate limited (HTTP 429); slow down SYNC_INTERVAL_SECONDS")
                last = f"HTTP {status}"
            if attempt < self.retries:
                time.sleep(0.5 * 2 ** attempt)
        raise SourceError(f"gave up after {self.retries + 1} attempts: {last}")


IST = timezone(timedelta(hours=5, minutes=30))


def _feed_ts(value) -> Optional[str]:
    """The feed's times are IST with no offset. Store them as UTC ("...Z") so SOURCE_TIMEZONE can never misread them
    (read as UTC they land 5.5 h in the future and the schedule is reported as unverified). Unparseable -> None, never guessed."""
    from app.timing import iso, parse_ts
    return iso(parse_ts(value, "Asia/Kolkata"))


class WinGoFeedSource(HttpJsonSource):
    """WinGo results feed used by WingoRadar (POST, JSON). Verified fields: issueNumber, number, state (0 open / 1 settled),
    startTime / endTime (IST, no offset), intervalM. The request's `Time` is the IST trading day. Also remembers the OPEN round
    and the upstream clock offset (from the Date header) so the countdown follows the source's own schedule."""

    def __init__(self, url: str, type_id: int, period_seconds: int, page_size: int = 50, backfill_rows: int = 1000, **kw):
        super().__init__(url, "data", "issueNumber", "number", "endTime", **kw)
        self.type_id, self.period_seconds, self.page_size, self.backfill_rows = type_id, period_seconds, page_size, backfill_rows
        self.open_round, self.clock_offset, self.fetched_at = None, None, None

    def _post(self, day: str, page: int, size: int):
        import httpx
        t0 = time.time()
        r = httpx.post(self.url, json={"PageIndex": page, "PageSize": size, "Time": day, "TypeId": self.type_id},
                       headers={"User-Agent": self.agent, "Accept": "application/json", **self.headers}, timeout=30 if size > 50 else self.timeout)
        t1 = time.time()
        if r.status_code in (401, 403):
            raise SourceError(f"not authorised (HTTP {r.status_code})")
        if r.status_code != 200:
            raise SourceError(f"HTTP {r.status_code}")
        date = r.headers.get("x-upstream-date") or r.headers.get("date")
        if date:
            from email.utils import parsedate_to_datetime
            self.clock_offset = round(parsedate_to_datetime(date).timestamp() - (t0 + t1) / 2, 3)
        try:
            return r.json()
        except ValueError as e:
            raise SourceError(f"response is not JSON: {e}") from e

    def _settled(self, items: list) -> list:
        good, bad = [], 0
        for it in items:
            if it.get("state") != 1 or str(it.get("number", "")).strip() == "":
                continue                                           # open round: no result yet, never stored
            try:
                period = str(it["issueNumber"]).strip()
                if not period.isdigit():
                    raise ValueError("period")
                good.append(RoundRecord(period, _digit(it["number"]), _feed_ts(it["endTime"])))
            except (KeyError, TypeError, ValueError):
                bad += 1
        self.last_invalid = bad
        return good

    def _remember_open(self, items: list):
        from app.timing import parse_ts
        op = next((it for it in items if it.get("state") == 0), None)
        if op:
            self.open_round = {"period_id": str(op["issueNumber"]), "start": parse_ts(op.get("startTime"), "Asia/Kolkata"),
                               "end": parse_ts(op.get("endTime"), "Asia/Kolkata"),
                               "interval_seconds": round(float(op.get("intervalM") or 0) * 60) or self.period_seconds}
        self.fetched_at = time.time()

    def fetch(self, backfill: bool = False):
        if not self.configured:
            raise SourceError("no data source configured")
        self._check_robots()
        today = datetime.now(IST).date()
        payload = self._post(today.isoformat(), 1, self.page_size)
        items = list(payload.get("data") or [])
        self._remember_open(items)
        if backfill:                                               # first run: load recent history so predictions can start
            items, page, day = [], 1, today
            while len(items) < self.backfill_rows and day >= today - timedelta(days=1):
                chunk = list(self._post(day.isoformat(), page, 200).get("data") or [])
                items += chunk
                if len(chunk) < 200:
                    day, page = day - timedelta(days=1), 1
                else:
                    page += 1
        records = self._settled(items)
        if not records and not self.open_round:
            raise SourceError("no settled rounds in the response")
        self.no_numbers = not records and any(it.get("state") == 1 for it in items)
        return records
