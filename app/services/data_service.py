"""Validation and parsing of incoming results. Everything is checked before anything is stored."""
import csv
import io
import re
from datetime import datetime, timezone
from typing import Optional
from app.services.errors import ValidationFailure

NUMBER_KEYS = ("number", "result", "digit", "num")
PERIOD_KEYS = ("period_id", "period", "round", "issue", "issue_number")
TIME_KEYS = ("timestamp", "time", "datetime", "date")
PERIOD_RE = re.compile(r"^\d{1,32}$")


def clean_number(v) -> int:
    if isinstance(v, bool):
        raise ValidationFailure("number must be an integer 0-9")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d\s*", v):
        v = int(v)
    if not isinstance(v, int) or not 0 <= v <= 9:
        raise ValidationFailure(f"invalid number {v!r}: must be an integer 0-9")
    return v


def clean_period(v) -> Optional[str]:
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip()
    if "e" in s.lower() or "." in s:
        raise ValidationFailure(f"period id {s!r} looks like scientific notation/decimal (Excel corruption?). Use text digits.")
    if not PERIOD_RE.match(s):
        raise ValidationFailure(f"invalid period id {s!r}: digits only, max 32 chars")
    return s


def clean_timestamp(v) -> Optional[str]:
    """ISO-8601 (with or without offset) or epoch seconds/milliseconds -> normalised ISO-8601. Epochs become UTC."""
    if v is None or str(v).strip() == "":
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    s = str(v).strip()
    if not isinstance(v, bool) and re.fullmatch(r"\d{9,}(\.\d+)?", s):          # epoch seconds (10 digits) or milliseconds (13)
        x = float(s)
        x = x / 1000.0 if x > 1e11 else x
        try:
            return datetime.fromtimestamp(x, timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            raise ValidationFailure(f"invalid timestamp {v!r}: epoch out of range")
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except ValueError:
        raise ValidationFailure(f"invalid timestamp {v!r}: use ISO-8601 or epoch seconds/milliseconds")


def clean_rows(rows: list, last_period: Optional[str], period_exists=lambda p: False) -> list:
    """Validate a batch; returns storable items. Raises ValidationFailure/Conflict on the first problem (all-or-nothing)."""
    from app.services.errors import Conflict
    out, seen, prev = [], set(), last_period
    for i, r in enumerate(rows):
        try:
            number = clean_number(r.get("number"))
            period = clean_period(r.get("period_id"))
            ts = clean_timestamp(r.get("timestamp"))
        except ValidationFailure as e:
            raise ValidationFailure(f"row {i + 1}: {e}")
        if period:
            if period in seen or period_exists(period):
                raise Conflict(f"row {i + 1}: duplicate period id {period}")
            if prev and prev.isdigit() and int(period) <= int(prev):
                raise Conflict(f"row {i + 1}: period {period} is not after previous period {prev} (out of chronological order)")
            seen.add(period); prev = period
        out.append({"raw": {"number": r.get("number"), "period_id": r.get("period_id"), "timestamp": r.get("timestamp")},
                    "number": number, "period_id": period, "timestamp": ts})
    return out


def parse_csv(text: str, newest_first: bool = False) -> list:
    """Parse CSV text. Period ids stay strings (never converted to numbers)."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    if not reader.fieldnames:
        raise ValidationFailure("CSV has no header row")
    cols = {c.strip().lower(): c for c in reader.fieldnames}
    pick = lambda keys: next((cols[k] for k in keys if k in cols), None)
    ncol, pcol, tcol = pick(NUMBER_KEYS), pick(PERIOD_KEYS), pick(TIME_KEYS)
    if not ncol:
        raise ValidationFailure(f"CSV needs a number column (one of {NUMBER_KEYS}); found {list(cols)}")
    rows = [{"number": r[ncol], "period_id": r[pcol] if pcol else None, "timestamp": r[tcol] if tcol else None} for r in reader]
    return rows[::-1] if newest_first else rows


# Allow short numeric round IDs as well as the common long identifiers in pasted results tables.
# A plain single-digit row such as "5" should still parse, while a long ID followed by a result
# stays valid when separated by whitespace or punctuation.
_PASTE_ROW = re.compile(r"^\W*(?:(\d{1,32})\W+)?(\d)(?!\d)(.*)$")
_SCI = re.compile(r"^\W*\d+\.\d+[eE][+-]?\d+")
_GREEN, _RED = {1, 3, 5, 7, 9}, {0, 2, 4, 6, 8}


def parse_paste(text: str, newest_first: bool = False) -> list:
    """Parse rows copied from a results table, one round per line, e.g.
    '20260101100051234 5 Big green violet' or just '5'. The number must be 0-9. If Big/Small or a colour word is
    present it must agree with the number (catches typos). Period ids must be plain digits, never Excel '2.03E+16'."""
    rows = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if _SCI.match(line):
            raise ValidationFailure(f"line {i}: round id looks like scientific notation (2.03E+16). Paste it as text digits.")
        m = _PASTE_ROW.match(line)
        if not m:
            raise ValidationFailure(f"line {i}: could not find a number 0-9 in {line!r}")
        period, n, rest = m.group(1), int(m.group(2)), m.group(3).lower()
        bad = (("big" in rest and n < 5) or ("small" in rest and n >= 5) or ("green" in rest and n not in _GREEN)
               or ("red" in rest and n not in _RED) or ("viol" in rest and n not in (0, 5)))
        if bad:
            raise ValidationFailure(f"line {i}: number {n} does not match the size/colour text in {line!r}")
        rows.append({"number": n, "period_id": period, "timestamp": None})
    if not rows:
        raise ValidationFailure("nothing to import: paste one round per line")
    return rows[::-1] if newest_first else rows
