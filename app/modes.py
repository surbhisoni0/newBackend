"""Game modes. Durations and feed TypeIds were checked against the live feed (intervalM, endTime - startTime)."""
from urllib.parse import urlsplit, urlunsplit

MODES = {
    "30s": {"label": "WinGo 30s", "seconds": 30, "type_id": 30},
    "1min": {"label": "WinGo 1Min", "seconds": 60, "type_id": 1},
    "3min": {"label": "WinGo 3Min", "seconds": 180, "type_id": 10},
    "5min": {"label": "WinGo 5Min", "seconds": 300, "type_id": 3},
}
DEFAULT_MODE = "1min"


def mode_db_url(url: str, mode: str) -> str:
    """1min keeps the original database (holds the Data.docx history); other modes get <db>_<mode>."""
    if mode == DEFAULT_MODE or not url.startswith("mongodb"):
        return url
    p = urlsplit(url)
    db = p.path.strip("/") or "wingo"
    return urlunsplit((p.scheme, p.netloc, f"/{db}_{mode}", p.query, p.fragment))
