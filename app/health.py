"""Health report for hosting platforms. Only a broken DATABASE makes the app unhealthy (HTTP 503); a failing data source is
reported as 'degraded' but must not make the platform restart a perfectly good server."""
from typing import Tuple


def build_health(repo, sync, poll_seconds: float) -> Tuple[dict, int]:
    db_ok, err, last = True, None, None
    try:
        last = repo.last_result()
    except Exception as e:                                             # noqa: BLE001
        db_ok, err = False, str(e)[:200]
    sync_info = None
    try:
        st = sync.status() if db_ok else None
        if st:
            sync_info = {"enabled": st["enabled"], "state": st["state"], "seconds_since_success": st["seconds_since_success"]}
    except Exception as e:                                             # noqa: BLE001
        sync_info = {"enabled": True, "state": "error", "detail": str(e)[:200]}
    degraded = bool(sync_info and sync_info["enabled"] and sync_info["state"] != "ok")
    body = {"status": "down" if not db_ok else "degraded" if degraded else "ok", "database": "ok" if db_ok else "error",
            "rounds": last["seq"] if last else 0, "sync": sync_info, "poll_seconds": poll_seconds}
    if err:
        body["error"] = err
    return body, (200 if db_ok else 503)
