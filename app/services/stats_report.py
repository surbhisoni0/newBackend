"""Accuracy report (from the stored walk-forward test) and hot/cold numbers for recent windows."""
import numpy as np
from app.utils.constants import BASELINES, COLOR_GROUP, COLOR_NAMES


def _rate(hits) -> dict:
    n = len(hits)
    return {"n": n, "hits": int(sum(hits)), "rate": round(sum(hits) / n, 4) if n else None}


def accuracy_report(bt) -> dict:
    if not bt or not bt.get("rows"):
        return {"available": False, "message": "The walk-forward test has not finished yet."}
    rows = bt["rows"]
    size = {"all": _rate([r["bs_hit"] for r in rows]),
            "by_pick": {s: _rate([r["bs_hit"] for r in rows if r["size"] == s]) for s in ("Big", "Small")}}
    colour = {"all": _rate([r["color_hit"] for r in rows]),
              "by_pick": {c: _rate([r["color_hit"] for r in rows if str(r["color"]).lower() == c.lower()]) for c in COLOR_NAMES}}
    return {"available": True, "rounds_tested": len(rows), "first_seq": rows[0]["target_seq"], "last_seq": rows[-1]["target_seq"],
            "size": size, "colour": colour, "baselines": {"size": BASELINES["bs"], "colour": BASELINES["color"]}}


STREAK_ALERT = 5


def _run(labels) -> dict:
    n = 0
    for x in reversed(labels):
        if x != labels[-1]:
            break
        n += 1
    return {"value": labels[-1], "length": n, "alert": n >= STREAK_ALERT}


def streaks(numbers) -> dict:
    if len(numbers) == 0:
        return {"threshold": STREAK_ALERT, "size": None, "colour": None}
    nums = [int(x) for x in numbers]
    return {"threshold": STREAK_ALERT, "size": _run(["Big" if x >= 5 else "Small" for x in nums]),
            "colour": _run(["Green" if x % 2 else "Red" for x in nums])}


def daily_summary(rows_oldest_first, top=3) -> list:
    days = {}
    for r in rows_oldest_first:
        pid = str(r.get("period_id") or "")
        if len(pid) >= 8 and pid[:8].isdigit():
            days.setdefault(pid[:8], []).append(int(r["number"]))
    out = []
    for d in sorted(days, reverse=True):
        nums = np.asarray(days[d])
        counts = np.bincount(nums, minlength=10)
        order = sorted(range(10), key=lambda k: (-counts[k], k))
        cold = sorted(range(10), key=lambda k: (counts[k], k))
        cc = np.bincount(COLOR_GROUP[nums], minlength=3)
        out.append({"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "rounds": len(nums),
                    "hot": [{"number": k, "count": int(counts[k])} for k in order[:top]],
                    "cold": [{"number": k, "count": int(counts[k])} for k in cold[:top]],
                    "colours": {COLOR_NAMES[i]: int(cc[i]) for i in range(3)},
                    "size": {"Big": int((nums >= 5).sum()), "Small": int((nums < 5).sum())}})
    return out


def hot_cold(numbers, windows=(50, 100), top=3) -> dict:
    nums = np.asarray(numbers, dtype=int)
    out = {}
    for w in windows:
        if len(nums) < w:
            continue
        last = nums[-w:]
        counts = np.bincount(last, minlength=10)
        since = [int(w - 1 - np.where(last == d)[0][-1]) if (last == d).any() else None for d in range(10)]
        order = sorted(range(10), key=lambda d: (-counts[d], d))
        nums_info = [{"number": d, "count": int(counts[d]), "since": since[d]} for d in range(10)]
        cc = np.bincount(COLOR_GROUP[last], minlength=3)
        colours = sorted(({"colour": COLOR_NAMES[i], "count": int(cc[i])} for i in range(3)), key=lambda c: -c["count"])
        big = int((last >= 5).sum())
        out[str(w)] = {"n": w, "numbers": nums_info, "hot": [nums_info[d] for d in order[:top]],
                       "cold": [nums_info[d] for d in sorted(range(10), key=lambda d: (counts[d], d))[:top]], "colours": colours,
                       "size": {"Big": big, "Small": w - big}, "expected_per_number": w / 10}
    return out
