"""Convert Data.docx rounds to a clean CSV. Times are ESTIMATED: newest round ends at the last full minute, older rounds count back 60s per period id."""
import csv, re, sys, time, zipfile
from datetime import datetime, timezone

GREEN, RED = {1, 3, 5, 7, 9}, {0, 2, 4, 6, 8}
src, dst = sys.argv[1], sys.argv[2]
xml = zipfile.ZipFile(src).read("word/document.xml").decode()
text = "\n".join("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p)) for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S))
rows = {}
for pid, n in re.findall(r"(\d{16,17}),(\d)(?!\d)", text):
    if len(pid) == 16 and pid.startswith("026"):
        pid = "2" + pid                                               # one id lost its leading '2' in the document
    rows[pid] = int(n)
ids = sorted(rows, key=int, reverse=True)
end = (int(time.time()) // 60) * 60
out = []
for i, pid in enumerate(ids):
    if i:
        prev = ids[i - 1]
        same_day = prev[:13] == pid[:13]
        end -= 60 * (int(prev) - int(pid) if same_day else 1)
    n = rows[pid]
    color = ("green" if n in GREEN else "red") + ("+violet" if n in (0, 5) else "")
    out.append([pid, n, "Big" if n >= 5 else "Small", color, datetime.fromtimestamp(end, timezone.utc).isoformat().replace("+00:00", "Z")])
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["period_id", "number", "size", "color", "timestamp"])
    w.writerows(reversed(out))                                        # oldest first
print(f"{len(out)} rounds -> {dst}")
