"""Feature extraction. feats(h) uses ONLY the history array h (rounds strictly before the one being predicted)."""
import numpy as np
from app.utils.constants import COLOR_GROUP

FEATURE_VERSION = "1.0"


def _big(x): return int(x >= 5)
def _cg(x): return int(COLOR_GROUP[x])


def feats(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=int)
    t = len(h)
    if t < 1:
        raise ValueError("need at least 1 previous result")
    f = []
    for k in range(1, 6):                                   # previous 1..5 results
        f.append(h[-k] if t >= k else -1)
    for k in (1, 2):                                        # one-hot of previous 1 and 2 results
        oh = np.zeros(10)
        if t >= k:
            oh[h[-k]] = 1
        f += list(oh)
    for w in (5, 10, 20, 50, 100):                          # recent frequency windows
        f += list(np.bincount(h[-w:], minlength=10) / max(1, min(w, t)))
    f += list(np.bincount(h, minlength=10) / t)             # cumulative frequency
    tail = h[-101:]                                         # the gap is capped at 100, so only the last 101 rounds can matter (same values, O(1) not O(t))
    for d in range(10):                                     # gap since last appearance
        idx = np.flatnonzero(tail == d)
        f.append(min(100, len(tail) - 1 - idx[-1]) if len(idx) else 100)
    C = np.ones((10, 10))                                   # smoothed transition row for the previous result
    if t > 1:
        C += np.bincount(h[:-1] * 10 + h[1:], minlength=100).reshape(10, 10)     # vectorised count of every a->b transition
    f += list(C[h[-1]] / C[h[-1]].sum())

    def run(fn):
        r = 1
        while r < t and fn(h[-1 - r]) == fn(h[-1]):
            r += 1
        return r
    f += [run(lambda x: x), run(_big), run(_cg), run(lambda x: x % 2)]   # streak lengths
    p = h[-1]
    f += [p % 2, _big(p)]
    oh = np.zeros(3); oh[_cg(p)] = 1
    f += list(oh)
    for w in (5, 10, 20):                                   # rolling mean / std
        f += [h[-w:].mean(), h[-w:].std()]
    for w in (10, 20, 50):
        f.append(np.mean([_big(x) for x in h[-w:]]))
    for w in (10, 20):
        f.append(np.mean(h[-w:] % 2))
    f.append(len(set(h[-10:])))
    return np.array(f, dtype=float)
