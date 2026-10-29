"""Game rules and shared constants. Big/Small and Color are deterministic functions of the number."""
import numpy as np

DIGITS = list(range(10))
BASELINES = {"top1": 0.10, "top3": 0.30, "bs": 0.50, "color": 0.40}  # color: best constant guess (green or red)
METRICS = ("top1", "top3", "bs", "color")

# colour groups: 0 green(1,3,7,9)  1 red(2,4,6,8)  2 violet(0 = red+violet, 5 = green+violet)
COLOR_GROUP = np.array([2, 0, 1, 0, 1, 2, 1, 0, 1, 0])
COLOR_NAMES = ["Green", "Red", "Violet"]
COLOR_DETAIL = {0: "red+violet", 1: "green", 2: "red", 3: "green", 4: "red",
                5: "green+violet", 6: "red", 7: "green", 8: "red", 9: "green"}


def size_of(n: int) -> str:
    return "Big" if n >= 5 else "Small"


def color_group_of(n: int) -> int:
    return int(COLOR_GROUP[n])
