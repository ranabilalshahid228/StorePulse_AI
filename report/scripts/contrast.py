"""WCAG 2.2 contrast ratios of the storm theme's text/background pairs (OKLCH tokens from globals.css).

    python report/scripts/contrast.py            # writes report/data/contrast.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "contrast.csv"

TOKENS = {  # frontend/src/app/globals.css (commit d3fadc8)
    "background": (0.155, 0.035, 266),
    "card": (0.2, 0.04, 266),
    "muted": (0.255, 0.045, 266),
    "sidebar": (0.13, 0.035, 268),
    "foreground": (0.95, 0.012, 255),
    "muted-foreground": (0.76, 0.03, 258),
    "sidebar-muted": (0.74, 0.03, 260),
    "primary": (0.8, 0.14, 232),
    "primary-foreground": (0.18, 0.05, 262),
    "warning": (0.82, 0.14, 80),
    "destructive": (0.68, 0.19, 22),
    "success": (0.78, 0.15, 160),
}
PAIRS = [
    ("foreground", "card", "body text on card"),
    ("muted-foreground", "card", "secondary text on card"),
    ("muted-foreground", "background", "secondary text on page"),
    ("muted-foreground", "muted", "secondary text on muted fill"),
    ("primary-foreground", "primary", "primary button label"),
    ("primary", "card", "link / accent text on card"),
    ("sidebar-muted", "sidebar", "inactive menu item"),
    ("warning", "card", "warning text on card"),
    ("destructive", "card", "error text on card"),
    ("success", "card", "success text on card"),
]


def oklch_to_srgb_linear(L: float, C: float, h: float) -> list[float]:
    a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    rgb = (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )
    return [min(max(x, 0.0), 1.0) for x in rgb]  # clip to the sRGB gamut (linear light)


def luminance(token: str) -> float:
    r, g, b = oklch_to_srgb_linear(*TOKENS[token])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def main() -> None:
    rows = []
    for fg, bg, use in PAIRS:
        hi, lo = sorted((luminance(fg), luminance(bg)), reverse=True)
        ratio = (hi + 0.05) / (lo + 0.05)
        rows.append({"use": use, "foreground": fg, "background": bg, "ratio": round(ratio, 2), "wcag_aa_text": ratio >= 4.5})
        print(f"{use:32s} {ratio:5.2f}")
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
