"""Crop focused regions out of the full-page screenshots in report/figures/screens (originals kept).

    cd backend
    .venv\\Scripts\\python ..\\report\\scripts\\crop_screens.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

SCREENS = Path(__file__).resolve().parents[1] / "figures" / "screens"
# name: (source, (left, top, right, bottom)) in source pixels
CROPS = {
    "eda_health": ("data_eda.png", (240, 600, 1425, 1150)),
    "eda_explore": ("data_eda.png", (240, 1330, 1425, 1890)),
    "autopilot_plan": ("autopilot.png", (0, 360, 1185, 1240)),
    "model_lab_compare": ("model_lab.png", (272, 455, 1410, 990)),
}


def main() -> None:
    for name, (src, box) in CROPS.items():
        im = Image.open(SCREENS / src)
        im.crop(box).save(SCREENS / f"{name}.png", optimize=True)
        print("cropped", name, box)


if __name__ == "__main__":
    main()
