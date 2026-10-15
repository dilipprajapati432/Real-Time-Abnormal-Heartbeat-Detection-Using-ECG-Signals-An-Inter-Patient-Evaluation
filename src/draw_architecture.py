"""
Draws the system block diagram (Fig. 1) in a plain IEEE style:
white boxes, thin black lines, serif font, short labels.

Output: models/figures/fig0_system_architecture.png (600 dpi) and .pdf (vector).
Sized for a full-width figure in a two-column IEEE paper (7.16 in).
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from config import FIG_DIR

os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

W, H = 18.0, 6.4
fig, ax = plt.subplots(figsize=(7.16, 7.16 * H / W))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

FS = 7.5          # label font size (pt)
LW = 0.7          # line width
BW, BH = 2.7, 1.25


def block(xc, yc, text, shaded=False):
    ax.add_patch(Rectangle((xc - BW / 2, yc - BH / 2), BW, BH, fill=True,
                           fc="#e8e8e8" if shaded else "white", ec="black", lw=LW))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=FS, linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=LW, color="black", shrinkA=0, shrinkB=0))


def group(x0, y0, x1, y1, label):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black",
                           lw=0.5, ls=(0, (4, 2.5))))
    ax.text(x0 + 0.15, y1 - 0.12, label, ha="left", va="top", fontsize=FS, style="italic")


xs = [1.75, 5.3, 8.85, 12.4, 15.95]

# (a) offline training
y_top = 4.75
group(0.15, 3.85, 17.85, 6.25, "(a) Offline training and evaluation")
block(xs[0], y_top, "MIT-BIH\nArrhythmia DB")
block(xs[1], y_top, "Beat selection,\nAAMI labels")
block(xs[2], y_top, "Feature\nextraction")
block(xs[3], y_top, "Model training\n(DS1, 5-fold CV)")
block(xs[4], y_top, "Trained model\n(scaler + LR)", shaded=True)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + BW / 2, y_top, b - BW / 2, y_top)
ax.text(xs[4], y_top + BH / 2 + 0.22, "tested on DS2", ha="center", va="bottom",
        fontsize=FS - 0.5, style="italic")

# (b) online monitoring
y_bot = 1.05
group(0.15, 0.15, 17.85, 2.55, "(b) Online monitoring")
block(xs[0], y_bot, "ECG signal\n(360 Hz)")
block(xs[1], y_bot, "QRS detection\n(Pan–Tompkins)")
block(xs[2], y_bot, "Feature\nextraction")
block(xs[3], y_bot, "Classification\n(REST API)")
block(xs[4], y_bot, "Dashboard\nand alerts")
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + BW / 2, y_bot, b - BW / 2, y_bot)

# trained model used by the classifier
ax.plot([xs[4], xs[4], xs[3]], [y_top - BH / 2, 3.2, 3.2], color="black", lw=LW)
arrow(xs[3], 3.2, xs[3], y_bot + BH / 2)

for ext, kw in (("png", {"dpi": 600}), ("pdf", {})):
    plt.savefig(os.path.join(FIG_DIR, f"fig0_system_architecture.{ext}"),
                bbox_inches="tight", pad_inches=0.02, **kw)
print("Saved models/figures/fig0_system_architecture.png and .pdf")
