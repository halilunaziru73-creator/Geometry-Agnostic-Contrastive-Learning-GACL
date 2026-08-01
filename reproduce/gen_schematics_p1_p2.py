import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size":10,"figure.dpi":150})

# --- Paper 1: GACL data/gradient flow schematic (complements existing Fig 35 / Fig A1) ---
fig, ax = plt.subplots(figsize=(9,4.2))
ax.set_xlim(0,10); ax.set_ylim(0,4.6); ax.axis("off")
boxes = [
    ("Input batch\n(images + geometry)", 0.3, 2.0, "#dfe7fd"),
    ("HGAViT\nSec. 3.3", 2.4, 2.0, "#bcd4f6"),
    ("GCATT\nSec. 3.5", 4.5, 3.2, "#a7c7e7"),
    ("DHGNN\nSec. 3.6", 4.5, 0.8, "#a7c7e7"),
    ("VLAE\nSec. 3.7", 6.6, 2.0, "#8fb8de"),
    ("Composite objective\nEq. 34, Sec. 3.8", 8.5, 2.0, "#6a9fd8"),
]
for label, x, y, c in boxes:
    ax.add_patch(FancyBboxPatch((x-0.9,y-0.5), 1.8, 1.0, boxstyle="round,pad=0.05", fc=c, ec="black"))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5)
arrows = [(1.2,2.0,1.5,2.0),(3.3,2.3,3.6,3.2),(3.3,1.7,3.6,0.8),
          (5.4,3.2,5.7,2.3),(5.4,0.8,5.7,1.7),(7.5,2.0,7.6,2.0)]
for x1,y1,x2,y2 in arrows:
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=14, color="black"))
ax.text(5.0, 4.2, "Shared encoder parameters θ (gradient flow, Sec. 3.9's extended derivation)", ha="center", fontsize=8, style="italic")
plt.title("Schematic: information flow and gradient path through the GACL composite objective\n(architectural diagram only \u2014 not a measured computation graph)")
plt.tight_layout(); plt.savefig("figs/p1_new_gradflow_schematic.png"); plt.close()

# --- Paper 2: two-interface software architecture schematic ---
fig, ax = plt.subplots(figsize=(9,5))
ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.4,4.2),3.0,1.2, boxstyle="round,pad=0.06", fc="#ffe8b3", ec="black"))
ax.text(1.9,4.8,"GIS Workbench\n(PyQt6, main.py)", ha="center", va="center", fontsize=9)
ax.add_patch(FancyBboxPatch((6.6,4.2),3.0,1.2, boxstyle="round,pad=0.06", fc="#ffe8b3", ec="black"))
ax.text(8.1,4.8,"Classic UI\n(Tkinter, main_classic.py)", ha="center", va="center", fontsize=9)
ax.add_patch(FancyBboxPatch((2.0,2.2),6.0,1.4, boxstyle="round,pad=0.06", fc="#c9e4ca", ec="black"))
ax.text(5.0,2.9,"Shared core/ analysis package\n(14 modules: vegetation, texture, spectral, alignment,\nembedding_metrics, ml_classifier, field_data, ...)", ha="center", va="center", fontsize=8.2)
ax.add_patch(FancyBboxPatch((0.6,0.3),8.8,1.3, boxstyle="round,pad=0.06", fc="#f6c6c6", ec="black"))
ax.text(5.0,0.95,"Verifiable Reporting Framework (VRF): every reported figure/metric is tagged\nREAL (computed on the user's own image) or SYNTHETIC (demonstration/benchmark data)", ha="center", va="center", fontsize=8.2)
for x1,y1,x2,y2 in [(1.9,4.2,4.2,3.6),(8.1,4.2,5.8,3.6),(5.0,2.2,5.0,1.6)]:
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=14, color="black"))
plt.title("Schematic: N_GACL two-interface software architecture over one shared analysis core")
plt.tight_layout(); plt.savefig("figs/p2_new_arch_schematic.png"); plt.close()
print("schematics done")
