"""
Publication figure style — single source of truth for colors, fonts and output.

Palette notes
-------------
Model colors are a colorblind-validated categorical set (Machado-2009 CVD
simulation, worst all-pairs ΔE = 15.3 under protanopia/deuteranopia; every hue
≥ 3:1 WCAG contrast on white). Observation products deliberately stay on
neutral inks with distinct dash patterns, so hue always means "model" and the
truth references recede. Lead-day curves use a single-hue ordinal blue ramp
(monotone lightness, adjacent ΔL ≥ 0.06) — in a lead-day figure color encodes
lead time only; the panel label carries the model identity.

Every figure is written twice by :func:`savefig`: vector PDF (fonttype 42, for
LaTeX) and 300-dpi PNG (for the docs site).
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Categorical: model identity ───────────────────────────────────────────────

MODEL_COLORS = {
    "gencast":     "#2a78d6",   # blue
    "neuralgcm":   "#199e70",   # teal-green
    "fourcastnet": "#c98500",   # amber
    "graphcast":   "#e34948",   # red
    "climatology": "#898781",   # neutral gray — baseline, not a competing series
}
MODEL_LABELS = {
    "gencast":     "GenCast",
    "neuralgcm":   "NeuralGCM",
    "fourcastnet": "FourCastNet",
    "graphcast":   "GraphCast",
    "climatology": "Climatology",
}

# ── Observation references: neutral inks + distinct dashes ───────────────────

OBS_STYLES = {
    "CHIRPS": dict(color="#0b0b0b", ls="-"),
    "ERA5":   dict(color="#52514e", ls=(0, (4, 2))),
    "TAMSAT": dict(color="#898781", ls=(0, (1, 1.2))),
}

# ── Ordinal: lead-day ramp (one hue, light → dark) ────────────────────────────

_LEAD_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def lead_color(i, n):
    """Color for the i-th of n lead days, sampled from the ordinal blue ramp."""
    if n <= 1:
        return _LEAD_RAMP[-1]
    if n <= len(_LEAD_RAMP):
        # spread across the full ramp so the darkest step is always used
        idx = round(i * (len(_LEAD_RAMP) - 1) / (n - 1))
        return _LEAD_RAMP[idx]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("leads", _LEAD_RAMP)
    return mpl.colors.to_hex(cmap(i / (n - 1)))


# ── Ink & chrome ──────────────────────────────────────────────────────────────

INK      = "#0b0b0b"   # primary text
INK2     = "#52514e"   # secondary text
MUTED    = "#898781"   # axis/annotation ink
GRID     = "#e1e0d9"   # hairline grid
BASELINE = "#c3c2b7"   # axis spines

# dashed = "reference/threshold", never used for grids
REF_LINE = dict(color=INK2, lw=0.8, ls=(0, (4, 2)))

# ── Maps ──────────────────────────────────────────────────────────────────────

LAND_COLOR  = "#f7f6f2"
OCEAN_COLOR = "#e9eef3"
NAN_COLOR   = "#d9d9d9"
CMAP_BIAS   = "BrBG"      # dry (brown) ↔ wet (green); neutral midpoint
CMAP_SKILL  = "RdBu"      # worse (red) ↔ better (blue); neutral midpoint
CMAP_ERROR  = "Oranges"   # magnitude: one hue, light → dark
COAST_LW, BORDER_LW = 0.6, 0.35

# ── Figure geometry (inches) ──────────────────────────────────────────────────

FULL_WIDTH = 7.08   # double-column width
HALF_WIDTH = 3.46   # single-column width


def apply_style():
    """Set global rcParams for publication figures. Call once per entry point."""
    mpl.rcParams.update({
        "font.family":         "sans-serif",
        "font.size":           8,
        "axes.titlesize":      8.5,
        "axes.titleweight":    "bold",
        "axes.labelsize":      8,
        "axes.labelcolor":     INK,
        "xtick.labelsize":     7,
        "ytick.labelsize":     7,
        "xtick.color":         INK2,
        "ytick.color":         INK2,
        "xtick.major.size":    2.5,
        "ytick.major.size":    2.5,
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        "axes.edgecolor":      BASELINE,
        "axes.linewidth":      0.6,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           False,
        "grid.color":          GRID,
        "grid.linewidth":      0.5,
        "grid.linestyle":      "-",
        "lines.linewidth":     1.4,
        "legend.fontsize":     7,
        "legend.frameon":      False,
        "legend.handlelength": 1.8,
        "figure.titlesize":    9.5,
        "figure.titleweight":  "bold",
        "figure.dpi":          110,
        "savefig.dpi":         300,
        "pdf.fonttype":        42,   # embed TrueType — editable text in the PDF
        "ps.fonttype":         42,
        "figure.constrained_layout.use": True,
    })


def grid_y(ax):
    """Recessive horizontal gridlines only (values are read off the y axis)."""
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_label(ax, letter, dx=0.0):
    """Small bold panel letter '(a)' at the top-left, outside the axes frame."""
    ax.text(dx, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom", ha="left", color=INK)


def savefig(fig, out_dir, name):
    """Write <name>.pdf (vector) + <name>.png (300 dpi), close, and report."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved → {name}.pdf/.png")
