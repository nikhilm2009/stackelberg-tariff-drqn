#!/usr/bin/env python3
# plot_AIML_fig_reference_vs_a0.py
# ═══════════════════════════════════════════════════════════════════════════════
# Generates the reference vs feedforward baseline (A0) comparison figure.
#
# Panels:
#   (a) Tabular Double Q-Learning — reference topology
#   (b) Feedforward DQN           — no recurrence, 46.9% agreement
#
# PD2 axes:
#   x — Follower retaliation ceiling (ρ_max), range 0.0 → 0.50
#   y — Leader tariff ceiling (τ_max),        range 0.10 → 0.45
#
# Usage:
#   python plot_AIML_fig_reference_vs_a0.py                     # default paths
#   python plot_AIML_fig_reference_vs_a0.py \
#       --tabular fig_pd2_retaliation_vs_tariff.csv \
#       --a0      pd2_a0_results.csv \
#       --out     fig_pd2_reference_vs_a0.png \
#       --dpi     250
#
# CSV formats:
#   Tabular: x_val, y_val, regime, mean_lpay, x_attr, y_attr, n_seeds
#   A0:      rho_max, tau_max, regime, mean_lpay, mean_tau, mean_rho, n_seeds
#
# Matches styling of plot_AIML_fig4_pd6_comparison.py exactly:
#   dpi=250, wspace=0.06, frameless legend, no suptitle (IEEE norm)
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ── Font sizes — scaled for single-column IEEE print ─────────────────────────
TITLE_SIZE  = 8    # panel title
AXIS_SIZE   = 8    # axis labels
TICK_SIZE   = 7    # tick labels
LEGEND_SIZE = 8    # legend text
STAR_LABEL  = 7    # LF/FF/CENTER text
STAR_S      = 90   # star marker size

# ── Colour map — V10 IEEE spec ─────────────────────────────────────────────────
CMAP  = ListedColormap(["#e74c3c", "#f39c12", "#27ae60", "#7f8c8d"])
R_IDX = {"Escalation": 0, "Transition": 1, "Deterrence": 2, "Multiple": 3}

# ── Star positions — PD2 calibration anchors ──────────────────────────────────
STARS = [
    (0.20, 0.35, "blue",       "LF"),
    (0.35, 0.25, "darkorange", "FF"),
    (0.27, 0.28, "white",      "CENTER"),
]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_csv(path):
    """Load PD2 CSV — auto-detects column names for tabular and DRQN formats."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    cols  = list(rows[0].keys())
    x_col = "rho_max" if "rho_max" in cols else "x_val"
    y_col = "tau_max" if "tau_max" in cols else "y_val"
    return {
        (float(r[x_col]), float(r[y_col])): (r["regime"], float(r["mean_lpay"]))
        for r in rows
    }


# ── Grid builder ───────────────────────────────────────────────────────────────
def make_grid(data):
    x_vals = np.array(sorted(set(k[0] for k in data)))
    y_vals = np.array(sorted(set(k[1] for k in data)))
    ng     = len(x_vals)
    rg     = np.full((ng, ng), R_IDX["Deterrence"], dtype=float)
    lg     = np.zeros((ng, ng))
    for (x, y), (regime, lpay) in data.items():
        xi = int(np.argmin(np.abs(x_vals - x)))
        yi = int(np.argmin(np.abs(y_vals - y)))
        rg[yi, xi] = R_IDX.get(regime, 3)
        lg[yi, xi] = lpay
    return x_vals, y_vals, rg, lg


# ── Panel renderer ─────────────────────────────────────────────────────────────
def draw_panel(ax, data, title_l1, title_l2,
               show_xlabel=True, show_ylabel=True):
    """Render one PD2 phase-diagram panel."""
    x_vals, y_vals, rg, lg = make_grid(data)
    ng = len(x_vals)
    dx = (x_vals[-1] - x_vals[0]) / (ng - 1) / 2
    dy = (y_vals[-1] - y_vals[0]) / (ng - 1) / 2
    xe = np.concatenate([[x_vals[0]-dx],
                         (x_vals[:-1]+x_vals[1:])/2,
                         [x_vals[-1]+dx]])
    ye = np.concatenate([[y_vals[0]-dy],
                         (y_vals[:-1]+y_vals[1:])/2,
                         [y_vals[-1]+dy]])

    # Regime background
    ax.pcolormesh(xe, ye, rg, cmap=CMAP, vmin=0, vmax=3,
                  alpha=0.75, shading="flat")

    # L=0 contour
    try:
        X, Y = np.meshgrid(x_vals, y_vals)
        ax.contour(X, Y, lg, levels=[0], colors=["black"], linewidths=1.8)
    except Exception:
        pass

    # Star markers
    _dx = (x_vals[-1] - x_vals[0]) * 0.025
    _dy = (y_vals[-1] - y_vals[0]) * 0.025
    for px, py, col, lbl in STARS:
        ax.scatter([px], [py], marker="*", s=STAR_S,
                   color=col, edgecolors="black", linewidths=0.8, zorder=5)
        ax.text(px + _dx, py + _dy, lbl, fontsize=STAR_LABEL, zorder=6)

    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(y_vals[0], y_vals[-1])
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

    if show_xlabel:
        ax.set_xlabel(r"Follower retaliation ceiling ($\rho_{\max}$)", fontsize=AXIS_SIZE)
    else:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(r"Leader tariff ceiling ($\tau_{\max}$)", fontsize=AXIS_SIZE)
    else:
        ax.tick_params(labelleft=False)

    # Two-line panel title — matches Figure 2 style exactly
    ax.set_title(f"{title_l1}\n{title_l2}",
                 fontsize=TITLE_SIZE, fontweight="bold",
                 pad=5, multialignment="center", linespacing=1.45)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate reference vs feedforward baseline (A0) figure")
    ap.add_argument("--tabular", default="fig_pd2_retaliation_vs_tariff.csv",
                    help="Tabular reference CSV")
    ap.add_argument("--a0",     default="pd2_a0_results.csv",
                    help="Feedforward baseline (A0) CSV")
    ap.add_argument("--out",    default="fig_pd2_reference_vs_a0.png",
                    help="Output PNG path")
    ap.add_argument("--dpi",    type=int, default=250)
    args = ap.parse_args()

    for name, path in [("tabular", args.tabular), ("A0", args.a0)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} CSV not found: {path}")

    tab = load_csv(args.tabular)
    a0  = load_csv(args.a0)

    # ── Figure — matches Fig 4 layout exactly ────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.0),
                                   gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor("white")

    draw_panel(ax1, tab,
               "(a) Tabular Double Q-Learning",
               "Reference topology",
               show_xlabel=True, show_ylabel=True)

    draw_panel(ax2, a0,
               "(b) Feedforward DQN",
               "No recurrence,  48.4% agreement",
               show_xlabel=True, show_ylabel=False)

    # ── Legend — frameless, 4 items including Multiple ────────────────────────
    patches = [mpatches.Patch(color=c, label=l) for c, l in [
        ("#e74c3c", "Escalation"),
        ("#f39c12", "Transition"),
        ("#27ae60", "Deterrence"),
        #("#7f8c8d", "Multiple"),
    ]]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=AXIS_SIZE, frameon=False,
               bbox_to_anchor=(0.5, 0.0),
               handlelength=1.4, columnspacing=1.8, handletextpad=0.5)

    # No suptitle — IEEE norm, title goes in figure caption
    plt.subplots_adjust(top=0.90, bottom=0.18, left=0.09, right=0.97)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.out)
    plt.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
