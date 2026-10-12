#!/usr/bin/env python3
# plot_pd6_comparison.py
# ═══════════════════════════════════════════════════════════════════════════════
# Generates the PD6 side-by-side comparison figure from two CSV files:
#   (a) Tabular Double Q-Learning reference
#   (b) FactorDRQN (A3) result
#
# PD6 axes:
#   x — Leader diplomatic cost (γ / xi),  range 0.0 → 3.0
#   y — Follower retaliation ceiling (ρ_max), range 0.0 → 0.50
#
# Usage:
#   python plot_pd6_comparison.py                              # default paths
#   python plot_pd6_comparison.py \
#       --tabular fig_pd6_diplomatic_cost_retaliation_tabular.csv \
#       --drqn    pd6_results_drqn.csv \
#       --out     fig_pd6_comparison.png \
#       --dpi     250
#
# CSV formats:
#   Tabular: x_val, y_val, regime, mean_lpay, x_attr, y_attr, n_seeds
#   DRQN:    gamma_diplo, rho_max, regime, mean_lpay, mean_tau, mean_rho, n_seeds
#
# Both must have 64 rows (8×8 grid), regime in {Deterrence, Transition, Escalation, Multiple}
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
TITLE_SIZE  = 6.5    # panel title
AXIS_SIZE   = 8    # axis labels
TICK_SIZE   = 7    # tick labels
LEGEND_SIZE = 8    # legend text
STAR_LABEL  = 7    # LF/FF/CENTER text
STAR_S      = 90   # star marker size

# ── Colour map — V10 IEEE spec ─────────────────────────────────────────────────
#CMAP  = ListedColormap(["#e74c3c", "#f39c12", "#27ae60", "#7f8c8d"])
#R_IDX = {"Escalation": 0, "Transition": 1, "Deterrence": 2, "Multiple": 3}
CMAP  = ListedColormap(["#e74c3c", "#f39c12", "#27ae60"])
R_IDX = {"Escalation": 0, "Transition": 1, "Deterrence": 2}
# ── Baseline star positions for PD6 (γ=0 for all — left edge) ─────────────────
STARS = [
    (0.0, 0.20, "blue",       "LF"),
    (0.0, 0.35, "darkorange", "FF"),
    (0.0, 0.27, "white",      "CENTER"),
]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_tabular(path):
    """Load tabular CSV: columns x_val, y_val, regime, mean_lpay."""
    rows = list(csv.DictReader(open(path)))
    return {(float(r["x_val"]), float(r["y_val"])): (r["regime"], float(r["mean_lpay"]))
            for r in rows}


def load_drqn(path):
    """Load DRQN CSV: columns gamma_diplo, rho_max, regime, mean_lpay."""
    rows = list(csv.DictReader(open(path)))
    return {(float(r["gamma_diplo"]), float(r["rho_max"])): (r["regime"], float(r["mean_lpay"]))
            for r in rows}


# ── Grid builder ───────────────────────────────────────────────────────────────
def make_grid(data):
    """Convert {(x,y): (regime, lpay)} dict to numpy grids for plotting."""
    x_vals = np.array(sorted(set(k[0] for k in data)))
    y_vals = np.array(sorted(set(k[1] for k in data)))
    ng     = len(x_vals)
    rg = np.full((ng, ng), R_IDX["Deterrence"], dtype=float)
    lg = np.zeros((ng, ng))
    for (x, y), (regime, lpay) in data.items():
        xi = int(np.argmin(np.abs(x_vals - x)))
        yi = int(np.argmin(np.abs(y_vals - y)))
        rg[yi, xi] = R_IDX.get(regime, 3)
        lg[yi, xi] = lpay
    return x_vals, y_vals, rg, lg


# ── Agreement metric ───────────────────────────────────────────────────────────
def compute_agreement(tab, drqn):
    """Cell-level topology agreement between tabular and DRQN results."""
    agree = 0; total = 0; disagree = []
    for key, (tab_r, _) in tab.items():
        best    = min(drqn.keys(), key=lambda k: abs(k[0]-key[0]) + abs(k[1]-key[1]))
        drqn_r  = drqn[best][0]
        if drqn_r == "Multiple": drqn_r = "Transition"
        total += 1
        if tab_r == drqn_r:
            agree += 1
        else:
            disagree.append((key[0], key[1], tab_r, drqn_r))
    pct = 100 * agree / total
    return agree, total, pct, disagree


# ── Panel renderer ─────────────────────────────────────────────────────────────
def draw_panel(ax, x_vals, y_vals, rg, lg, title, subtitle,
               show_ylabel=True):
    """Render one PD6 panel."""
    ng = len(x_vals)
    dx = (x_vals[-1] - x_vals[0]) / (ng - 1) / 2
    dy = (y_vals[-1] - y_vals[0]) / (ng - 1) / 2
    xe = np.concatenate([[x_vals[0]-dx],
                         (x_vals[:-1]+x_vals[1:])/2,
                         [x_vals[-1]+dx]])
    ye = np.concatenate([[y_vals[0]-dy],
                         (y_vals[:-1]+y_vals[1:])/2,
                         [y_vals[-1]+dy]])

    ax.pcolormesh(xe, ye, rg, cmap=CMAP, vmin=0, vmax=3,
                  alpha=0.75, shading="flat")

    # L=0 contour — line only, no label (IEEE norm)
    try:
        X, Y = np.meshgrid(x_vals, y_vals)
        ax.contour(X, Y, lg, levels=[0], colors=["black"], linewidths=1.8)
    except Exception:
        pass

    # Star markers + text labels
    _dx = (x_vals[-1] - x_vals[0]) * 0.025
    _dy = (y_vals[-1] - y_vals[0]) * 0.025
    for px, py, col, lbl in STARS:
        ax.scatter([px], [py], marker="*", s=STAR_S,
                   color=col, edgecolors="black", linewidths=0.8, zorder=5)
        ax.text(px + _dx, py + _dy, lbl, fontsize=STAR_LABEL, zorder=6)

    # x-axis padding so stars at x=0 are not clipped
    x_pad = (x_vals[-1] - x_vals[0]) * 0.03
    ax.set_xlim(x_vals[0] - x_pad, x_vals[-1])
    ax.set_ylim(y_vals[0], y_vals[-1])
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

    ax.set_xlabel(r"Leader diplomatic cost ($\xi$)", fontsize=AXIS_SIZE)
    if show_ylabel:
        ax.set_ylabel(r"Follower retaliation ceiling ($\rho_{\max}$)", fontsize=AXIS_SIZE)
    else:
        ax.tick_params(labelleft=False)

    # Panel title — no figure-level suptitle (IEEE norm: title in caption)
    ax.set_title(f"{title}\n{subtitle}", fontsize=TITLE_SIZE, fontweight="bold",
                 pad=5, multialignment="center", linespacing=1.45)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate PD6 side-by-side comparison: tabular vs FactorDRQN")
    ap.add_argument("--tabular", default="fig_pd6_diplomatic_cost_retaliation_tabular.csv",
                    help="Tabular reference CSV (x_val, y_val, regime, mean_lpay, ...)")
    ap.add_argument("--drqn",    default="pd6_results_newA3.csv",
                    help="FactorDRQN CSV (gamma_diplo, rho_max, regime, mean_lpay, ...)")
    ap.add_argument("--out",     default="fig_pd6_comparison_factordrqn.png")
    ap.add_argument("--dpi",     type=int, default=250)
    args = ap.parse_args()

    for name, path in [("tabular", args.tabular), ("drqn", args.drqn)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} CSV not found: {path}")

    # Load data
    tab  = load_tabular(args.tabular)
    drqn = load_drqn(args.drqn)

    # Compute agreement
    agree, total, pct, disagree = compute_agreement(tab, drqn)
    print(f"PD6 cell agreement: {agree}/{total} = {pct:.1f}%")
    if disagree:
        print(f"Disagreements ({len(disagree)}):")
        for g, r, t, d in sorted(disagree):
            print(f"  γ={g:.3f} ρ={r:.3f}: tabular={t:12s} drqn={d}")

    # Build grids
    xv_t, yv_t, rg_t, lg_t = make_grid(tab)
    xv_d, yv_d, rg_d, lg_d = make_grid(drqn)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.0),
                                   gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor("white")

    draw_panel(ax1, xv_t, yv_t, rg_t, lg_t,
               "(a) Tabular Double Q-Learning",
               "Reference topology",
               show_ylabel=True)

    draw_panel(ax2, xv_d, yv_d, rg_d, lg_d,
               "(b) FactorDRQN (A3)",
               f"Hybrid-factorized, GRU-bypass \u03c1-head, {pct:.1f}% agreement",
               show_ylabel=False)

    # ── Legend — frameless, bottom centre ────────────────────────────────────
    patches = [mpatches.Patch(color=c, label=l) for c, l in [
        ("#e74c3c", "Escalation"),
        ("#f39c12", "Transition"),
        ("#27ae60", "Deterrence"),
    ]]
    fig.legend(handles=patches, loc="lower center", ncol=3,
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
