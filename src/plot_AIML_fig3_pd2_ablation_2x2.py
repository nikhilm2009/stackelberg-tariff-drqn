#!/usr/bin/env python3
# plot_pd2_ablation_2x2.py
# ═══════════════════════════════════════════════════════════════════════════════
# Generates the 2×2 ablation centerpiece figure from four CSV files.
#
# Panels:
#   (a) Tabular Double Q-Learning  — reference topology
#   (b) Joint DRQN (A1)            — shared recurrent representation
#   (c) Recurrent-Factorized DRQN (A2) — factorized heads, shared GRU
#   (d) FactorDRQN (proposed)      — GRU-bypass ρ-head, 128-unit, no noise
#
# Panel (d) uses A2b_no_noise CSV (pd2_a2b_no_noise_results.csv) —
# the proposed FactorDRQN: τ-only input, 128-unit ρ-head, no τ-noise.
#
# Usage:
#   python plot_AIML_fig3_pd2_ablation_2x2.py               # uses default paths
#   python plot_AIML_fig3_pd2_ablation_2x2.py \
#       --tabular  fig_pd2_retaliation_vs_tariff.csv \
#       --a1       pd2_a1_results.csv \
#       --a2       pd2_a2_results.csv \
#       --a3       pd2_a2b_no_noise_results.csv \
#       --out      fig_pd2_ablation_2x2.png \
#       --dpi      250
#
# CSV format (all four files):
#   rho_max, tau_max, regime, mean_lpay, mean_tau, mean_rho, n_seeds
#   regime values: Deterrence | Transition | Escalation | Multiple
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

# ── Colour map — V10 IEEE spec ─────────────────────────────────────────────────
CMAP  = ListedColormap(["#e74c3c", "#f39c12", "#27ae60", "#7f8c8d"])
R_IDX = {"Escalation": 0, "Transition": 1, "Deterrence": 2, "Multiple": 3}

# ── Baseline star positions ────────────────────────────────────────────────────
STARS = [
    (0.20, 0.35, "blue",       "LF"),
    (0.35, 0.25, "darkorange", "FF"),
    (0.27, 0.28, "white",      "CENTER"),
]


# ── Panel renderer ─────────────────────────────────────────────────────────────
def make_panel(ax, csv_path, label, subtitle, agreement,
               show_xlabel=True, show_ylabel=True):
    """Render one phase-diagram panel from a CSV file."""
    rows   = list(csv.DictReader(open(csv_path)))
    # Handle both column naming conventions:
    #   DRQN CSVs:    rho_max, tau_max
    #   Tabular CSVs: x_val, y_val  (x=rho_max, y=tau_max)
    cols   = list(rows[0].keys())
    x_col  = "rho_max" if "rho_max" in cols else "x_val"
    y_col  = "tau_max" if "tau_max" in cols else "y_val"
    x_vals = np.array(sorted(set(float(r[x_col]) for r in rows)))
    y_vals = np.array(sorted(set(float(r[y_col]) for r in rows)))
    ng     = len(x_vals)

    rg = np.full((ng, ng), R_IDX["Deterrence"], dtype=float)
    lg = np.zeros((ng, ng))
    for r in rows:
        xi = int(np.argmin(np.abs(x_vals - float(r[x_col]))))
        yi = int(np.argmin(np.abs(y_vals - float(r[y_col]))))
        rg[yi, xi] = R_IDX.get(r["regime"], 3)
        lg[yi, xi] = float(r["mean_lpay"])

    # Cell-edge computation (centred boundaries)
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

    # L=0 contour — line only, no label
    try:
        X, Y = np.meshgrid(x_vals, y_vals)
        ax.contour(X, Y, lg, levels=[0], colors=["black"], linewidths=1.5)
    except Exception:
        pass

    # Star markers + text labels
    _dx = (x_vals[-1] - x_vals[0]) * 0.02
    _dy = (y_vals[-1] - y_vals[0]) * 0.02
    for px, py, col, lbl in STARS:
        ax.scatter([px], [py], marker="*", s=100,
                   color=col, edgecolors="black", linewidths=0.7, zorder=5)
        ax.text(px + _dx, py + _dy, lbl, fontsize=6.5, zorder=6)

    # Axis limits and ticks
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(y_vals[0], y_vals[-1])
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.2)

    # Axis labels — only where needed (reduces clutter in 2×2)
    if show_xlabel:
        ax.set_xlabel(r"Follower retaliation ceiling ($\rho_{\max}$)", fontsize=9)
    else:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(r"Leader tariff ceiling ($\tau_{\max}$)", fontsize=9)
    else:
        ax.tick_params(labelleft=False)

    # Panel title: bold label + subtitle + agreement on separate lines
    title_str = f"{label}\n{subtitle}"
    if agreement.strip():
        title_str += f"\n{agreement}"
    ax.set_title(title_str, fontsize=8.5, fontweight="bold",
                 pad=6, multialignment="center", linespacing=1.45)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate 2×2 ablation phase-diagram figure")
    ap.add_argument("--tabular", default="fig_pd2_retaliation_vs_tariff.csv",
                    help="CSV for panel (a): tabular reference")
    ap.add_argument("--a1",      default="pd2_a1_results.csv",
                    help="CSV for panel (b): naive DRQN A1")
    ap.add_argument("--a2",      default="pd2_a2_results.csv",
                    help="CSV for panel (c): factorised A2")
    ap.add_argument("--a3",      default="pd2_a2b_no_noise_results.csv",
                    help="CSV for panel (d): FactorDRQN (proposed, A2b_no_noise)")
    ap.add_argument("--out",     default="fig_pd2_ablation_2x2_factordrqn.png",
                    help="Output PNG path")
    ap.add_argument("--dpi",     type=int, default=250)
    args = ap.parse_args()

    # Verify inputs exist
    for name, path in [("tabular", args.tabular), ("A1", args.a1),
                       ("A2", args.a2), ("A3", args.a3)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} CSV not found: {path}")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5),
                             gridspec_kw={"hspace": 0.22, "wspace": 0.10})
    fig.patch.set_facecolor("white")

    # (a) Tabular reference — top-left
    make_panel(axes[0, 0], args.tabular,
               "(a) Tabular Double Q-Learning",
               "Reference topology",
               "",
               show_xlabel=False, show_ylabel=True)

    # (b) Joint DRQN (A1) — top-right
    make_panel(axes[0, 1], args.a1,
               "(b) Joint DRQN (A1)",
               "Shared recurrent representation, 65.6% agreement",
               "",
               show_xlabel=False, show_ylabel=False)

    # (c) Recurrent-Factorized DRQN (A2) — bottom-left
    make_panel(axes[1, 0], args.a2,
               "(c) Recurrent-Factorized DRQN (A2)",
               "Factorized heads, shared GRU, 87.5% agreement",
               "",
               show_xlabel=True, show_ylabel=True)

    # (d) FactorDRQN — proposed architecture (A2b_no_noise)
    make_panel(axes[1, 1], args.a3,
               "(d) FactorDRQN (A3)",
               "Hybrid-factorized, GRU-bypass ρ-head, 98.4% agreement",
               "",
               show_xlabel=True, show_ylabel=False)

    # ── Legend — frameless, bottom centre ────────────────────────────────────
    patches = [mpatches.Patch(color=c, label=l) for c, l in [
        ("#e74c3c", "Escalation"),
        ("#f39c12", "Transition"),
        ("#27ae60", "Deterrence"),
        #("#7f8c8d", "Multiple"),
    ]]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0),
               handlelength=1.4, columnspacing=1.4, handletextpad=0.5)

    # ── Supertitle ────────────────────────────────────────────────────────────
    #fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.97)
    # fig.suptitle(
    #     "Topology Recovery Under Recurrent MARL\n"
    #     r"Retaliation Capacity ($\rho_{\max}$) vs Tariff Capacity ($\tau_{\max}$)",
    #     fontsize=12, fontweight="bold", y=0.99)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.out)
    plt.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
