#!/usr/bin/env python3
# compare_ablation_agreement.py
# ═══════════════════════════════════════════════════════════════════════════════
# Computes cell-level topology agreement between a tabular reference CSV
# and one or more DRQN ablation CSVs.
#
# Usage:
#   # Compare all known conditions against tabular reference
#   python src/compare_ablation_agreement.py \
#       --tabular fig_pd2_retaliation_vs_tariff.csv \
#       --a0  pd2_a0_results.csv \
#       --a1  pd2_a1_results.csv \
#       --a2  pd2_a2_results.csv \
#       --a2b pd2_a2b_results.csv \
#       --a3  pd2_a3_results.csv
#
#   # Compare just new conditions
#   python src/compare_ablation_agreement.py \
#       --tabular fig_pd2_retaliation_vs_tariff.csv \
#       --a0  pd2_a0_results.csv \
#       --a2b pd2_a2b_results.csv
#
# Output: agreement table printed to stdout, disagreements listed per condition.
#
# CSV formats accepted:
#   Tabular: x_val, y_val, regime, mean_lpay, ...   (x=rho_max, y=tau_max)
#   DRQN:    rho_max, tau_max, regime, mean_lpay, ...
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


# ── Loading ───────────────────────────────────────────────────────────────────
def load_csv(path):
    """Load any PD2 CSV, auto-detecting column names."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    cols  = list(rows[0].keys())
    x_col = "rho_max" if "rho_max" in cols else "x_val"
    y_col = "tau_max" if "tau_max" in cols else "y_val"
    return {
        (float(r[x_col]), float(r[y_col])): r["regime"]
        for r in rows
    }


# ── Agreement ─────────────────────────────────────────────────────────────────
def compute_agreement(ref, pred, label):
    """
    Cell-level topology agreement between reference and prediction.
    Multiple cells treated as Transition (conservative).
    Returns (agree, total, pct, disagree_list).
    """
    agree = 0; total = 0; disagree = []
    for (rho, tau), ref_r in ref.items():
        if ref_r == "Multiple": ref_r = "Transition"
        # Nearest-neighbour match (handles rounding differences)
        best = min(pred.keys(), key=lambda k: abs(k[0]-rho)+abs(k[1]-tau))
        pr   = pred[best]
        if pr == "Multiple": pr = "Transition"
        total += 1
        if ref_r == pr:
            agree += 1
        else:
            disagree.append((tau, rho, ref_r, pr))
    pct = 100.0 * agree / total if total > 0 else 0.0
    return agree, total, pct, disagree


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Compute topology agreement between tabular reference and ablation CSVs")
    ap.add_argument("--tabular", required=True,
                    help="Tabular reference CSV")
    ap.add_argument("--a0",  default=None, help="A0 feedforward CSV")
    ap.add_argument("--a1",  default=None, help="A1 naive DRQN CSV")
    ap.add_argument("--a2",  default=None, help="A2 factorised+GRU-rho CSV")
    ap.add_argument("--a2b", default=None, help="A2b isolation ablation CSV")
    ap.add_argument("--a3",          default=None, help="A3 FactorDRQN CSV")
    ap.add_argument("--a3_no_noise", default=None, help="A3 no-noise isolation CSV")
    ap.add_argument("--a2b_no_noise",default=None, help="A2b no-noise isolation CSV")
    ap.add_argument("--verbose", action="store_true",
                    help="Print all disagreements per condition")
    args = ap.parse_args()

    # Load tabular reference
    if not Path(args.tabular).exists():
        raise FileNotFoundError(f"Tabular CSV not found: {args.tabular}")
    ref = load_csv(args.tabular)
    print(f"\nTabular reference: {args.tabular}")
    print(f"  Cells: {len(ref)}  "
          f"Regimes: {dict(Counter(ref.values()))}")

    # Collect conditions to compare
    conditions = []
    for label, path, description in [
        ("A0",  args.a0,  "Feedforward baseline — no recurrence"),
        ("A1",  args.a1,  "Naive DRQN — joint action, shared reward"),
        ("A2",  args.a2,  "Factorised DRQN — GRU→ρ-head"),
        ("A2b", args.a2b, "Isolation — tau-only input, GRU-sized ρ-head"),
        ("A3",          args.a3,          "FactorDRQN — tau-conditioned ρ-head (proposed)"),
        ("A3−(no noise)",args.a3_no_noise, "A3 architecture, τ-noise disabled"),
        ("A2b−(no noise)",args.a2b_no_noise,"A2b architecture, τ-noise disabled"),
    ]:
        if path is not None:
            if not Path(path).exists():
                print(f"  WARNING: {label} CSV not found: {path} — skipping")
                continue
            conditions.append((label, path, description))

    if not conditions:
        print("\nNo ablation CSVs provided. Use --a0, --a1, --a2, --a2b, --a3.")
        return

    # Compute agreement for each condition
    print(f"\n{'='*65}")
    print(f"{'Condition':<6}  {'Agreement':>10}  {'Cells':>7}  Description")
    print(f"{'='*65}")

    results = []
    for label, path, description in conditions:
        pred = load_csv(path)
        agree, total, pct, disagree = compute_agreement(ref, pred, label)
        results.append((label, agree, total, pct, disagree, description))
        print(f"{label:<6}  {pct:>9.1f}%  {agree:>3}/{total:<3}  {description}")

    print(f"{'='*65}")

    # Delta column
    if len(results) > 1:
        print(f"\nStep-wise gains:")
        prev_pct = None
        prev_label = None
        for label, agree, total, pct, _, _ in results:
            if prev_pct is not None:
                delta = pct - prev_pct
                sign  = "+" if delta >= 0 else ""
                print(f"  {prev_label} → {label}: {sign}{delta:.1f} pp")
            prev_pct   = pct
            prev_label = label

    # Verbose: print disagreements
    if args.verbose:
        for label, agree, total, pct, disagree, description in results:
            print(f"\n{label} disagreements ({total-agree} cells):")
            if not disagree:
                print("  None — perfect agreement")
            else:
                print(f"  {'tau':>6}  {'rho':>6}  {'reference':>12}  {'pred':>12}")
                for tau, rho, ref_r, pr in sorted(disagree):
                    print(f"  {tau:>6.3f}  {rho:>6.3f}  {ref_r:>12}  {pr:>12}")

    # Regime count comparison
    print(f"\nRegime counts:")
    print(f"  {'Condition':<10} {'Deterrence':>12} {'Transition':>12} {'Escalation':>12} {'Multiple':>10}")
    print(f"  {'Tabular':<10} ", end="")
    rc = Counter(ref.values())
    print(f"{rc.get('Deterrence',0):>12} {rc.get('Transition',0):>12} "
          f"{rc.get('Escalation',0):>12} {rc.get('Multiple',0):>10}")
    for label, path, _ in conditions:
        pred = load_csv(path)
        rc   = Counter(pred.values())
        print(f"  {label:<10} ", end="")
        print(f"{rc.get('Deterrence',0):>12} {rc.get('Transition',0):>12} "
              f"{rc.get('Escalation',0):>12} {rc.get('Multiple',0):>10}")


if __name__ == "__main__":
    main()
