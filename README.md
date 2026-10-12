# TariffWar-DRQN
## FactorDRQN: Recurrent Multi-Agent Reinforcement Learning for a Stackelberg Tariff Conflict

This repository implements the FactorDRQN architecture studied in:

> **"Recovering Strategic Phase Structure in Recurrent Multi-Agent Socio-Economic Systems"**
> Nikhil Muthukumar, Micah Chiang, Jeffrey Chen, Phil Mui
> ASDRP, Fremont CA
> *Preprint — under review*

The project studies whether recurrent multi-agent systems can recover the qualitative phase structure of a two-country Stackelberg tariff conflict under function approximation.

---

## Architecture

Two agents interact in a leader-follower game:

**Leader** — sets tariff τ ∈ {0, …, τ\_max} over 6 bins using a GRU-based DRQN.

**Follower (FactorDRQN)** — responds with depreciation d and retaliation ρ using factorised Q-heads:
- **Q\_d head**: full GRU trunk → dueling head (6 depreciation bins)
- **Q\_ρ head**: τ/τ\_max only → stateless 128-unit MLP → dueling head (5 retaliation bins)

The key architectural contribution is the **τ-conditional retaliation pathway** — the ρ-head bypasses the GRU entirely, conditioning only on current tariff provocation. This prevents economic-flow updates from overwriting the sparse retaliation signal (FM3). No τ-noise regulariser is needed at 128-unit capacity.

---

## Repository Structure

```
tariffwar-drqn/
├── src/
│   ├── recurrent_q_net_v5.py               # RecurrentQNet (leader) + FactorisedFollowerQNet
│   │                                        # + FeedforwardFollowerQNet (A0 baseline)
│   ├── drqn_agent_v5.py                    # DRQNAgent + FactorisedFollowerAgent
│   │                                        # + FeedforwardFollowerAgent (A0 baseline)
│   ├── stackelberg_q3_tariff_econ_sim_v5.py  # Environment (payoff-first, beta_E, gamma_diplo)
│   ├── seq_replay.py                        # Sequence replay buffer
│   │
│   ├── train_drqn_v5.py                    # Single LF training run
│   ├── train_drqn_PD.py                    # Unified phase diagram runner (PD2 + PD6)
│   ├── train_drqn_pd2_ablation.py          # Ablation runner — A1 / A2 / A3 (paper conditions)
│   ├── train_drqn_pd2_ablation_v2.py       # Ablation runner — A0 / A2b / A2b_no_noise (reviewer)
│   │
│   ├── plot_AIML_fig3_pd2_ablation_2x2.py  # 2×2 ablation centerpiece figure
│   ├── plot_AIML_fig4_pd6_comparison.py    # PD6 side-by-side comparison figure
│   ├── plot_AIML_fig_reference_vs_a0.py    # Reference vs feedforward baseline figure
│   ├── compare_ablation_agreement.py       # CLI: compute topology agreement across conditions
│   ├── plot_drqn_v5.py                     # Training diagnostics (5-figure suite)
│   └── probe_follower_v5.py                # Load checkpoint, probe ρ-rate per τ bin
│
├── models_v5/                             # Training checkpoints and CSV logs
├── pd_results/                             # Phase diagram CSVs and figures
├── pd2_ablation_results/                   # Ablation study CSVs and figures
└── README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib pandas
```

---

## Training

### Single training run (LF calibration)
```bash
python src/train_drqn_v5.py
```

### Phase diagram sweep — PD2 and PD6
```bash
# FactorDRQN (proposed): tau_only input, 128-unit rho-head, no noise
python src/train_drqn_PD.py --pds PD2 PD6 --seeds 10 --workers 6 \
    --rho_input_mode tau_gru_size

# PD6 only (robustness axis)
python src/train_drqn_PD.py --pds PD6 --seeds 10 --workers 6 \
    --rho_input_mode tau_gru_size
```

### Ablation study — paper conditions (A1 / A2 / A3)
```bash
# All three conditions
python src/train_drqn_pd2_ablation.py --ablation all --seeds 10 --workers 6

# Single condition
python src/train_drqn_pd2_ablation.py --ablation A2 --seeds 10 --workers 6

# Fast sanity check (4×4 grid, 1 seed, 200 episodes)
python src/train_drqn_pd2_ablation.py --ablation A3 --fast
```

### Reviewer ablations (A0 / A2b / A2b_no_noise)
```bash
# All reviewer conditions
python src/train_drqn_pd2_ablation_v2.py --ablation all --seeds 10 --workers 6

# Feedforward baseline only (A0)
python src/train_drqn_pd2_ablation_v2.py --ablation A0 --seeds 10 --workers 6

# Isolation ablation only (A2b_no_noise = FactorDRQN proposed)
python src/train_drqn_pd2_ablation_v2.py --ablation A2b_no_noise --seeds 10 --workers 6
```

---

## Plotting

### 2×2 ablation centerpiece (Figure 3)
```bash
python src/plot_AIML_fig3_pd2_ablation_2x2.py \
    --tabular fig_pd2_retaliation_vs_tariff.csv \
    --a1      pd2_a1_results.csv \
    --a2      pd2_a2_results.csv \
    --a3      pd2_a2b_no_noise_results.csv \
    --out     fig_pd2_ablation_2x2.png
```

### Reference vs feedforward baseline (Figure 2)
```bash
python src/plot_AIML_fig_reference_vs_a0.py \
    --tabular fig_pd2_retaliation_vs_tariff.csv \
    --a0      pd2_a0_results.csv \
    --out     fig_pd2_reference_vs_a0.png
```

### PD6 robustness comparison (Figure 4)
```bash
python src/plot_AIML_fig4_pd6_comparison.py \
    --tabular fig_pd6_diplomatic_cost_retaliation.csv \
    --drqn    pd6_results_newA3.csv \
    --out     fig_pd6_comparison.png
```

### Agreement comparison (all conditions)
```bash
# Tabular reference CSV generated separately (see companion tabular repository)
python src/compare_ablation_agreement.py \
    --tabular  fig_pd2_retaliation_vs_tariff.csv \
    --a0       pd2_a0_results.csv \
    --a1       pd2_a1_results.csv \
    --a2       pd2_a2_results.csv \
    --a2b      pd2_a2b_results.csv \
    --a3       pd2_a3_results.csv \
    --a2b_no_noise pd2_a2b_no_noise_results.csv \
    --verbose
```

---

## Pre-computed Results

Pre-computed CSVs and figures are included so the paper figures can be reproduced without rerunning training.

```
pd_results/
├── fig_pd6_drqn_diplomatic_cost_retaliation.png  # PD6 FactorDRQN phase diagram
├── pd6_results.csv                                # PD6 10-seed results (FactorDRQN)
└── tabular_pd6_diplomatic_cost_retaliation.csv    # PD6 tabular reference (from companion repo)

pd2_ablation_results/
├── fig_pd2_ablation_A0_feedforward.png            # A0 feedforward baseline
├── fig_pd2_ablation_A1_naive.png                  # A1 Joint DRQN
├── fig_pd2_ablation_A2_factorised_gru_rho.png     # A2 Recurrent-Factorized
├── fig_pd2_ablation_A2b_no_noise.png              # FactorDRQN (proposed)
├── pd2_a0_results.csv                             # A0 10-seed results
├── pd2_a1_results.csv                             # A1 10-seed results
├── pd2_a2_results.csv                             # A2 10-seed results
├── pd2_a2b_no_noise_results.csv                   # FactorDRQN 10-seed results
└── tabular_pd2_retaliation_vs_tariff.csv          # Tabular reference (from companion repo)
```

To regenerate the paper figures from these CSVs:

```bash
# Figure 3 — 2×2 ablation centerpiece
python src/plot_AIML_fig3_pd2_ablation_2x2.py \
    --tabular pd2_ablation_results/tabular_pd2_retaliation_vs_tariff.csv \
    --a1      pd2_ablation_results/pd2_a1_results.csv \
    --a2      pd2_ablation_results/pd2_a2_results.csv \
    --a3      pd2_ablation_results/pd2_a2b_no_noise_results.csv \
    --out     fig_pd2_ablation_2x2.png

# Figure 2 — reference vs feedforward baseline
python src/plot_AIML_fig_reference_vs_a0.py \
    --tabular pd2_ablation_results/tabular_pd2_retaliation_vs_tariff.csv \
    --a0      pd2_ablation_results/pd2_a0_results.csv \
    --out     fig_pd2_reference_vs_a0.png

# Figure 4 — PD6 robustness
python src/plot_AIML_fig4_pd6_comparison.py \
    --tabular pd_results/tabular_pd6_diplomatic_cost_retaliation.csv \
    --drqn    pd_results/pd6_results.csv \
    --out     fig_pd6_comparison.png
```

---

## Key Results

All numbers computed under the **payoff-primary rule**: majority vote across 10 seeds; ties broken by mean leader payoff threshold (Deterrence > +15, Escalation < −15, Transition between). Applied identically to tabular reference and all DRQN architectures.

### PD2 — Retaliation vs Tariff Capacity (ρ\_max × τ\_max, 8×8 grid)

| Condition | Architecture | Topology Agreement |
|---|---|---|
| A0 — Feedforward | No recurrence, joint action space | 48.4% |
| A1 — Joint DRQN | Joint 30-action space, shared reward | 65.6% |
| A2 — Recurrent-Factorized DRQN | Factorized heads, shared GRU | 87.5% |
| **FactorDRQN (proposed)** | τ-conditioned ρ-head, 128-unit, no noise | **98.4%** |

### PD6 — Diplomatic Cost vs Retaliation (ξ × ρ\_max, 8×8 grid)

| Condition | Topology Agreement |
|---|---|
| FactorDRQN (proposed) | **93.8%** |

Topology recovery generalises across the diplomatic cost perturbation axis (ξ ∈ [0, 3.0]).

### Return Table

| Condition | Mean L return | Mean F return | Mean ρ̄ | Topology |
|---|---|---|---|---|
| A0 Feedforward | −15.5 | +44.2 | 0.24 | 48.4% |
| A1 Joint DRQN | +7.2 | +42.1 | 0.16 | 65.6% |
| A2 Recurrent-Factorized | +9.6 | +40.6 | 0.15 | 87.5% |
| **FactorDRQN** | **+16.6** | **+39.6** | **0.13** | **98.4%** |

Follower return declines as topology recovery improves, tracking a corresponding decline in mean retaliation intensity: architectures that mis-recover regime boundaries retaliate more aggressively and earn higher short-term follower return without recovering the correct strategic structure.

---

## Ablation Condition Mapping

| Paper name | Script | Flag | CSV |
|---|---|---|---|
| **FactorDRQN (proposed)** | `train_drqn_pd2_ablation_v2.py` | `A2b_no_noise` | `pd2_a2b_no_noise_results.csv` |
| A2b (128-unit + noise) | `train_drqn_pd2_ablation_v2.py` | `A2b` | `pd2_a2b_results.csv` |
| A3 (64-unit + noise) | `train_drqn_pd2_ablation.py` | `A3` | `pd2_a3_results.csv` |
| A2 (Recurrent-Factorized) | `train_drqn_pd2_ablation.py` | `A2` | `pd2_a2_results.csv` |
| A1 (Joint DRQN) | `train_drqn_pd2_ablation.py` | `A1` | `pd2_a1_results.csv` |
| A0 (Feedforward) | `train_drqn_pd2_ablation_v2.py` | `A0` | `pd2_a0_results.csv` |

> **Note:** `--ablation A3` in `train_drqn_pd2_ablation.py` produces the old 64-unit+noise condition (93.8%), **not** the proposed FactorDRQN. The proposed architecture is `--ablation A2b_no_noise` in `train_drqn_pd2_ablation_v2.py`.

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Episodes | 1,200 |
| Episode length | 256 steps |
| Burn-in / unroll | 10 / 32 |
| Leader ε decay | 1.0 → 0.05 over 120k steps |
| Follower ε decay | 1.0 → 0.05 over 60k steps |
| Leader replay | 300 episodes |
| Follower replay | 100 episodes |
| Hard sync interval | 80 episodes |
| Reward scale (d-head) | Δ\_d = 30 |
| Spite parameter | β\_E = 0.5 |
| Discount factor | γ = 0.98 |
| ρ-head hidden units | 128 (FactorDRQN) |
| τ-noise | disabled (FactorDRQN) |

---

## Phase Diagram Axes

| PD | X-axis | Y-axis | Fixed |
|---|---|---|---|
| PD2 | ρ\_max (0 → 0.50) | τ\_max (0.10 → 0.45) | β\_E = 0.5, ξ = 0 |
| PD6 | ξ (0 → 3.0) | ρ\_max (0 → 0.50) | β\_E = 0.5, τ\_max = 0.35 |

Regime thresholds (mean leader payoff, payoff-primary rule):
- **Deterrence**: > +15
- **Transition**: −15 to +15
- **Escalation**: < −15
