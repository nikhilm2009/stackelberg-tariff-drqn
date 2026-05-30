# TariffWar-DRQN
**FactorDRQN: Recurrent Multi-Agent Reinforcement Learning for a Stackelberg Tariff Conflict**

This repository implements the FactorDRQN architecture studied in:

> *"From Tabular Q-Learning to Recurrent Multi-Agent Systems: Recovering Strategic Phase Structure in Adaptive Socio-Economic Conflict"*  
> AIMLSystems 2026, Track 3 — AI-ML–Driven Socio-Economic and Decision Systems

The project studies whether recurrent multi-agent systems can recover the qualitative phase structure of a two-country Stackelberg tariff conflict under function approximation.

---

## Architecture

Two agents interact in a Stackelberg game:

- **Leader** — sets tariff τ ∈ {0, …, τ_max} over 6 bins using a GRU-based DRQN
- **Follower (FactorDRQN)** — responds with depreciation d and retaliation ρ using **factorised Q-heads**:
  - `Q_d` head: full GRU trunk → dueling head (6 depreciation bins)
  - `Q_ρ` head: τ/τ_max only → stateless MLP → dueling head (5 retaliation bins)

The key architectural contribution is the **τ-conditional retaliation pathway** — the ρ-head bypasses the GRU entirely, conditioning only on current tariff provocation. This prevents economic-flow updates from overwriting the sparse retaliation signal.

---

## Repository Structure

```
tariffwar-drqn/
├── src/
│   ├── recurrent_q_net_v5.py          # RecurrentQNet (leader) + FactorisedFollowerQNet
│   ├── drqn_agent_v5.py               # DRQNAgent + FactorisedFollowerAgent
│   ├── stackelberg_q3_tariff_econ_sim_v5.py   # Environment (payoff-first, beta_E, gamma_diplo)
│   ├── seq_replay.py                  # Sequence replay buffer
│   │
│   ├── train_drqn_v5.py               # Single LF training run
│   ├── train_drqn_PD.py               # Unified phase diagram runner (PD2 + PD6)
│   ├── train_drqn_pd2_ablation.py     # Ablation study runner (A1 / A2 / A3)
│   ├── train_drqn_pd2_v3.py           # Naive DRQN baseline (V3, joint action space)
│   │
│   ├── plot_drqn_v5.py                # Training diagnostics (5-figure suite)
│   ├── plot_pd2_ablation_2x2.py       # 2×2 ablation centerpiece figure
│   ├── plot_pd6_comparison.py         # PD6 side-by-side comparison figure
│   ├── plot_convergence_inset.py      # 2-panel convergence inset figure
│   ├── probe_follower_v5.py           # Load checkpoint, probe ρ-rate per τ bin
│   │
│   └── v10_pd_ieee_leaderDepr_PD.py   # Tabular V10 phase diagram runner (reference)
│
├── models_v5/                         # Training checkpoints and CSV logs
├── pd_results/                        # Phase diagram CSVs and figures
├── pd2_ablation_results/              # Ablation study CSVs and figures
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

**Single training run (LF calibration):**
```bash
python src/train_drqn_v5.py
```

**Phase diagram sweep — PD2 and PD6:**
```bash
python src/train_drqn_PD.py --pds PD2 PD6 --seeds 5 --workers 6
```

**Ablation study:**
```bash
# A2 only (new condition — factorised + GRU→ρ-head)
python src/train_drqn_pd2_ablation.py --ablation A2 --seeds 5 --workers 6

# All three conditions sequentially
python src/train_drqn_pd2_ablation.py --ablation all --seeds 5 --workers 6
```

**Naive baseline (A1):**
```bash
python src/train_drqn_pd2_v3.py --seeds 5 --workers 6
```

---

## Plotting

**Training diagnostics:**
```bash
python src/plot_drqn_v5.py --regime lf --rnn gru
# Output: plots_v5/fig1_training_lf_gru.png ... fig5_episode_lf_gru.png
```

**2×2 ablation centerpiece figure:**
```bash
python src/plot_pd2_ablation_2x2.py \
    --tabular pd2_tabular_results.csv \
    --a1      pd2_v3_results.csv \
    --a2      pd2_a2_results.csv \
    --a3      pd2_results.csv \
    --out     fig_pd2_ablation_2x2.png
```

**PD6 comparison figure:**
```bash
python src/plot_pd6_comparison.py \
    --tabular fig_pd6_diplomatic_cost_retaliation_tabular.csv \
    --drqn    pd6_results_drqn.csv \
    --out     fig_pd6_comparison.png
```

**Convergence inset:**
```bash
python src/plot_convergence_inset.py \
    --steps models_v5/steps_lf_gru.csv \
    --eval  models_v5/eval_lf_gru.csv \
    --out   fig_convergence_inset.png
```

---

## Key Results

| Condition | Architecture | Topology Agreement |
|---|---|---|
| A1 — Naive DRQN | Joint 30-action space, shared reward | 57.8% |
| A2 — Factorised DRQN | Separate heads, GRU→ρ-head | 82.8% |
| A3 — FactorDRQN (proposed) | τ-conditioned ρ-head | **92.2%** |

- PD2 (retaliation vs tariff capacity): **92.2%** cell-level agreement with tabular reference
- PD6 (diplomatic cost vs retaliation): **90.6%** agreement — topology recovery generalises across perturbation axes

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
| Reward scale (d-head) | Δ_d = 30 |
| Spite parameter | β_E = 0.5 |
| Discount factor | γ = 0.98 |

---

## Phase Diagram Conditions

| PD | X-axis | Y-axis |
|---|---|---|
| PD2 | ρ_max (0 → 0.50) | τ_max (0.10 → 0.45) |
| PD6 | γ_diplo (0 → 3.0) | ρ_max (0 → 0.50) |

Three regime thresholds (mean leader payoff): Deterrence > +15, Escalation < −15, Transition between.

---

## Citation

> *From Tabular Q-Learning to Recurrent Multi-Agent Systems: Recovering Strategic Phase Structure in Adaptive Socio-Economic Conflict*  
> Submitted to AIMLSystems 2026, Track 3 — AI-ML–Driven Socio-Economic and Decision Systems  
> *(under review)*
