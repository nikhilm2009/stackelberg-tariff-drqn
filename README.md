# TariffWar-DRQN  
Deep Recurrent Reinforcement Learning for a Stackelberg Tariff Game

This repository implements a **Deep Recurrent Q-Network (DRQN)** framework for studying a dynamic **Stackelberg leader–follower tariff game**.  
A learning **leader** (government) sets tariffs over time to maximize long-run economic value, while a **follower** (foreign economy) currently responds via a deterministic best-response rule.

The project investigates how **temporal memory architectures (GRU vs LSTM)** affect long-horizon policy learning in macroeconomic environments with inertia and delayed feedback.

---

## 1. Motivation

Classical Stackelberg models assume static optimization and instantaneous best responses.  
Real-world economic policy involves delayed responses, path dependence, and strategic reasoning over long horizons.  
This project explores whether **recurrent deep reinforcement learning** can recover economically meaningful Stackelberg equilibria — and how architecture choice influences learned policy.

---

## 2. High-Level Overview

### Agents
- **Leader (DRQN)**  
  - Chooses tariff level τ each timestep  
  - Observes macroeconomic state with partial observability  
  - Uses sequence replay with burn-in and unroll windows  

- **Follower (Baseline)**  
  - Deterministic best-response agent  
  - Chooses depreciation d to maximize its own instantaneous payoff  

### Environment
- State variables: Imports (M), Exports (X), previous tariff, previous depreciation  
- Dynamics include inertia and adjustment rates  
- Rewards capture revenue, consumer loss, inflation, and action costs  

---

## 3. Leader Model Architecture

Observation → Encoder (MLP) → GRU or LSTM → Dueling Q-Head

Q(s,a) = V(s) + (A(s,a) − mean(A))

Key features:
- GRU or LSTM backend
- Dueling architecture
- Double DQN targets
- Soft target updates
- Gradient clipping

---

## 4. Experience Replay (Sequence-Based)

- Episode-wise replay buffer
- Training samples are contiguous sequences
- Each sample contains burn-in steps (hidden warmup) and unroll steps (TD loss)

Constraint:
burn_in + unroll + 1 ≤ episode_length

---

## 5. Repository Structure

tariffwar-drqn/
├── README.md
├── src/
│   ├── train_drqn.py
│   ├── drqn_agent.py
│   ├── recurrent_q_net.py
│   ├── seq_replay.py
│   ├── device.py
│   ├── stackelberg_q3_tariff_econ_sim_drqn.py
│   ├── plot_all.py
│   └── plot_gru_lstm.py
├── artifacts/
├── models/
└── data/

---

## 6. Installation & Setup

Requirements:
- Python ≥ 3.10
- PyTorch
- numpy
- matplotlib

Using uv:
uv venv
source .venv/bin/activate
uv pip install torch numpy matplotlib

---

## 7. Training

GRU:
uv run python src/train_drqn.py --rnn gru --episodes 400 --unroll 64 --fresh_log --run_tag u64_h128_gru

LSTM:
uv run python src/train_drqn.py --rnn lstm --episodes 400 --unroll 64 --fresh_log --run_tag u64_h128_lstm

---

## 8. Key Findings

- GRU and LSTM both train stably.
- GRU learns a moderate tariff policy with higher long-run PV.
- LSTM tends toward maximal tariffs and lower PV.

---

## 9. Planned Extensions

- Learning follower (DQN/DRQN)
- Alternating Stackelberg training
- Stochastic follower responses

---

## 10. License

Specify license.

---

## 11. Citation

TariffWar-DRQN: Deep Recurrent Reinforcement Learning for a Stackelberg Tariff Game
