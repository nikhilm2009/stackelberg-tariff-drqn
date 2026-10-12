# train_drqn_PD.py
# ═══════════════════════════════════════════════════════════════════════════════
# DRQN Phase Diagram Runner — V10-compatible architecture
#
# Mirrors v10_pd_ieee_leaderDepr_PD.py structure:
#   - RUN_CONFIG selects which PDs to run via "run_pds" list
#   - PD registry defines parameter axes, ranges, labels, star positions
#   - plot matches V10 IEEE spec exactly (_plot_phase_diagram)
#   - standalone PNG per PD, IEEE filenames
#
# Follower architecture controlled by RUN_CONFIG:
#   rho_input_mode: "tau_gru_size" = proposed FactorDRQN (τ-only, 128-unit)
#                   "tau_only"     = old A3 (τ-only, 64-unit)
#                   "gru"          = A2 (GRU→ρ-head)
#   tau_noise:      False = proposed FactorDRQN (no noise)
#                   True  = old A3 (with noise)
#
# Usage:
#   python src/train_drqn_PD.py                         # uses RUN_CONFIG defaults
#   python src/train_drqn_PD.py --pds PD6               # PD6 only
#   python src/train_drqn_PD.py --pds PD6 --seeds 10 --workers 6
#   python src/train_drqn_PD.py --fast                  # sanity check
#
# Supported PDs:
#   PD2 — rho_max × tau_max          (retaliation vs tariff capacity)
#   PD6 — gamma_diplo × rho_max      (diplomatic cost vs retaliation)
# ═══════════════════════════════════════════════════════════════════════════════

import argparse, csv, importlib, itertools, os, sys, time
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ── RUN_CONFIG ────────────────────────────────────────────────────────────────
RUN_CONFIG = {
    # Which PDs to run — override with --pds on CLI
    "run_pds":          ["PD6"],

    # Grid and seeds
    "grid":             8,          # 8×8 grid per PD
    "seeds":            10,         # seeds per grid point (10 for final results)
    "workers":          6,          # parallel workers (0 = serial)

    # Training budget
    "episodes":         1200,
    "ep_len":           256,
    "eval_frac":        0.25,       # last 25% for regime classification

    # Output
    "out_dir":          "pd_results",

    # Economic baseline (LF, V10-aligned, beta_E=0.5 for rho-head signal)
    "beta_E":           0.5,
    "gamma_diplo":      0.0,        # swept in PD6, fixed here for PD2

    # Follower architecture — proposed FactorDRQN (τ-only, 128-unit, no noise)
    "rho_input_mode":   "tau_gru_size",  # τ-only input, 128-unit ρ-head
    "tau_noise":        False,           # no noise — clean architectural result
}

# ── PD REGISTRY ───────────────────────────────────────────────────────────────
# LF and FF star positions (from drqn_params_v10_aligned.py)
_LF = dict(rho_max=0.20, tau_max=0.35, gamma_diplo=0.0)
_FF = dict(rho_max=0.35, tau_max=0.25, gamma_diplo=0.0)
_CT = dict(rho_max=0.27, tau_max=0.28, gamma_diplo=0.0)

PD_REGISTRY = [
    dict(
        name      = "PD2",
        title     = "Phase Diagram: Retaliation Capacity vs Tariff Capacity",
        x_attr    = "rho_max",    x_range = (0.00, 0.50),
        y_attr    = "tau_max",    y_range = (0.10, 0.45),
        xlabel    = r"Follower retaliation ceiling ($\rho_{\max}$)",
        ylabel    = r"Leader tariff ceiling ($\tau_{\max}$)",
        lf_x      = _LF["rho_max"],  lf_y  = _LF["tau_max"],
        ff_x      = _FF["rho_max"],  ff_y  = _FF["tau_max"],
        center_x  = _CT["rho_max"],  center_y = _CT["tau_max"],
        out_name  = "fig_pd2_drqn_retaliation_vs_tariff.png",
    ),
    dict(
        name      = "PD6",
        title     = "Phase Diagram: Diplomatic Cost vs Retaliation Capacity",
        x_attr    = "gamma_diplo", x_range = (0.00, 3.00),
        y_attr    = "rho_max",     y_range = (0.00, 0.50),
        xlabel    = r"Leader diplomatic cost ($\xi$)",
        ylabel    = r"Follower retaliation ceiling ($\rho_{\max}$)",
        lf_x      = _LF["gamma_diplo"], lf_y = _LF["rho_max"],
        ff_x      = _FF["gamma_diplo"], ff_y = _FF["rho_max"],
        center_x  = _CT["gamma_diplo"],  center_y = _CT["rho_max"],  # (0.0, 0.27)
        out_name  = "fig_pd6_drqn_diplomatic_cost_retaliation.png",
    ),
]

# ── REGIME COLOURS ─────────────────────────────────────────────────────────────
REGIME_COLORS = {"Escalation":0, "Transition":1, "Deterrence":2, "Multiple":3}
CMAP = ListedColormap(["#e74c3c","#f39c12","#27ae60","#7f8c8d"])

def classify_regime(mean_leader_payoff):
    if mean_leader_payoff > 15.0:  return "Deterrence"
    if mean_leader_payoff < -15.0: return "Escalation"
    return "Transition"

# ── BASE PARAMS ────────────────────────────────────────────────────────────────
def build_base_params(beta_E=0.5, gamma_diplo=0.0, **overrides):
    """LF-aligned base params. x_attr/y_attr values are injected as overrides."""
    p = dict(
        trade_form="power", M0=450.0, X0=500.0, E0=500.0,
        demand_elast=1.30, supply_elast=0.85, export_elast=1.00,
        kappa=0.20, lam=0.40, eta=0.30, delta=0.98,
        phi_inflation=0.50, leader_cost_w=0.008, follower_cost_w=0.015,
        tau_max=0.35, d_max=0.18, psi_E=0.90, rho_max=0.20,
        follower_cost_rho=0.10,
        beta_E=beta_E,
        gamma_diplo=gamma_diplo,
    )
    p.update(overrides)
    return p

# ── WORKER INIT ────────────────────────────────────────────────────────────────
def _worker_init(src_dir):
    for d in [src_dir, str(Path(src_dir).parent)]:
        if d not in sys.path:
            sys.path.insert(0, d)

# ── SINGLE GRID POINT ─────────────────────────────────────────────────────────
def run_grid_point(args_tuple):
    (x_val, y_val, x_attr, y_attr, episodes, ep_len,
     eval_frac, seed, force_cpu, beta_E, gamma_diplo,
     rho_input_mode, tau_noise) = args_tuple

    import random as _r, numpy as np, torch
    _r.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device("cpu") if force_cpu else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if hasattr(torch.backends,"mps") and
            torch.backends.mps.is_available() else torch.device("cpu"))

    sim = None
    for mod in ("stackelberg_q3_tariff_econ_sim_v5",
                "stackelberg_q3_tariff_econ_sim_v4"):
        try: sim = importlib.import_module(mod); break
        except ImportError: pass
    if sim is None:
        raise RuntimeError("Cannot find stackelberg sim v5")

    from drqn_agent_v5 import (DRQNAgent, DRQNConfig,
                                FactorisedFollowerAgent, FactorisedFollowerConfig)
    from seq_replay import SequenceReplay

    # Inject swept parameters
    overrides = {x_attr: x_val, y_attr: y_val}
    # overrides contains the swept param (e.g. gamma_diplo for PD6)
    # pass beta_E only; gamma_diplo comes from overrides to avoid duplicate kwarg
    base_gamma = overrides.pop("gamma_diplo", gamma_diplo)
    base_beta  = overrides.pop("beta_E", beta_E)
    econ_kw = build_base_params(beta_E=base_beta, gamma_diplo=base_gamma, **overrides)

    env = sim.make_env(episode_len=ep_len, n_tau_bins=6, n_d_bins=6,
                       n_rho_bins=5, **econ_kw)

    l_cfg = DRQNConfig(obs_dim=6, act_dim=6, hidden=128, rnn_type="gru",
                       dueling=True, gamma=0.98, lr=3e-4, clip_grad=1.0,
                       target_tau=0.015, burn_in=10, unroll=32)
    f_cfg = FactorisedFollowerConfig(obs_dim=6, n_d=6, n_rho=5, hidden=128,
                                     rnn_type="gru", gamma=0.98, lr=3e-4,
                                     clip_grad=1.0, target_tau=0.015,
                                     burn_in=10, unroll=32,
                                     rho_input_mode=rho_input_mode,
                                     tau_noise=tau_noise)
    L  = DRQNAgent(l_cfg, device)
    F  = FactorisedFollowerAgent(f_cfg, device)
    LR = SequenceReplay(capacity_episodes=300)
    FR = SequenceReplay(capacity_episodes=100)

    def to_t(x):
        return torch.tensor(np.asarray(x, dtype=np.float32),
                            dtype=torch.float32, device=device)

    # Epsilon — asymmetric decay: leader 120k steps, follower 60k steps
    l_eps = 1.0; f_eps = 1.0; eps_end = 0.05
    l_eps_delta = (1.0 - eps_end) / max(1, 120000)
    f_eps_delta = (1.0 - eps_end) / max(1,  60000)
    warmup      = 15
    f_curr_eps  = 100
    train_eps   = 0

    eval_start = int(episodes * (1 - eval_frac))
    l_payoffs, f_payoffs, taus, rhos = [], [], [], []
    l_obs0, f_obs0 = env.reset(seed=seed)

    for episode in range(1, episodes + 1):
        curr_tau_idx = (episode - 1) % env.leader_act_dim
        in_f_curr    = (train_eps < f_curr_eps)

        l_ep = {"obs":[], "act":[], "rew":[], "done":[]}
        f_ep = {"obs":[], "act":[], "rew":[], "done":[]}
        lh = L.online.init_hidden(1, device)
        fh = F.online.init_hidden(1, device)
        lo, fo = l_obs0, f_obs0
        ep_lr, ep_fr, ep_t, ep_r = [], [], [], []

        for t in range(ep_len):
            lot = to_t(lo).view(1,1,-1)
            if in_f_curr:
                li = curr_tau_idx
                with torch.no_grad(): _, lh = L.online(lot, lh)
            else:
                li, lh = L.select_action(lot, lh, l_eps)

            tau_c = env.tau_values[li]
            fo_s  = env.follower_obs(tau_c)
            fot   = to_t(fo_s).view(1,1,-1)
            fd, fri, fh = F.select_action(fot, fh, f_eps)

            lo2, lr, fo2, fr, done, info = env.step(li, f_d_idx=fd, f_rho_idx=fri)

            l_ep["obs"].append(np.asarray(lo, dtype=np.float32))
            l_ep["act"].append(li)
            l_ep["rew"].append(float(lr))
            l_ep["done"].append(1.0 if done else 0.0)

            f_ep["obs"].append(np.asarray(fo_s, dtype=np.float32))
            f_ep["act"].append(np.array([fd, fri], dtype=np.int64))
            D_HEAD_SCALE = 30.0; RHO_SCALE = 1.0
            f_ep["rew"].append(np.array([
                np.clip(float(info.get("reward_d",   fr)) / D_HEAD_SCALE, -3.0, 3.0),
                np.clip(float(info.get("reward_rho", 0.0)) / RHO_SCALE,   -0.1, 0.1),
            ], dtype=np.float32))
            f_ep["done"].append(1.0 if done else 0.0)

            ep_lr.append(float(lr))
            ep_fr.append(float(fr))
            ep_t.append(info["tau"])
            ep_r.append(info["rho"])
            lo, fo = lo2, fo2
            if l_eps > eps_end: l_eps -= l_eps_delta
            if f_eps > eps_end: f_eps -= f_eps_delta
            if done: break

        min_len = 10 + 32 + 1
        l_ep["obs"].append(np.asarray(lo, dtype=np.float32))
        f_ep["obs"].append(np.asarray(fo2 if not done else fo, dtype=np.float32))
        if len(l_ep["obs"]) >= min_len:
            LR.push_episode(l_ep); FR.push_episode(f_ep)

        l_obs0, f_obs0 = env.reset(seed=seed + episode)
        if episode < warmup: continue
        train_eps += 1

        # Epsilon reset after curriculum
        if train_eps == f_curr_eps + 1:
            l_eps = max(l_eps, 0.20)

        def _learn(agent, replay, n=1):
            for _ in range(n):
                if not replay.episodes: break
                batch = replay.sample(32, 10, 32, device)
                agent.learn(batch)

        if in_f_curr:
            _learn(F, FR, 2)
        else:
            _learn(L, LR, 1)
            _learn(F, FR, 2)

        # Hard sync every 80 eps
        if episode % 80 == 0:
            F.hard_sync_target()

        if episode >= eval_start:
            l_payoffs.extend(ep_lr)
            f_payoffs.extend(ep_fr)
            taus.extend(ep_t)
            rhos.extend(ep_r)

    mean_lpay = float(np.mean(l_payoffs)) if l_payoffs else 0.0
    mean_fpay = float(np.mean(f_payoffs)) if f_payoffs else 0.0
    mean_tau  = float(np.mean(taus))      if taus else 0.0
    mean_rho  = float(np.mean(rhos))      if rhos else 0.0
    regime    = classify_regime(mean_lpay)

    print(f"  {x_attr}={x_val:.3f} {y_attr}={y_val:.3f} seed={seed}"
          f" -> {regime:12s} L={mean_lpay:+.1f}"
          f" tau={mean_tau:.3f} rho={mean_rho:.3f}")
    return x_val, y_val, regime, mean_lpay, mean_fpay, mean_tau, mean_rho

# ── PLOTTER — exact V10 IEEE spec ─────────────────────────────────────────────
def _plot_phase_diagram(ax, x_vals, y_vals, regime_grid, Lpay_grid,
                        xlabel, ylabel, title,
                        lf_x=None, lf_y=None,
                        ff_x=None, ff_y=None,
                        center_x=None, center_y=None,
                        font_size=11):
    x_vals = np.array(x_vals); y_vals = np.array(y_vals)
    nx, ny = len(x_vals), len(y_vals)
    dx = (x_vals[-1]-x_vals[0])/(nx-1)/2 if nx>1 else 0.05
    dy = (y_vals[-1]-y_vals[0])/(ny-1)/2 if ny>1 else 0.05
    xe = np.concatenate([[x_vals[0]-dx],(x_vals[:-1]+x_vals[1:])/2,[x_vals[-1]+dx]])
    ye = np.concatenate([[y_vals[0]-dy],(y_vals[:-1]+y_vals[1:])/2,[y_vals[-1]+dy]])

    ax.set_facecolor("white")
    ax.pcolormesh(xe, ye, regime_grid, cmap=CMAP, vmin=0, vmax=3,
                  alpha=0.75, shading="flat")

    # L=0 contour — line only, no label (IEEE spec)
    try:
        X, Y = np.meshgrid(x_vals, y_vals)
        cs = ax.contour(X, Y, Lpay_grid, levels=[0],
                        colors=["black"], linewidths=2.5)
        # Keep only longest path — drop disjointed fragments
        try:
            all_paths = (cs.get_paths() if hasattr(cs,"get_paths")
                         else cs.collections[0].get_paths())
            if len(all_paths) > 1:
                longest = max(all_paths, key=lambda p: len(p.vertices))
                for col in cs.collections: col.remove()
                from matplotlib.patches import PathPatch
                ax.add_patch(PathPatch(longest, fill=False,
                             edgecolor="black", linewidth=2.5, zorder=4))
        except Exception:
            pass
    except Exception:
        pass

    # Multiple cell annotations
    _dx = (x_vals[-1]-x_vals[0]) * 0.015
    _dy = (y_vals[-1]-y_vals[0]) * 0.015
    for yi in range(ny):
        for xi in range(nx):
            if regime_grid[yi, xi] == REGIME_COLORS["Multiple"]:
                mean_L = Lpay_grid[yi, xi]
                ax.text(x_vals[xi], y_vals[yi], f"{mean_L:+.1f}",
                        ha="center", va="center", fontsize=7,
                        color="black", fontweight="bold", zorder=8)

    # Star markers + text labels (V10 style)
    if lf_x is not None:
        ax.scatter([lf_x],[lf_y], marker="*", s=180, color="blue",
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.text(lf_x+_dx, lf_y+_dy, "LF", fontsize=8,
                color="black", ha="left", va="bottom", zorder=6)
    if ff_x is not None:
        ax.scatter([ff_x],[ff_y], marker="*", s=180, color="darkorange",
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.text(ff_x+_dx, ff_y+_dy, "FF", fontsize=8,
                color="black", ha="left", va="bottom", zorder=6)
    if center_x is not None:
        ax.scatter([center_x],[center_y], marker="*", s=180, color="white",
                   edgecolors="black", linewidths=1.0, zorder=5)
        ax.text(center_x+_dx, center_y+_dy, "CENTER", fontsize=8,
                color="black", ha="left", va="bottom", zorder=6)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    x_pad = (x_vals[-1] - x_vals[0]) * 0.03
    ax.set_xlim(x_vals[0] - x_pad, x_vals[-1])
    ax.set_ylim(y_vals[0], y_vals[-1])
    ax.set_title(title, fontsize=font_size, pad=2)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25)


def _save_pd_figure(pd_cfg, x_vals, y_vals, regime_grid, Lpay_grid,
                    out_dir, n_seeds):
    patches = [mpatches.Patch(color=c, label=l) for c, l in [
        ("#e74c3c","Escalation"), ("#f39c12","Transition"),
        ("#27ae60","Deterrence"), ("#7f8c8d","Multiple"),
    ]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(pd_cfg["title"], fontsize=11, fontweight="bold")

    _plot_phase_diagram(
        ax, x_vals, y_vals, regime_grid, Lpay_grid,
        xlabel   = pd_cfg["xlabel"],
        ylabel   = pd_cfg["ylabel"],
        title    = "",
        lf_x     = pd_cfg["lf_x"],     lf_y     = pd_cfg["lf_y"],
        ff_x     = pd_cfg["ff_x"],     ff_y     = pd_cfg["ff_y"],
        center_x = pd_cfg.get("center_x"), center_y = pd_cfg.get("center_y"),
        font_size= 11,
    )

    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.00),
               borderpad=0.4, handlelength=1.2, columnspacing=0.8)
    plt.subplots_adjust(top=0.93, bottom=0.13, left=0.10, right=0.97)

    out_path = Path(out_dir) / pd_cfg["out_name"]
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── RUN ONE PD ────────────────────────────────────────────────────────────────
def run_pd(pd_cfg, cfg, out_dir, src_dir):
    ng       = cfg["grid"]
    seeds    = cfg["seeds"]
    episodes = cfg["episodes"]
    ep_len   = cfg["ep_len"]
    eval_f   = cfg["eval_frac"]
    beta_E   = cfg["beta_E"]
    gamma_d  = cfg["gamma_diplo"]
    workers  = cfg["workers"]

    x_vals = np.linspace(*pd_cfg["x_range"], ng)
    y_vals = np.linspace(*pd_cfg["y_range"], ng)

    force_cpu = (workers > 1)
    tasks = []
    rho_input_mode = cfg.get("rho_input_mode", "tau_gru_size")
    tau_noise      = cfg.get("tau_noise", False)

    for xv, yv in itertools.product(x_vals, y_vals):
        for seed in range(42, 42 + seeds):
            tasks.append((float(xv), float(yv),
                          pd_cfg["x_attr"], pd_cfg["y_attr"],
                          episodes, ep_len, eval_f,
                          seed, force_cpu, beta_E, gamma_d,
                          rho_input_mode, tau_noise))

    name = pd_cfg["name"]
    print(f"\n{'='*60}")
    print(f"{name}: {ng}×{ng} grid × {seeds} seeds = {len(tasks)} runs")
    print(f"x: {pd_cfg['x_attr']} {pd_cfg['x_range']}")
    print(f"y: {pd_cfg['y_attr']} {pd_cfg['y_range']}")
    print(f"{'='*60}")
    t0 = time.time()

    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers,
                      initializer=_worker_init,
                      initargs=(src_dir,)) as pool:
            raw = pool.map(run_grid_point, tasks)
    else:
        _worker_init(src_dir)
        raw = [run_grid_point(t) for t in tasks]

    print(f"\n{name} completed {len(raw)} runs in {(time.time()-t0)/60:.1f} min")

    # Aggregate with majority vote + payoff tiebreaker
    grid_agg = {}
    for xv, yv, regime, lpay, mf, mt, mr in raw:
        key = (round(float(xv),4), round(float(yv),4))
        if key not in grid_agg:
            grid_agg[key] = {"regimes":[], "lpays":[], "fpays":[], "taus":[], "rhos":[]}
        grid_agg[key]["regimes"].append(regime)
        grid_agg[key]["lpays"].append(lpay)
        grid_agg[key]["fpays"].append(mf)
        grid_agg[key]["taus"].append(mt)
        grid_agg[key]["rhos"].append(mr)

    # Build grids for plotting
    regime_grid = np.full((ng, ng), REGIME_COLORS["Deterrence"], dtype=float)
    Lpay_grid   = np.zeros((ng, ng))
    final_rows  = []

    csv_path = Path(out_dir) / f"{name.lower()}_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([pd_cfg["x_attr"], pd_cfg["y_attr"],
                    "regime","mean_lpay","mean_fpay","mean_tau","mean_rho","n_seeds"])
        for (xv, yv), d in sorted(grid_agg.items()):
            cnt    = Counter(d["regimes"])
            winner = cnt.most_common(1)[0]
            n_runs = len(d["regimes"])
            ml     = float(np.mean(d["lpays"]))
            mf_avg = float(np.mean(d["fpays"]))
            mt     = float(np.mean(d["taus"]))
            mr     = float(np.mean(d["rhos"]))
            # Majority vote; payoff tiebreaker on tie
            regime = winner[0] if winner[1] > n_runs/2 else classify_regime(ml)
            final_rows.append((xv, yv, regime, ml, mf_avg, mt, mr))
            w.writerow([f"{xv:.4f}", f"{yv:.4f}", regime,
                        f"{ml:.4f}", f"{mf_avg:.4f}", f"{mt:.4f}", f"{mr:.4f}", n_runs])

            xi = int(np.argmin(np.abs(x_vals - xv)))
            yi = int(np.argmin(np.abs(y_vals - yv)))
            regime_grid[yi, xi] = REGIME_COLORS.get(regime, 3)
            Lpay_grid[yi, xi]   = ml

    print(f"Regime counts: {dict(Counter(r[2] for r in final_rows))}")
    print(f"CSV: {csv_path}")

    _save_pd_figure(pd_cfg, list(x_vals), list(y_vals),
                    regime_grid, Lpay_grid, out_dir, seeds)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="DRQN Phase Diagram Runner")
    ap.add_argument("--pds",     nargs="+", default=None,
                    help="PDs to run e.g. --pds PD2 PD6")
    ap.add_argument("--grid",    type=int,   default=RUN_CONFIG["grid"])
    ap.add_argument("--seeds",   type=int,   default=RUN_CONFIG["seeds"])
    ap.add_argument("--workers", type=int,   default=RUN_CONFIG["workers"])
    ap.add_argument("--episodes",type=int,   default=RUN_CONFIG["episodes"])
    ap.add_argument("--out_dir", type=str,   default=RUN_CONFIG["out_dir"])
    ap.add_argument("--beta_E",  type=float, default=RUN_CONFIG["beta_E"])
    ap.add_argument("--rho_input_mode", type=str,
                    default=RUN_CONFIG["rho_input_mode"],
                    choices=["tau_gru_size","tau_only","gru"],
                    help="ρ-head input mode: tau_gru_size=FactorDRQN, tau_only=A3, gru=A2")
    ap.add_argument("--tau_noise", action="store_true",
                    default=RUN_CONFIG["tau_noise"],
                    help="Enable τ-input noise during training (default: off)")
    ap.add_argument("--fast",    action="store_true",
                    help="4×4 grid, 200 episodes, 1 seed (sanity check)")
    args = ap.parse_args()

    cfg = dict(RUN_CONFIG)
    cfg["grid"]     = args.grid
    cfg["seeds"]    = args.seeds
    cfg["workers"]  = args.workers
    cfg["episodes"] = args.episodes
    cfg["beta_E"]        = args.beta_E
    cfg["rho_input_mode"]= args.rho_input_mode
    cfg["tau_noise"]     = args.tau_noise
    cfg["ep_len"]        = 256
    cfg["eval_frac"]     = 0.25

    if args.fast:
        cfg["grid"] = 4; cfg["episodes"] = 200; cfg["seeds"] = 1

    run_pds = args.pds if args.pds else RUN_CONFIG["run_pds"]
    registry = {pd["name"]: pd for pd in PD_REGISTRY}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = str(Path(__file__).parent.resolve())

    print(f"DRQN PD Runner — PDs: {run_pds}")
    print(f"grid={cfg['grid']}  seeds={cfg['seeds']}  episodes={cfg['episodes']}")
    print(f"beta_E={cfg['beta_E']}  workers={cfg['workers']}")
    print(f"rho_input_mode={cfg['rho_input_mode']}  tau_noise={cfg['tau_noise']}")

    for pd_name in run_pds:
        if pd_name not in registry:
            print(f"[warn] Unknown PD: {pd_name} — skipping")
            continue
        run_pd(registry[pd_name], cfg, out_dir, src_dir)

    print("\nAll PDs complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
