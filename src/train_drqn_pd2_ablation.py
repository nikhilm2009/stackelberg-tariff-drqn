# train_drqn_pd2_ablation.py
# ═══════════════════════════════════════════════════════════════════════════════
# Unified ablation runner for AIML Systems paper Section 6.1
#
# Three architectural conditions, identical training infrastructure:
#
#   A1 — Naive DRQN (V3):
#         Joint 30-action follower, shared reward
#         Baseline failure mode
#
#   A2 — Factorised heads, GRU→ρ-head (no tau-conditioning):
#         Separate d/ρ Q-heads, decomposed rewards
#         ρ-head receives full GRU hidden output (not tau-only)
#         Isolates contribution of tau-conditioning
#
#   A3 — Full FactorDRQN (tau-conditioned ρ-head):
#         Separate d/ρ Q-heads, decomposed rewards
#         ρ-head receives ONLY current tau (stateless MLP)
#         The proposed architecture
#
# Usage:
#   python src/train_drqn_pd2_ablation.py --ablation A1 --fast --workers 4
#   python src/train_drqn_pd2_ablation.py --ablation A2 --seeds 10 --workers 6
#   python src/train_drqn_pd2_ablation.py --ablation A3 --seeds 10 --workers 6
#   python src/train_drqn_pd2_ablation.py --ablation all --seeds 10 --workers 6
#
# All conditions use identical:
#   - Economic params (LF, beta_E=0.5)
#   - Episodes, ep_len, eval_frac, seeds
#   - Asymmetric epsilon, hard sync, replay buffer sizes
#   - Regime classifier and plot styling
# ═══════════════════════════════════════════════════════════════════════════════

import argparse, csv, importlib, itertools
import multiprocessing as mp
import os, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ── Regime colours — V10 IEEE spec ────────────────────────────────────────────
REGIME_IDX = {"Escalation":0, "Transition":1, "Deterrence":2, "Multiple":3}
CMAP = ListedColormap(["#e74c3c","#f39c12","#27ae60","#7f8c8d"])

def classify_regime(mean_leader_payoff):
    if mean_leader_payoff > 15.0:  return "Deterrence"
    if mean_leader_payoff < -15.0: return "Escalation"
    return "Transition"

# ── V10-aligned LF params ─────────────────────────────────────────────────────
def build_params(tau_max, rho_max):
    return dict(
        trade_form="power", M0=450.0, X0=500.0, E0=500.0,
        demand_elast=1.30, supply_elast=0.85, export_elast=1.00,
        kappa=0.20, lam=0.40, eta=0.30, delta=0.98,
        phi_inflation=0.50, leader_cost_w=0.008, follower_cost_w=0.015,
        tau_max=tau_max, d_max=0.18, psi_E=0.90, rho_max=rho_max,
        follower_cost_rho=0.10, beta_E=0.5,
    )

# ── Worker init ────────────────────────────────────────────────────────────────
def _worker_init(src_dir):
    for d in [src_dir, str(Path(src_dir).parent)]:
        if d not in sys.path:
            sys.path.insert(0, d)

# ── Single grid point ──────────────────────────────────────────────────────────
def run_grid_point(args_tuple):
    (tau_max, rho_max, ablation, episodes, ep_len,
     eval_frac, seed, force_cpu) = args_tuple

    import random as _r
    import numpy as np
    import torch

    _r.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device("cpu") if force_cpu else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if hasattr(torch.backends,"mps") and
            torch.backends.mps.is_available() else torch.device("cpu"))

    # Import sim — A1 uses V3 sim (joint step interface), A2/A3 use V5
    if ablation == "A1":
        sim_candidates = ("stackelberg_q3_tariff_econ_sim_v3",
                          "stackelberg_q3_tariff_econ_sim_v5")
    else:
        sim_candidates = ("stackelberg_q3_tariff_econ_sim_v5",
                          "stackelberg_q3_tariff_econ_sim_v3")
    sim = None
    for mod in sim_candidates:
        try: sim = importlib.import_module(mod); break
        except ImportError: pass
    if sim is None:
        raise RuntimeError("Cannot find stackelberg sim")

    econ_kw = build_params(tau_max, rho_max)
    env = sim.make_env(episode_len=ep_len, n_tau_bins=6, n_d_bins=6,
                       n_rho_bins=5, **econ_kw)

    from seq_replay import SequenceReplay

    def to_t(x):
        return torch.tensor(np.asarray(x, dtype=np.float32),
                            dtype=torch.float32, device=device)

    # ── Build agents based on ablation condition ───────────────────────────────
    if ablation == "A1":
        # A1: V3 naive — joint 30-action, shared reward, GRU follower
        # Replay capacity matches V3 original (symmetric LR=FR=500)
        from drqn_agent_v3 import DRQNAgent, DRQNConfig
        import torch.nn as nn

        l_cfg = DRQNConfig(obs_dim=6, act_dim=6,  hidden=128, rnn_type="gru",
                           dueling=True, gamma=0.98, lr=3e-4, clip_grad=1.0,
                           target_tau=0.015, burn_in=10, unroll=32)
        f_cfg = DRQNConfig(obs_dim=6, act_dim=30, hidden=128, rnn_type="gru",
                           dueling=True, gamma=0.98, lr=3e-4, clip_grad=1.0,
                           target_tau=0.015, burn_in=10, unroll=32)
        L = DRQNAgent(l_cfg, device)
        F = DRQNAgent(f_cfg, device)
        LR = SequenceReplay(capacity_episodes=500)
        FR = SequenceReplay(capacity_episodes=500)

    elif ablation in ("A2", "A3"):
        # A2/A3: Factorised follower, decomposed rewards
        from drqn_agent_v5 import (DRQNAgent, DRQNConfig,
                                    FactorisedFollowerAgent,
                                    FactorisedFollowerConfig)
        import torch.nn as nn

        l_cfg = DRQNConfig(obs_dim=6, act_dim=6, hidden=128, rnn_type="gru",
                           dueling=True, gamma=0.98, lr=3e-4, clip_grad=1.0,
                           target_tau=0.015, burn_in=10, unroll=32)
        f_cfg = FactorisedFollowerConfig(obs_dim=6, n_d=6, n_rho=5, hidden=128,
                                         rnn_type="gru", gamma=0.98, lr=3e-4,
                                         clip_grad=1.0, target_tau=0.015,
                                         burn_in=10, unroll=32,
                                         rho_input_mode="gru" if ablation=="A2" else "tau_only")
        L  = DRQNAgent(l_cfg, device)
        F  = FactorisedFollowerAgent(f_cfg, device)
        LR = SequenceReplay(capacity_episodes=300)
        FR = SequenceReplay(capacity_episodes=100)

    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    # ── Training loop ─────────────────────────────────────────────────────────
    l_eps = 1.0; f_eps = 1.0; eps_end = 0.05
    if ablation == "A1":
        # V3: single shared epsilon decay, no asymmetric schedule
        l_eps_delta = (1.0 - eps_end) / max(1, 60000)
        f_eps_delta = (1.0 - eps_end) / max(1, 60000)
    else:
        # A2/A3: asymmetric — leader 120k steps, follower 60k steps
        l_eps_delta = (1.0 - eps_end) / max(1, 100000)
        f_eps_delta = (1.0 - eps_end) / max(1,  50000)
    warmup     = 15
    f_curr_eps = 100 if ablation != "A1" else 0
    train_eps  = 0

    eval_start = int(episodes * (1 - eval_frac))
    l_payoffs, f_payoffs, taus, rhos = [], [], [], []
    l_obs0, f_obs0 = env.reset(seed=seed)

    for episode in range(1, episodes + 1):
        in_f_curr = (train_eps < f_curr_eps)
        curr_tau_idx = (episode - 1) % env.leader_act_dim

        l_ep = {"obs":[], "act":[], "rew":[], "done":[]}
        f_ep = {"obs":[], "act":[], "rew":[], "done":[]}
        lh = L.online.init_hidden(1, device)
        fh = F.online.init_hidden(1, device)
        lo, fo = l_obs0, f_obs0
        ep_lr, ep_fr, ep_t, ep_r = [], [], [], []

        for t in range(ep_len):
            lot = to_t(lo).view(1,1,-1)

            if in_f_curr and ablation != "A1":
                li = curr_tau_idx
                with torch.no_grad(): _, lh = L.online(lot, lh)
            else:
                li, lh = L.select_action(lot, lh, l_eps)

            tau_c = env.tau_values[li]
            fo_s  = env.follower_obs(tau_c)
            fot   = to_t(fo_s).view(1,1,-1)

            if ablation == "A1":
                fi, fh = F.select_action(fot, fh, f_eps)
                lo2, lr, fo2, fr, done, info = env.step(li, fi)
            else:
                fd, fri, fh = F.select_action(fot, fh, f_eps)
                lo2, lr, fo2, fr, done, info = env.step(li, f_d_idx=fd,
                                                         f_rho_idx=fri)

            l_ep["obs"].append(np.asarray(lo, dtype=np.float32))
            l_ep["act"].append(li)
            l_ep["rew"].append(float(lr))
            l_ep["done"].append(1.0 if done else 0.0)

            f_ep["obs"].append(np.asarray(fo_s, dtype=np.float32))

            if ablation == "A1":
                f_ep["act"].append(fi)
                f_ep["rew"].append(float(fr))
            else:
                f_ep["act"].append(np.array([fd, fri], dtype=np.int64))
                D_HEAD_SCALE = 30.0
                f_ep["rew"].append(np.array([
                    np.clip(float(info.get("reward_d",   fr)) / D_HEAD_SCALE, -3.0, 3.0),
                    np.clip(float(info.get("reward_rho", 0.0)),               -0.1, 0.1),
                ], dtype=np.float32))

            f_ep["done"].append(1.0 if done else 0.0)
            ep_lr.append(float(lr)); ep_fr.append(float(fr)); ep_t.append(info["tau"]); ep_r.append(info["rho"])
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

        def _learn(agent, replay, n=1):
            for _ in range(n):
                if not replay.episodes: break
                batch = replay.sample(32, 10, 32, device)
                agent.learn(batch)

        if ablation == "A1":
            _learn(L, LR, 1); _learn(F, FR, 1)
        elif in_f_curr:
            _learn(F, FR, 2)
        else:
            _learn(L, LR, 1); _learn(F, FR, 2)

        if ablation != "A1" and episode % 80 == 0:
            F.hard_sync_target()

        if episode >= eval_start:
            l_payoffs.extend(ep_lr); f_payoffs.extend(ep_fr); taus.extend(ep_t); rhos.extend(ep_r)

    mean_lpay = float(np.mean(l_payoffs)) if l_payoffs else 0.0
    mean_fpay = float(np.mean(f_payoffs)) if f_payoffs else 0.0
    mean_tau  = float(np.mean(taus))      if taus else 0.0
    mean_rho  = float(np.mean(rhos))      if rhos else 0.0
    regime    = classify_regime(mean_lpay)

    print(f"  [{ablation}] tau_max={tau_max:.2f} rho_max={rho_max:.2f} "
          f"seed={seed} -> {regime:12s} L={mean_lpay:+.1f} F={mean_fpay:+.1f} "
          f"tau={mean_tau:.3f} rho={mean_rho:.3f}")
    return tau_max, rho_max, regime, mean_lpay, mean_fpay, mean_tau, mean_rho

# ── Plot — V10 IEEE spec ──────────────────────────────────────────────────────
ABLATION_SUBTITLES = {
    "A1": "(A1) Naive DRQN — Joint Action Space, Shared Reward",
    "A2": "(A2) Factorised Heads — GRU→ρ-head, No Tau-Conditioning",
    "A3": "(A3) FactorDRQN — Tau-Conditioned ρ-head",
}
ABLATION_FILENAMES = {
    "A1": "fig_pd2_ablation_A1_naive.png",
    "A2": "fig_pd2_ablation_A2_factorised_gru_rho.png",
    "A3": "fig_pd2_ablation_A3_full_v5.png",
}

def plot_ablation(results, x_vals, y_vals, out_path, n_seeds, ablation):
    x_vals = np.array(x_vals); y_vals = np.array(y_vals)
    ng = len(x_vals)

    rg = np.full((ng, ng), REGIME_IDX["Deterrence"], dtype=float)
    lg = np.zeros((ng, ng))
    for tau_max, rho_max, regime, lpay, _, _, _ in results:
        xi = int(np.argmin(np.abs(x_vals - rho_max)))
        yi = int(np.argmin(np.abs(y_vals - tau_max)))
        rg[yi, xi] = REGIME_IDX.get(regime, 3)
        lg[yi, xi] = lpay

    dx = (x_vals[-1]-x_vals[0])/(ng-1)/2 if ng>1 else 0.05
    dy = (y_vals[-1]-y_vals[0])/(ng-1)/2 if ng>1 else 0.05
    xe = np.concatenate([[x_vals[0]-dx],(x_vals[:-1]+x_vals[1:])/2,[x_vals[-1]+dx]])
    ye = np.concatenate([[y_vals[0]-dy],(y_vals[:-1]+y_vals[1:])/2,[y_vals[-1]+dy]])

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.patch.set_facecolor("white")
    ax.pcolormesh(xe, ye, rg, cmap=CMAP, vmin=0, vmax=3, alpha=0.75, shading="flat")

    try:
        X, Y = np.meshgrid(x_vals, y_vals)
        ax.contour(X, Y, lg, levels=[0], colors=["black"], linewidths=2.0)
    except Exception: pass

    _dx = (x_vals[-1]-x_vals[0]) * 0.015
    _dy = (y_vals[-1]-y_vals[0]) * 0.015
    for yi in range(ng):
        for xi in range(ng):
            if rg[yi, xi] == REGIME_IDX["Multiple"]:
                ax.text(x_vals[xi], y_vals[yi], f"{lg[yi,xi]:+.1f}",
                        ha="center", va="center", fontsize=7,
                        color="black", fontweight="bold", zorder=6)

    for px, py, col, lbl in [
        (0.20, 0.35, "blue",       "LF"),
        (0.35, 0.25, "darkorange", "FF"),
        (0.27, 0.28, "white",      "CENTER"),
    ]:
        ax.scatter([px],[py], marker="*", s=180,
                   color=col, edgecolors="black", linewidths=0.8, zorder=5)
        ax.text(px+_dx, py+_dy, lbl, fontsize=8, zorder=6)

    ax.set_xlabel("Follower retaliation ceiling ($\\rho_{\\max}$)", fontsize=10)
    ax.set_ylabel("Leader tariff ceiling ($\\tau_{\\max}$)",        fontsize=10)
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(y_vals[0], y_vals[-1])
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25)
    fig.suptitle(f"Phase Diagram: Retaliation Capacity vs Tariff Capacity\n"
                 f"{ABLATION_SUBTITLES[ablation]}",
                 fontsize=10, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=l) for c, l in [
        ("#e74c3c","Escalation"), ("#f39c12","Transition"),
        ("#27ae60","Deterrence"), ("#7f8c8d","Multiple"),
    ]]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.00),
               borderpad=0.4, handlelength=1.2, columnspacing=0.8)
    plt.subplots_adjust(top=0.91, bottom=0.13, left=0.10, right=0.97)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Run one ablation condition ─────────────────────────────────────────────────
def run_ablation(ablation, args, out_dir, src_dir):
    rho_vals = np.linspace(0.00, 0.50, args.grid)
    tau_vals = np.linspace(0.10, 0.45, args.grid)
    force_cpu = (args.workers > 1)

    tasks = []
    for tau_max, rho_max in itertools.product(tau_vals, rho_vals):
        for seed in range(42, 42 + args.seeds):
            tasks.append((float(tau_max), float(rho_max), ablation,
                          args.episodes, args.ep_len, args.eval_frac,
                          seed, force_cpu))

    print(f"\n{'='*60}")
    print(f"Ablation {ablation}: {args.grid}×{args.grid} × {args.seeds} seeds"
          f" = {len(tasks)} runs")
    print(f"{ABLATION_SUBTITLES[ablation]}")
    print(f"{'='*60}")
    t0 = time.time()

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers,
                      initializer=_worker_init,
                      initargs=(src_dir,)) as pool:
            raw = pool.map(run_grid_point, tasks)
    else:
        _worker_init(src_dir)
        raw = [run_grid_point(t) for t in tasks]

    print(f"\n[{ablation}] Completed {len(raw)} runs in"
          f" {(time.time()-t0)/60:.1f} min")

    grid_agg = {}
    for tau_max, rho_max, regime, lpay, mf, mt, mr in raw:
        key = (round(tau_max,4), round(rho_max,4))
        if key not in grid_agg:
            grid_agg[key] = {"regimes":[], "lpays":[], "fpays":[], "taus":[], "rhos":[]}
        grid_agg[key]["regimes"].append(regime)
        grid_agg[key]["lpays"].append(lpay)
        grid_agg[key]["fpays"].append(mf)
        grid_agg[key]["taus"].append(mt)
        grid_agg[key]["rhos"].append(mr)

    final = []
    csv_path = out_dir / f"pd2_{ablation.lower()}_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_max","rho_max","regime","mean_lpay","mean_fpay",
                    "mean_tau","mean_rho","n_seeds"])
        for (tau_max, rho_max), d in sorted(grid_agg.items()):
            cnt    = Counter(d["regimes"])
            winner = cnt.most_common(1)[0]
            n_runs = len(d["regimes"])
            ml     = float(np.mean(d["lpays"]))
            mf_avg = float(np.mean(d["fpays"]))
            mt     = float(np.mean(d["taus"]))
            mr     = float(np.mean(d["rhos"]))
            regime = winner[0] if winner[1] > n_runs/2 else classify_regime(ml)
            final.append((tau_max, rho_max, regime, ml, mf_avg, mt, mr))
            w.writerow([f"{tau_max:.4f}", f"{rho_max:.4f}", regime,
                        f"{ml:.4f}", f"{mf_avg:.4f}", f"{mt:.4f}", f"{mr:.4f}", n_runs])

    print(f"[{ablation}] Regime counts: {dict(Counter(r[2] for r in final))}")
    print(f"[{ablation}] CSV: {csv_path}")

    out_fig = out_dir / ABLATION_FILENAMES[ablation]
    plot_ablation(final, list(rho_vals), list(tau_vals),
                  out_fig, args.seeds, ablation)
    return final

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="DRQN PD2 ablation sweep — A1/A2/A3")
    ap.add_argument("--ablation", type=str, default="A3",
                    choices=["A1","A2","A3","all"],
                    help="A1=naive, A2=factorised+GRU-rho, A3=full FactorDRQN, all=run all three")
    ap.add_argument("--grid",      type=int,   default=8)
    ap.add_argument("--episodes",  type=int,   default=1200)
    ap.add_argument("--ep_len",    type=int,   default=256)
    ap.add_argument("--eval_frac", type=float, default=0.25)
    ap.add_argument("--seeds",     type=int,   default=10)
    ap.add_argument("--workers",   type=int,   default=1)
    ap.add_argument("--out_dir",   type=str,   default="pd2_ablation_results")
    ap.add_argument("--fast",      action="store_true",
                    help="4×4 grid, 200 episodes, 1 seed — sanity check")
    args = ap.parse_args()

    if args.fast:
        args.grid = 4; args.episodes = 200; args.seeds = 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = str(Path(__file__).parent.resolve())

    ablations = ["A1","A2","A3"] if args.ablation == "all" else [args.ablation]

    print(f"DRQN PD2 Ablation Study (A1/A2/A3)")
    print(f"Conditions: {ablations}")
    print(f"grid={args.grid}  seeds={args.seeds}  episodes={args.episodes}")

    for abl in ablations:
        run_ablation(abl, args, out_dir, src_dir)

    print(f"\nAll ablations complete. Results in {out_dir}/")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
