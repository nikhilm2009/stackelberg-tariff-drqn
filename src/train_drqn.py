# train_drqn.py (drop-in)
import argparse, random
import importlib
import numpy as np
import torch
from pathlib import Path

from device import get_device
from seq_replay import SequenceReplay
from drqn_agent import DRQNAgent, DRQNConfig

# ---- Wrapper to ensure info["d"] exists; with overrides + probe ----
class InfoAugmentEnv:
    """
    Wrap ANY env; ensures info["d"] is present using:
      1) Explicit overrides: d_from="info:KEY" | "attr:NAME" | "state:KEY" | "obs:INDEX"
      2) Heuristics: scan info keys/attrs/state for likely 'd' names
      3) Optional probe dump to artifacts/d_probe.txt for first few steps
    Returns classic (obs, reward, done, info).
    """
    def __init__(self, env, d_from:str=None, probe:bool=False, probe_steps:int=5, probe_path:Path=None):
        self._env = env
        self._d_from = d_from
        self._probe = probe
        self._probe_steps = probe_steps
        self._probe_count = 0
        self._probe_path = probe_path or Path("artifacts/d_probe.txt")

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_from_override(self, obs, info):
        if not self._d_from:
            return None
        try:
            src, spec = self._d_from.split(":", 1)
            src = src.strip().lower()
            spec = spec.strip()
            if src == "info":
                return info.get(spec, None) if isinstance(info, dict) else None
            elif src == "attr":
                return getattr(self._env, spec, None)
            elif src == "state":
                st = getattr(self._env, "state", None)
                return st.get(spec, None) if isinstance(st, dict) else None
            elif src == "obs":
                idx = int(spec)
                arr = np.asarray(obs)
                if arr.ndim == 0:
                    return None
                return float(arr[idx]) if (idx >= -arr.shape[-1] and idx < arr.shape[-1]) else None
        except Exception:
            return None
        return None

    def _extract_d_heuristic(self, obs, info):
        # 1) info keys
        if isinstance(info, dict):
            # exact first
            for k in ("d", "follower_d"):
                if k in info: return info[k]
            # fuzzy
            for k in list(info.keys()):
                kl = str(k).lower()
                if any(token in kl for token in ["depr", "depreci", "fx", "curr"]) and isinstance(info[k], (int, float, np.floating)):
                    return info[k]
        # 2) env attributes
        for k in ("d", "follower_d", "last_d", "_last_d", "fx_depr", "fx_depreciation", "currency_dep"):
            if hasattr(self._env, k):
                v = getattr(self._env, k)
                if isinstance(v, (int, float, np.floating)): 
                    return v

        # 3) state dict inside env
        st = getattr(self._env, "state", None)
        if isinstance(st, dict):
            for k, v in st.items():
                kl = str(k).lower()
                if (k in ("d","follower_d")) or any(t in kl for t in ["depr","depreci","fx","curr"]):
                    if isinstance(v, (int, float, np.floating)): return v
        return None

    def _probe_dump(self, obs, info):
        if not self._probe or self._probe_count >= self._probe_steps:
            return
        self._probe_count += 1
        try:
            self._probe_path.parent.mkdir(parents=True, exist_ok=True)
            with self._probe_path.open("a") as f:
                f.write("\n=== PROBE STEP {} ===\n".format(self._probe_count))
                # info keys/values (first 20 keys)
                if isinstance(info, dict):
                    f.write("info keys: {}\n".format(list(info.keys())[:20]))
                    sample = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(type(v)))
                              for k, v in list(info.items())[:20]}
                    f.write("info sample: {}\n".format(sample))
                else:
                    f.write("info is not dict: {}\n".format(type(info)))
                # env attrs candidates
                cand_attrs = [k for k in dir(self._env) if any(t in k.lower() for t in ["d","depr","depreci","fx","curr"])]
                f.write("env attrs (cand): {}\n".format(cand_attrs[:30]))
                # state dict
                st = getattr(self._env, "state", None)
                if isinstance(st, dict):
                    f.write("state keys: {}\n".format(list(st.keys())[:30]))
        except Exception:
            pass

    def step(self, action):
        out = self._env.step(action)
        # Gymnasium 5-tuple support
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
        else:
            obs, reward, done, info = out

        if info is None:
            info = {}

        # 1) explicit override
        d_val = self._get_from_override(obs, info)
        # 2) heuristic if override failed
        if d_val is None:
            d_val = self._extract_d_heuristic(obs, info)

        # write back if we found it
        if d_val is not None:
            try:
                info["d"] = float(d_val)
            except Exception:
                info["d"] = d_val  # keep original if not float-able
        else:
            # probe a few early steps to a file
            self._probe_dump(obs, info)

        # also surface useful econ signals if present (imports/exports)
        for key in ("M", "X"):
            if isinstance(info, dict) and key in info:
                continue
            # try env attribute
            v = getattr(self._env, key, None)
            if v is None:
                # try state dict on env
                st = getattr(self._env, "state", None)
                if isinstance(st, dict):
                    v = st.get(key, None)
            if v is not None:
                try:
                    info[key] = float(v)
                except Exception:
                    info[key] = v  # keep original type if not float-able

        return obs, reward, done, info


# --- Import simulator (supports either filename) ---
sim = None
for modname in ("stackelberg_q3_tariff_econ_sim_drqn", "stackelberg_q3_tariff_econ_sim"):
    try:
        sim = importlib.import_module(modname)
        print(f"[info] using simulator module: {modname}")
        break
    except Exception:
        pass
if sim is None:
    raise RuntimeError(
        "Could not import simulator. Expected stackelberg_q3_tariff_econ_sim_drqn "
        "or stackelberg_q3_tariff_econ_sim in project root."
    )


def make_env(episode_len: int):
    """Create env with given horizon via simulator helper."""
    return sim.make_env(episode_len)


def to_tensor(x, device):
    x = np.asarray(x, dtype=np.float32)
    return torch.tensor(x, dtype=torch.float32, device=device)


def run(args):
    device = get_device()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.ep_len)
    env = InfoAugmentEnv(
        env,
        d_from=args.d_from,                  # e.g., "info:dep_rate" or "attr:last_dep" or "state:d" or "obs:3"
        probe=args.probe_d,
        probe_steps=args.probe_d_steps,
        probe_path=Path(args.ckpt_dir) / f"d_probe_{args.rnn}{('_' + args.run_tag) if args.run_tag else ''}.txt"
    )

    # Probe observation/action dims
    obs0 = env.reset(seed=args.seed)
    obs_dim = int(np.asarray(obs0, dtype=np.float32).shape[-1]) if hasattr(obs0, "__len__") else 1
    act_dim = args.act_bins  # discretized actions for the trained agent (e.g., leader τ bins)

    cfg = DRQNConfig(
        obs_dim=obs_dim, act_dim=act_dim, hidden=args.hidden, rnn_type=args.rnn,
        dueling=not args.no_dueling, gamma=args.gamma, lr=args.lr, clip_grad=1.0,
        target_tau=args.tau, burn_in=args.burn_in, unroll=args.unroll
    )
    agent = DRQNAgent(cfg, device)
    replay = SequenceReplay(capacity_episodes=args.replay_episodes)

    # ---- Global EMA loss tracker (cross-episode) ----
    loss_ema = None

    # ---- Logging / checkpoint paths (support fresh_log + run_tag) ----
    art_dir = Path("artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.run_tag}" if args.run_tag else ""
    log_path = art_dir / f"train_{args.rnn}{suffix}_log.csv"
    eval_log_path = art_dir / f"eval_{args.rnn}{suffix}_log.csv"

    # ---- Action log path (one row per step) ----
    act_log_path = art_dir / f"actions_{args.rnn}{suffix}.csv"
    if args.fresh_log or not act_log_path.exists():
        with act_log_path.open("w") as f:
            f.write("episode,timestep,action_idx\n")


# ---- Step log path (one row per step) ----
    step_log_path = art_dir / f"steps_{args.rnn}{suffix}.csv"
    if args.fresh_log or not step_log_path.exists():
        with step_log_path.open("w") as f:
            # now includes X and M so we can choose heatmap variable
            f.write("episode,t,action_idx,tau,d,X,M,reward,q_max,q_gap\n")



    if args.fresh_log:
        with log_path.open("w") as f:
            f.write("episode,loss,ema_loss,epsilon\n")
        with eval_log_path.open("w") as f:
            f.write("episode,eval_pv\n")
    else:
        if not log_path.exists():
            with log_path.open("w") as f:
                f.write("episode,loss,ema_loss,epsilon\n")
        if not eval_log_path.exists():
            with eval_log_path.open("w") as f:
                f.write("episode,eval_pv\n")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_prefix = f"{args.ckpt_prefix}_{args.rnn}{suffix}"

    global_step = 0
    eps = args.eps_start
    eps_delta = (args.eps_start - args.eps_end) / max(1, args.eps_decay_steps)

    def discretize_action(a_idx: int):
        # Map discrete index to real tariff value; customize to your bins
        return (
            args.min_act + (args.max_act - args.min_act) * (a_idx / (act_dim - 1))
            if act_dim > 1 else args.min_act
        )

    for episode in range(1, args.episodes + 1):
        ep = {k: [] for k in ["obs", "act", "rew", "done"]}
        # Initialize per-episode hidden state for agent
        h = agent.online.init_hidden(batch_size=1, device=device)
        obs = obs0

        for t in range(args.ep_len):
            o_t = to_tensor(obs, device).view(1, 1, -1)  # [1,1,obs]
            # select action + also fetch Q-values for logging
            a_idx, h = agent.select_action(o_t, h, epsilon=eps)
            act = discretize_action(a_idx)

            # compute Q-values on the SAME input (greedy view) for logging
            q_vec, _ = agent.eval_q(o_t, h)
            q_sorted = np.sort(q_vec)[::-1]
            q_max = float(q_sorted[0])
            q_gap = float(q_sorted[0] - (q_sorted[1] if len(q_sorted) > 1 else q_sorted[0]))

            nxt, r, done, info = env.step(act)
            # log action index for policy histogram
            with act_log_path.open("a") as f:
                f.write(f"{episode},{t},{a_idx}\n")
            # build the per-step row for consolidated plots (τ–d heatmap, Q traces)
            row = f"{episode},{t},{a_idx},{act},{float(info.get('d', float('nan')))}," \
                f"{float(r)},{q_max},{q_gap}"
            with step_log_path.open("a") as f:
                f.write(row + "\n")

            # follower d from env info (ensure your env adds 'd' to info; otherwise fallback)
            d_val = float(info.get("d", info.get("follower_d", float("nan"))))

            # build the step row
            row = f"{episode},{t},{a_idx},{act},{d_val},{float(r)},{q_max},{q_gap}"

            # optional: append hidden vector every K steps
            if args.log_hidden_every and (t % args.log_hidden_every == 0):
                def _flat_hidden(hh):
                    # GRU: Tensor[L=1,B=1,H] -> [H] ; LSTM: tuple(h,c) -> use h
                    if isinstance(hh, tuple):  # LSTM
                        hh = hh[0]
                    return hh.squeeze(0).squeeze(0).detach().cpu().numpy()
                h_vec = _flat_hidden(h)
                row += "," + ",".join(map(str, h_vec.tolist()))

            # write the row
            with step_log_path.open("a") as f:
                f.write(row + "\n")

            ep["obs"].append(np.asarray(obs, dtype=np.float32))
            ep["act"].append(a_idx)
            ep["rew"].append(float(r))
            ep["done"].append(1.0 if done else 0.0)

            obs = nxt
            global_step += 1

            if eps > args.eps_end:
                eps -= eps_delta
            if done:
                break

        # push terminal observation and store only if long enough
        ep["obs"].append(np.asarray(obs, dtype=np.float32))
        if len(ep["obs"]) >= (cfg.burn_in + cfg.unroll + 1):
            replay.push_episode(ep)
        obs0 = env.reset(seed=args.seed + episode)

        # warmup
        if episode < args.warmup_episodes:
            if episode % 10 == 0:
                print(f"[warmup] episode {episode}/{args.episodes}, replay size={len(replay.episodes)}")
            continue

        # training steps per episode (can be >1)
        for _ in range(args.updates_per_ep):
            batch = replay.sample(
                batch_size=args.batch, burn_in=cfg.burn_in, unroll=cfg.unroll, device=device
            )
            loss = agent.learn(batch)

        # update global EMA once per episode
        if loss_ema is None:
            loss_ema = loss
        else:
            loss_ema = 0.9 * loss_ema + 0.1 * loss

        if episode % args.log_interval == 0:
            print(
                f"ep {episode:04d} | loss {loss:.4f} | ema_loss {loss_ema:.4f} "
                f"| eps {eps:.3f} | replay {len(replay.episodes)}"
            )
            with log_path.open("a") as f:
                f.write(f"{episode},{loss:.6f},{loss_ema:.6f},{eps:.4f}\n")

            # checkpoint
            if episode % args.ckpt_interval == 0:
                ckpt_path = ckpt_dir / f"{ckpt_prefix}_ep{episode}.pt"
                torch.save(agent.online.state_dict(), ckpt_path)

        # periodic quick eval
        if episode % (args.log_interval * 2) == 0:
            eval_return = 0.0
            obs = env.reset(seed=args.seed + 1000 + episode)
            h = agent.online.init_hidden(1, device)
            for _ in range(args.eval_len):
                with torch.no_grad():
                    o_t = to_tensor(obs, device).view(1, 1, -1)
                    a_idx, h = agent.select_action(o_t, h, epsilon=0.0)
                    act = discretize_action(a_idx)
                obs, r, done, _ = env.step(act)
                eval_return += float(r)
                if done:
                    break
            print(f"[eval] ep {episode:04d} | PV {eval_return:.2f}")
            with eval_log_path.open("a") as f:
                f.write(f"{episode},{eval_return:.6f}\n")

    # final eval
    eval_return = 0.0
    obs = env.reset(seed=args.seed + 999)
    h = agent.online.init_hidden(1, device)
    for t in range(args.eval_len):
        with torch.no_grad():
            o_t = to_tensor(obs, device).view(1, 1, -1)
            a_idx, h = agent.select_action(o_t, h, epsilon=0.0)
            act = discretize_action(a_idx)
        obs, r, done, info = env.step(act)
        eval_return += float(r)
        if done:
            break
    print({"eval_return": eval_return, "episodes": args.episodes, "device": str(device)})
    with eval_log_path.open("a") as f:
        f.write(f"{episode},{eval_return:.6f}\n")

    # final checkpoint
    final_ckpt = ckpt_dir / f"{ckpt_prefix}_final.pt"
    torch.save(agent.online.state_dict(), final_ckpt)
    print(f"[ckpt] saved {final_ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--ep_len", type=int, default=256)
    ap.add_argument("--eval_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)

    # Model/learning
    ap.add_argument("--rnn", type=str, default="gru", choices=["gru", "lstm"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--burn_in", type=int, default=10)
    ap.add_argument("--unroll", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--updates_per_ep", type=int, default=1)
    ap.add_argument("--replay_episodes", type=int, default=1000)
    ap.add_argument("--warmup_episodes", type=int, default=10)

    # Exploration
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=8000)

    # Discrete action mapping
    ap.add_argument("--act_bins", type=int, default=6)
    ap.add_argument("--min_act", type=float, default=0.0)
    ap.add_argument("--max_act", type=float, default=0.3)

    ap.add_argument("--no_dueling", action="store_true")
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--log_hidden_every", type=int, default=0, help="If >0, log hidden state every K steps to steps_* CSV")


    # Checkpoints + logging
    ap.add_argument("--ckpt_interval", type=int, default=20)
    ap.add_argument("--ckpt_prefix", type=str, default="drqn")
    ap.add_argument("--ckpt_dir", type=str, default="models")
    ap.add_argument("--fresh_log", action="store_true", help="Overwrite train/eval CSV logs at start")
    ap.add_argument("--run_tag", type=str, default="", help="Optional tag for unique CSV/ckpt names (e.g., 'u64_h256')")

    ap.add_argument("--d_from", type=str, default=None,
                    help="Where to read follower d from: 'info:KEY' | 'attr:NAME' | 'state:KEY' | 'obs:INDEX'")
    ap.add_argument("--probe_d", action="store_true",
                    help="Dump a short discovery log of info keys/env attrs to artifacts/d_probe_*.txt when d is missing")
    ap.add_argument("--probe_d_steps", type=int, default=5,
                    help="How many early steps to probe (only when d is missing)")
    args = ap.parse_args()
    run(args)
