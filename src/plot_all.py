# plot_all.py
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def _read_csv_dicts(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def _read_csv_rows(path):
    if not Path(path).exists():
        return []
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows

def _dedupe_sort_train(rows):
    # keep last record per episode; return episodes, loss, ema, eps
    by_ep = {}
    for r in rows:
        try:
            e = int(r["episode"])
            by_ep[e] = r
        except Exception:
            pass
    eps = sorted(by_ep.keys())
    loss = [float(by_ep[e]["loss"]) for e in eps]
    ema  = [float(by_ep[e]["ema_loss"]) for e in eps]
    epsi = [float(by_ep[e]["epsilon"]) for e in eps]
    return eps, loss, ema, epsi

def _dedupe_sort_eval(rows):
    by_ep = {}
    for r in rows:
        try:
            e = int(r["episode"])
            by_ep[e] = float(r["eval_pv"])
        except Exception:
            pass
    eps = sorted(by_ep.keys())
    pv  = [by_ep[e] for e in eps]
    return eps, pv

def _load_actions(path):
    # expects header: episode,timestep,action_idx
    rows = _read_csv_rows(path)
    if not rows:
        return []
    header = rows[0]
    if len(header) < 3 or header[2] != "action_idx":
        # permissive: handle our extended step log too
        # in that case, action_idx is col 3 anyway
        pass
    acts = []
    for row in rows[1:]:
        if not row or len(row) < 3: 
            continue
        try:
            acts.append(int(row[2]))
        except Exception:
            pass
    return acts

def _load_steps_basic(path, episode=None):
    """
    Load per-step rows from steps_*.csv:
    expected first 8 cols: episode,t,action_idx,tau,d,reward,q_max,q_gap,(opt hidden...)
    Returns filtered rows for a single episode, sorted by t.
    """
    rows = _read_csv_rows(path)
    if not rows:
        return []
    out = []
    for row in rows[1:]:
        if len(row) < 8: 
            continue
        try:
            e = int(row[0]); t = int(row[1])
            if (episode is not None) and (e != episode):
                continue
            # expected first 10 cols now:
            # episode,t,action_idx,tau,d,X,M,reward,q_max,q_gap
            a_idx = int(row[2])
            tau   = float(row[3])
            dval  = float(row[4]) if row[4] not in ("", "nan", "NaN") else np.nan
            xval  = float(row[5]) if row[5] not in ("", "nan", "NaN") else np.nan
            mval  = float(row[6]) if row[6] not in ("", "nan", "NaN") else np.nan
            rew   = float(row[7])
            qmax  = float(row[8])
            qgap  = float(row[9])
            out.append((e,t,a_idx,tau,dval,xval,mval,rew,qmax,qgap))
        except Exception:
            pass
    out.sort(key=lambda z: z[1])
    return out

def _last_episode_in_steps(path):
    rows = _read_csv_rows(path)
    if not rows:
        return None
    last = None
    for row in rows[1:]:
        if len(row) < 1: continue
        try:
            e = int(row[0])
            last = e if (last is None or e > last) else last
        except Exception:
            pass
    return last

def _load_hidden_vectors(path, episode=None):
    """
    If hidden was logged (variable-length rows), return (T, 2) PCA or None if unavailable.
    """
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return None, None
    rows = _read_csv_rows(path)
    if not rows:
        return None, None
    X, ts = [], []
    for row in rows[1:]:
        # require >8 columns to have hidden vector
        if len(row) <= 8: 
            continue
        try:
            e = int(row[0]); t = int(row[1])
            if (episode is not None) and (e != episode):
                continue
            vec = np.array([float(v) for v in row[8:]], dtype=np.float32)
            X.append(vec); ts.append(t)
        except Exception:
            pass
    if not X:
        return None, None
    X = np.vstack(X)
    p = PCA(n_components=2).fit_transform(X)
    return p, np.array(ts)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rnn", choices=["gru","lstm"], required=True)
    ap.add_argument("--art_dir", default="artifacts")
    ap.add_argument("--run_tag", default="", help="suffix used by trainer --run_tag")
    ap.add_argument("--episode", type=int, default=-1, help="episode for Q-traces/PCA; -1 selects last available")
    ap.add_argument("--bins", type=int, default=20, help="bins for (tau,d) heatmap")
    ap.add_argument("--heat_y", choices=["d","X","M"], default="d", help="variable for heatmap Y-axis")
    ap.add_argument("--out", default=None, help="output PNG (defaults to artifacts/summary_<rnn>[_tag].png)")
    args = ap.parse_args()

    suffix = f"_{args.run_tag}" if args.run_tag else ""
    art = Path(args.art_dir)
    plot = Path("plots")

    train_path = art / f"train_{args.rnn}{suffix}_log.csv"
    eval_path  = art / f"eval_{args.rnn}{suffix}_log.csv"
    act_path   = art / f"actions_{args.rnn}{suffix}.csv"
    step_path  = art / f"steps_{args.rnn}{suffix}.csv"

    # load logs
    train = _read_csv_dicts(train_path)
    if not train:
        raise SystemExit(f"Missing training log: {train_path}")
    ep_train, loss, ema, eps = _dedupe_sort_train(train)

    evalr = _read_csv_dicts(eval_path)
    ep_eval, pv = _dedupe_sort_eval(evalr) if evalr else ([], [])

    actions = _load_actions(act_path) if act_path.exists() else []
    steps_ep = None
    chosen_ep = None
    if step_path.exists():
        chosen_ep = (args.episode if args.episode >= 0 else _last_episode_in_steps(step_path))
        if chosen_ep is not None:
            steps_ep = _load_steps_basic(step_path, chosen_ep)

    # figure + layout (2 columns x 3 rows)
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    ax1, ax2, ax3, ax4, ax5, ax6 = axs.flat

    # 1) train loss + ema
    ax1.plot(ep_train, loss, "o-", label="loss", alpha=0.9)
    ax1.plot(ep_train, ema,  "s-", label="ema_loss", alpha=0.9)
    ax1.set_title(f"Training Loss ({args.rnn.upper()}{suffix})")
    ax1.set_xlabel("episode"); ax1.set_ylabel("loss")
    ax1.grid(True, linestyle="--", alpha=0.4); ax1.legend()

    # 2) eval PV
    if ep_eval:
        ax2.plot(ep_eval, pv, "o-", label="PV", alpha=0.9)
    else:
        ax2.text(0.5,0.5,"No eval logs found", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("Eval Present Value")
    ax2.set_xlabel("episode"); ax2.set_ylabel("PV")
    ax2.grid(True, linestyle="--", alpha=0.4); ax2.legend()

    # 3) policy histogram (action_idx)
    if actions:
        from collections import Counter
        c = Counter(actions)
        xs = sorted(c.keys()); ys = [c[x] for x in xs]
        ax3.bar(xs, ys)
        ax3.set_title("Policy Histogram (τ bins)")
        ax3.set_xlabel("action index"); ax3.set_ylabel("count")
        ax3.grid(True, axis="y", linestyle="--", alpha=0.3)
    else:
        ax3.text(0.5,0.5,"No action log found", ha="center", va="center", transform=ax3.transAxes)
    # 4) (tau, Y) heatmap where Y is chosen by --heat_y
    if steps_ep:
        tau = np.array([s[3] for s in steps_ep], dtype=np.float32)
        if args.heat_y == "d":
            yy = np.array([s[4] for s in steps_ep], dtype=np.float32)
            y_label = "d (follower)"
        elif args.heat_y == "X":
            yy = np.array([s[5] for s in steps_ep], dtype=np.float32)
            y_label = "X (exports)"
        else:  # "M"
            yy = np.array([s[6] for s in steps_ep], dtype=np.float32)
            y_label = "M (imports)"

        if np.all(np.isnan(yy)):
            ax4.text(0.5,0.5,f"No {y_label} in step log", ha="center", va="center", transform=ax4.transAxes)
        else:
            tb = np.linspace(np.nanmin(tau), np.nanmax(tau), args.bins+1)
            yb = np.linspace(np.nanmin(yy),  np.nanmax(yy),  args.bins+1)
            H, xedges, yedges = np.histogram2d(tau, yy, bins=[tb, yb])
            im = ax4.imshow(H.T, origin="lower",
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                            aspect="auto")
            fig.colorbar(im, ax=ax4, label="count")
            ax4.set_xlabel("τ (leader)")
            ax4.set_ylabel(y_label)
            ax4.set_title(f"(τ, {args.heat_y}) Heatmap (ep {chosen_ep})")
    else:
        ax4.text(0.5,0.5,"No step log found", ha="center", va="center", transform=ax4.transAxes)


    # 5) Q traces + rewards (episode)
    if steps_ep:
        ts  = [s[1] for s in steps_ep]
        qmx = [s[6] for s in steps_ep]
        qgp = [s[7] for s in steps_ep]
        rew = [s[5] for s in steps_ep]
        ax5.plot(ts, qmx, "o-", label="q_max", alpha=0.9)
        ax5.plot(ts, qgp, "s-", label="q_gap", alpha=0.9)
        ax5.set_title(f"Q Traces (episode {chosen_ep})")
        ax5.set_xlabel("t"); ax5.set_ylabel("Q")
        ax5.grid(True, linestyle="--", alpha=0.4); ax5.legend()

        ax6.plot(ts, rew, ".", label="reward", alpha=0.9)
        ax6.set_title(f"Rewards (episode {chosen_ep})")
        ax6.set_xlabel("t"); ax6.set_ylabel("reward")
        ax6.grid(True, linestyle="--", alpha=0.4); ax6.legend()
    else:
        ax5.text(0.5,0.5,"No step log found", ha="center", va="center", transform=ax5.transAxes)
        ax6.text(0.5,0.5,"No step log found", ha="center", va="center", transform=ax6.transAxes)

    plt.tight_layout()

    # optional: separate PCA plot if available (only if hidden logged)
    try:
        from sklearn.decomposition import PCA  # noqa: F401
        p, ts = (None, None)
        if step_path.exists():
            ep_for_pca = chosen_ep
            p, ts = _load_hidden_vectors(step_path, ep_for_pca)
        if p is not None:
            fig2, axp = plt.subplots(1,1, figsize=(5,4))
            sc = axp.scatter(p[:,0], p[:,1], c=ts, s=12)
            fig2.colorbar(sc, label="t")
            axp.set_title(f"Hidden PCA (ep {ep_for_pca})")
            axp.set_xlabel("PC1"); axp.set_ylabel("PC2")
            plt.tight_layout()
            out_p = Path(args.out).with_name(Path(args.out).stem + "_pca.png") if args.out else plot / f"summary_{args.rnn}{suffix}_pca.png"
            fig2.savefig(out_p)
            print("Saved PCA to", out_p)
    except Exception:
        pass

    out = Path(args.out) if args.out else plot / f"summary_{args.rnn}{suffix}.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
