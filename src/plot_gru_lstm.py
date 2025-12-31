import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--gru_tag", default="u64_h128_auto")
ap.add_argument("--lstm_tag", default="u64_h128_lstm")
ap.add_argument("--metric", default="pv", choices=["loss", "ema_loss", "pv"])
args = ap.parse_args()

art = Path("artifacts")

def load_logs(rnn, tag):
    train = pd.read_csv(art / f"train_{rnn}_{tag}_log.csv")
    evalf = art / f"eval_{rnn}_{tag}_log.csv"
    eval = pd.read_csv(evalf) if evalf.exists() else None
    return train, eval

train_g, eval_g = load_logs("gru", args.gru_tag)
train_l, eval_l = load_logs("lstm", args.lstm_tag)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# ---- training loss
ax[0].plot(train_g["episode"], train_g["ema_loss"], label="GRU", color="tab:blue")
ax[0].plot(train_l["episode"], train_l["ema_loss"], label="LSTM", color="tab:orange")
ax[0].set_title("EMA Training Loss")
ax[0].set_xlabel("Episode"); ax[0].set_ylabel("EMA loss")
ax[0].legend(); ax[0].grid(True, alpha=0.3)

# ---- evaluation PV
if eval_g is not None and eval_l is not None:
    ax[1].plot(eval_g["episode"], eval_g["eval_pv"], label="GRU", color="tab:blue")
    ax[1].plot(eval_l["episode"], eval_l["eval_pv"], label="LSTM", color="tab:orange")
    ax[1].set_title("Evaluation Present Value")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("PV")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)

plt.suptitle("GRU vs LSTM Training Comparison", fontsize=13, y=1.03)
plt.tight_layout()
out = art / "compare_gru_lstm.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
