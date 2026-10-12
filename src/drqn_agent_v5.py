# drqn_agent_v5.py
#
# V4 adds FactorisedFollowerAgent alongside the unchanged DRQNAgent (leader).
#
# FactorisedFollowerAgent
# -----------------------
# Wraps FactorisedFollowerQNet. Maintains online + target networks.
# select_action: returns (d_idx, rho_idx) independently via argmax of each head.
# learn:         Double-DQN TD loss summed over both heads.
# Replay buffer: stores acts as {"d": d_idx, "rho": rho_idx} dicts.
#                seq_replay stores act as a 2-element array [d_idx, rho_idx].

from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from recurrent_q_net_v5 import RecurrentQNet, FactorisedFollowerQNet, FeedforwardFollowerQNet


# ── Leader agent (unchanged from v3) ─────────────────────────────────────────
@dataclass
class DRQNConfig:
    obs_dim:     int   = 6
    act_dim:     int   = 6
    hidden:      int   = 128
    rnn_type:    str   = "gru"
    dueling:     bool  = True
    gamma:       float = 0.98
    lr:          float = 3e-4
    clip_grad:   float = 1.0
    target_tau:  float = 0.015
    burn_in:     int   = 10
    unroll:      int   = 32


class DRQNAgent:
    def __init__(self, cfg: DRQNConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.online = RecurrentQNet(
            cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.rnn_type, cfg.dueling
        ).to(device)
        self.target = RecurrentQNet(
            cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.rnn_type, cfg.dueling
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt   = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.huber = nn.SmoothL1Loss(reduction="none")

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, h, epsilon: float):
        if torch.rand(()) < epsilon:
            return int(torch.randint(0, self.cfg.act_dim, (1,)).item()), h
        q, h2 = self.online(obs, h)
        return int(q.squeeze(0).squeeze(0).argmax().item()), h2

    def _soft_update(self):
        tau = self.cfg.target_tau
        for t_p, s_p in zip(self.target.parameters(), self.online.parameters()):
            t_p.data.copy_(tau * s_p.data + (1 - tau) * t_p.data)

    def hard_sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def sequence_td_loss(self, batch):
        gamma        = self.cfg.gamma
        burn, unroll = self.cfg.burn_in, self.cfg.unroll
        obs, acts, rews, dones, masks = batch
        N            = obs.size(0)
        obs_burn     = obs[:, :burn, :]
        obs_main     = obs[:, burn:, :]
        s_t          = obs_main[:, :-1, :]
        s_tp1        = obs_main[:, 1:,  :]
        h0           = self.online.init_hidden(N, self.device)
        with torch.no_grad():
            if burn > 0:
                _, h0 = self.online(obs_burn, h0)
        h0 = tuple(h.detach() for h in h0) if isinstance(h0, tuple) else h0.detach()
        Q_t,  h_t = self.online(s_t,   h0)
        with torch.no_grad():
            Q_tp1_online, _ = self.online(s_tp1, h_t)
            a_star          = Q_tp1_online.argmax(dim=-1)
            Q_tp1_target, _ = self.target(s_tp1, h_t)
            q_tp1           = Q_tp1_target.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)
            y               = rews + gamma * (1.0 - dones) * q_tp1
        q_pred = Q_t.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
        loss   = self.huber(q_pred, y)
        return (loss * masks).sum() / (masks.sum() + 1e-8)

    def learn(self, batch):
        loss = self.sequence_td_loss(batch)
        self.opt.zero_grad(); loss.backward()
        if self.cfg.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.clip_grad)
        self.opt.step(); self._soft_update()
        return float(loss.item())

    def eval_q(self, obs_t: torch.Tensor, h):
        with torch.no_grad():
            q, h2 = self.online(obs_t, h)
        return q.squeeze(0).squeeze(0).detach().cpu().numpy(), h2


# ── Factorised follower agent ─────────────────────────────────────────────────
@dataclass
class FactorisedFollowerConfig:
    obs_dim:         int   = 6
    n_d:             int   = 6     # number of depreciation bins
    n_rho:           int   = 5     # number of retaliation bins
    hidden:          int   = 128
    rnn_type:        str   = "gru"
    gamma:           float = 0.98
    lr:              float = 3e-4
    clip_grad:       float = 1.0
    target_tau:      float = 0.015
    burn_in:         int   = 10
    unroll:          int   = 32
    rho_input_mode:  str   = "tau_only"  # "tau_only" (A3/V5) | "gru" (A2) | "tau_gru_size" (A2b)
    tau_noise:       bool  = True        # σ=0.02 noise on τ input during training (tau_only modes only)


class FactorisedFollowerAgent:
    """
    DRQN follower with factorised action space.
    Two independent dueling Q heads over d-bins and rho-bins.
    The shared GRU trunk processes the full obs sequence once.

    Replay buffer contract:
        ep["act"] is a list of [d_idx, rho_idx] pairs (shape-2 arrays).
        seq_replay.sample() returns acts: [N, U, 2].
        learn() splits acts[:,:,0] for d and acts[:,:,1] for rho.

    select_action returns (d_idx, rho_idx).
    eval_q       returns (q_d_array, q_rho_array, next_hidden).
    """
    def __init__(self, cfg: FactorisedFollowerConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.online = FactorisedFollowerQNet(
            cfg.obs_dim, cfg.n_d, cfg.n_rho, cfg.hidden, cfg.rnn_type,
            rho_input_mode=cfg.rho_input_mode,
        ).to(device)
        self.target = FactorisedFollowerQNet(
            cfg.obs_dim, cfg.n_d, cfg.n_rho, cfg.hidden, cfg.rnn_type,
            rho_input_mode=cfg.rho_input_mode,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt   = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        # delta=10.0: Q-values reach ~100-800; default delta=1.0 clips gradients
        # outside ±1 causing systematic upward bias (Q-inflation)
        self.huber = nn.SmoothL1Loss(reduction="none", beta=10.0)

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, h, epsilon: float):
        """Returns (d_idx, rho_idx, next_hidden)."""
        if torch.rand(()) < epsilon:
            d_idx   = int(torch.randint(0, self.cfg.n_d,   (1,)).item())
            rho_idx = int(torch.randint(0, self.cfg.n_rho, (1,)).item())
            # Still need to advance hidden state for GRU continuity
            _, _, h2 = self.online(obs, h)
            return d_idx, rho_idx, h2
        q_d, q_rho, h2 = self.online(obs, h)
        d_idx   = int(q_d.squeeze(0).squeeze(0).argmax().item())
        rho_idx = int(q_rho.squeeze(0).squeeze(0).argmax().item())
        return d_idx, rho_idx, h2

    def _soft_update(self):
        tau = self.cfg.target_tau
        for t_p, s_p in zip(self.target.parameters(), self.online.parameters()):
            t_p.data.copy_(tau * s_p.data + (1 - tau) * t_p.data)

    def hard_sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def sequence_td_loss(self, batch):
        """
        batch from seq_replay where acts has shape [N, U, 2]:
            acts[:, :, 0] = d_idx
            acts[:, :, 1] = rho_idx
        V5: rews is [N, U, 2] with rews[:,:,0]=reward_d, rews[:,:,1]=reward_rho
        This gives each head its own reward signal, preventing the large
        export_gain from d drowning out the small spite signal for rho.
        Loss = Huber(Q_d, reward_d) + Huber(Q_rho, reward_rho).
        """
        gamma        = self.cfg.gamma
        burn, unroll = self.cfg.burn_in, self.cfg.unroll
        obs, acts, rews, dones, masks = batch
        # acts: [N, U, 2];  rews: [N, U, 2]
        acts_d   = acts[:, :, 0]   # [N, U]
        acts_rho = acts[:, :, 1]   # [N, U]
        # rews shape depends on replay: [N, U, 2] when decomposed
        if rews.dim() == 3 and rews.shape[-1] == 2:
            rews_d   = rews[:, :, 0]   # [N, U]
            rews_rho = rews[:, :, 1]   # [N, U]
        else:
            rews_d   = rews            # [N, U] fallback
            rews_rho = rews
        # masks/dones must be [N, U] — squeeze extra dims from ones_like(2D rews)
        if masks.dim() == 3: masks = masks[:, :, 0]
        if dones.dim() == 3: dones = dones[:, :, 0]
        N        = obs.size(0)
        obs_burn = obs[:, :burn, :]
        obs_main = obs[:, burn:,  :]
        s_t      = obs_main[:, :-1, :]
        s_tp1    = obs_main[:, 1:,  :]

        h0 = self.online.init_hidden(N, self.device)
        with torch.no_grad():
            if burn > 0:
                _, _, h0 = self.online(obs_burn, h0)
        h0 = tuple(h.detach() for h in h0) if isinstance(h0, tuple) else h0.detach()

        # Online Q at t — add_tau_noise controlled by config (default True)
        # Noise only active here during TD loss, never during action selection/eval
        qd_t, qrho_t, h_t = self.online(s_t, h0, add_tau_noise=self.cfg.tau_noise)  # [N, U, n_d/n_rho]

        with torch.no_grad():
            # Double-DQN: online selects action, target evaluates
            qd_tp1_on, qrho_tp1_on, _ = self.online(s_tp1, h_t)
            ad_star   = qd_tp1_on.argmax(dim=-1)             # [N, U]
            arho_star = qrho_tp1_on.argmax(dim=-1)           # [N, U]

            qd_tp1_tgt, qrho_tp1_tgt, _ = self.target(s_tp1, h_t)
            qd_next   = qd_tp1_tgt.gather(-1, ad_star.unsqueeze(-1)).squeeze(-1)
            qrho_next = qrho_tp1_tgt.gather(-1, arho_star.unsqueeze(-1)).squeeze(-1)

            # Per-head rewards — each head trained on its own signal
            y_d   = rews_d   + gamma * (1.0 - dones) * qd_next
            y_rho = rews_rho + gamma * (1.0 - dones) * qrho_next

        qd_pred   = qd_t.gather(-1,   acts_d.unsqueeze(-1)).squeeze(-1)
        qrho_pred = qrho_t.gather(-1, acts_rho.unsqueeze(-1)).squeeze(-1)

        loss_d   = (self.huber(qd_pred,   y_d)   * masks).sum() / (masks.sum() + 1e-8)
        loss_rho = (self.huber(qrho_pred, y_rho) * masks).sum() / (masks.sum() + 1e-8)
        return loss_d + loss_rho

    def learn(self, batch):
        loss = self.sequence_td_loss(batch)
        self.opt.zero_grad(); loss.backward()
        if self.cfg.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.clip_grad)
        self.opt.step(); self._soft_update()
        return float(loss.item())

    def eval_q(self, obs_t: torch.Tensor, h):
        """Returns (q_d_np, q_rho_np, next_hidden)."""
        with torch.no_grad():
            q_d, q_rho, h2 = self.online(obs_t, h)
        q_d_np   = q_d.squeeze(0).squeeze(0).detach().cpu().numpy()
        q_rho_np = q_rho.squeeze(0).squeeze(0).detach().cpu().numpy()
        return q_d_np, q_rho_np, h2


# ── Feedforward follower agent (A0 ablation) ─────────────────────────────────
@dataclass
class FeedforwardFollowerConfig:
    obs_dim:    int   = 6
    n_d:        int   = 6
    n_rho:      int   = 5
    hidden:     int   = 128
    gamma:      float = 0.98
    lr:         float = 3e-4
    clip_grad:  float = 1.0
    target_tau: float = 0.015


class FeedforwardFollowerAgent:
    """A0 ablation: pure MLP follower, no recurrence.

    Joint 30-action space (6d × 5rho), shared reward.
    Standard transition replay (no burn-in/unroll — no hidden state to warm up).
    Uses standard experience replay: stores single (obs, act, rew, next_obs, done) tuples.
    The sequence replay buffer is used with burn_in=0 and unroll=1 for compatibility.
    """
    def __init__(self, cfg: FeedforwardFollowerConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device
        act_dim     = cfg.n_d * cfg.n_rho  # 30 joint actions
        self.act_dim = act_dim

        self.online = FeedforwardFollowerQNet(
            cfg.obs_dim, cfg.n_d, cfg.n_rho, cfg.hidden).to(device)
        self.target = FeedforwardFollowerQNet(
            cfg.obs_dim, cfg.n_d, cfg.n_rho, cfg.hidden).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt   = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.huber = nn.SmoothL1Loss(reduction="none")

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, h, epsilon: float):
        """Returns (joint_idx, None) — no hidden state."""
        if torch.rand(()) < epsilon:
            return int(torch.randint(0, self.act_dim, (1,)).item()), None
        q, _ = self.online(obs, None)
        return int(q.squeeze(0).squeeze(0).argmax().item()), None

    def init_hidden(self, batch_size: int, device: torch.device):
        return None

    def _soft_update(self):
        tau = self.cfg.target_tau
        for t_p, s_p in zip(self.target.parameters(), self.online.parameters()):
            t_p.data.copy_(tau * s_p.data + (1 - tau) * t_p.data)

    def hard_sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def sequence_td_loss(self, batch):
        """TD loss over sequence batch. burn_in ignored (no hidden state).
        Handles both burn_in=0 and burn_in>0 from seq_replay gracefully.
        """
        gamma        = self.cfg.gamma
        obs, acts, rews, dones, masks = batch
        # obs: [N, burn+unroll+1, obs_dim]
        # acts/rews/dones/masks: [N, unroll] — already stripped of burn-in by seq_replay
        # s_t and s_tp1 must have same T as acts
        if acts.dim() == 3:   acts  = acts[:, :, 0]
        if rews.dim() == 3:   rews  = rews[:, :, 0]
        if masks.dim() == 3:  masks = masks[:, :, 0]
        if dones.dim() == 3:  dones = dones[:, :, 0]

        U     = acts.shape[1]                # unroll length
        # obs has burn+unroll+1 steps; last U+1 steps are the main window
        s_t   = obs[:, -U-1:-1, :]           # [N, U, obs_dim]
        s_tp1 = obs[:, -U:,     :]           # [N, U, obs_dim]

        Q_t,  _ = self.online(s_t,   None)
        with torch.no_grad():
            Q_tp1_on,  _ = self.online(s_tp1, None)
            a_star        = Q_tp1_on.argmax(dim=-1)
            Q_tp1_tgt, _  = self.target(s_tp1, None)
            q_next        = Q_tp1_tgt.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)
            y             = rews + gamma * (1.0 - dones) * q_next

        q_pred = Q_t.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
        loss   = self.huber(q_pred, y)
        return (loss * masks).sum() / (masks.sum() + 1e-8)

    def learn(self, batch):
        loss = self.sequence_td_loss(batch)
        self.opt.zero_grad(); loss.backward()
        if self.cfg.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.clip_grad)
        self.opt.step(); self._soft_update()
        return float(loss.item())
