# recurrent_q_net_v5.py
#
# V4 adds FactorisedFollowerQNet alongside the existing RecurrentQNet.
#
# FactorisedFollowerQNet
# ----------------------
# Shared encoder + shared GRU, then TWO independent dueling heads:
#   Q_d  (s, d)   — n_d_bins outputs
#   Q_rho(s, rho) — n_rho_bins outputs
#
# Action selection: argmax Q_d and argmax Q_rho independently.
# TD loss:         sum of Huber losses from both heads.
# Hidden state:    single GRU shared across both heads.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Original single-head net (leader uses this unchanged) ────────────────────
class RecurrentQNet(nn.Module):
    """Encoder -> {GRU|LSTM} -> (Dueling) Q head.
    Forward accepts [B, T, obs_dim] and an initial hidden state h0.
    Returns Q-values [B, T, act_dim] and next hidden state.
    """
    def __init__(self, obs_dim:int, act_dim:int, hidden:int=128,
                 rnn_type:str="gru", dueling:bool=True):
        super().__init__()
        self.dueling = dueling
        self.is_lstm = (rnn_type.lower() == "lstm")
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden),  nn.ReLU(),
        )
        if self.is_lstm:
            self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
        else:
            self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        if dueling:
            self.v = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.a = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, act_dim))
        else:
            self.q = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, act_dim))

    def init_hidden(self, batch_size:int, device:torch.device):
        h = torch.zeros(1, batch_size, self.rnn.hidden_size, device=device)
        if self.is_lstm:
            return (h, h.clone())
        return h

    def forward(self, obs:torch.Tensor, h0):
        x = self.feat(obs)
        y, hN = self.rnn(x, h0)
        if self.dueling:
            V = self.v(y)
            A = self.a(y)
            Q = V + (A - A.mean(dim=-1, keepdim=True))
        else:
            Q = self.q(y)
        return Q, hN


# ── Factorised follower net: shared trunk, two dueling heads ─────────────────
class FactorisedFollowerQNet(nn.Module):
    """Shared encoder + GRU for d-head; stateless MLP on tau-only for rho-head.

    V5 CHANGE: rho-head now receives ONLY the current normalised tau value
    (obs[:,:,3] = tau/tau_max), with NO connection to the GRU trunk.

    Why: the GRU trunk is dominated by the large d-reward signal (~60/step).
    Even with concat([gru_out, tau]), the GRU hidden state overwrites the
    tau signal. Removing GRU entirely from rho forces Q_rho to be a pure
    function of current tau, enabling tau-conditional retaliation threshold.

    Q_d  uses: encoder -> GRU -> dueling head     (history-aware)
    Q_rho uses: obs[:,3:4] -> 2-layer MLP -> dueling head  (stateless)
    """
    TAU_IDX = 3  # position of tau/tau_max in the normalised obs vector

    def __init__(self, obs_dim:int, n_d:int, n_rho:int,
                 hidden:int=128, rnn_type:str="gru",
                 tau_obs_idx:int=3,
                 rho_input_mode:str="tau_only"):
        """
        rho_input_mode:
            "tau_only"     (default/V5) — rho-head receives only current tau scalar.
                           Stateless MLP(1→64). Prevents GRU dominance overwriting
                           sparse retaliation signal. Used in A3/full V5.
            "gru"          — rho-head receives full GRU hidden output.
                           Same hidden dim as d-head. Used in A2 ablation.
            "tau_gru_size" — rho-head receives only tau scalar but uses the
                           same larger hidden dim (128) as the GRU path.
                           Used in A2b ablation to isolate tau-conditioning
                           from network capacity change.
        """
        super().__init__()
        self.n_d             = n_d
        self.n_rho           = n_rho
        self.is_lstm         = (rnn_type.lower() == "lstm")
        self.tau_obs_idx     = tau_obs_idx
        self.rho_input_mode  = rho_input_mode

        # ── d-head: full GRU path ────────────────────────────────────────────
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden),  nn.ReLU(),
        )
        if self.is_lstm:
            self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
        else:
            self.rnn = nn.GRU(hidden, hidden, batch_first=True)

        self.v_d = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.a_d = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_d))

        # ── rho-head: input and size depend on rho_input_mode ──────────────
        if rho_input_mode == "gru":
            # A2 ablation: rho-head receives full GRU output
            # Same hidden dim — GRU dominance is the failure mode being studied
            self.v_rho = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.a_rho = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_rho))
        elif rho_input_mode == "tau_gru_size":
            # A2b ablation: tau-only input but GRU-sized network (128 units)
            # Isolates contribution of tau-conditioning from network capacity
            self.v_rho = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.a_rho = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, n_rho))
        else:
            # tau_only (default/V5/A3): stateless MLP on tau scalar only
            # Deliberately small (64 units) — task is: map tau → rho Q-values
            rho_h = 64
            self.v_rho = nn.Sequential(nn.Linear(1, rho_h), nn.ReLU(), nn.Linear(rho_h, 1))
            self.a_rho = nn.Sequential(nn.Linear(1, rho_h), nn.ReLU(), nn.Linear(rho_h, n_rho))

    def init_hidden(self, batch_size:int, device:torch.device):
        h = torch.zeros(1, batch_size, self.rnn.hidden_size, device=device)
        if self.is_lstm:
            return (h, h.clone())
        return h

    def forward(self, obs:torch.Tensor, h0, add_tau_noise:bool=False):
        """
        Args:
            obs: [B, T, obs_dim]  obs[:,:,TAU_IDX] = tau/tau_max in [0,1]
            h0:  GRU/LSTM hidden state (only used by d-head)

        Returns:
            q_d:   [B, T, n_d]   — from GRU trunk
            q_rho: [B, T, n_rho] — from tau-only MLP, NO GRU
            hN:    next hidden state
        """
        # ── d-head (GRU) ─────────────────────────────────────────────────────
        x     = self.feat(obs)
        y, hN = self.rnn(x, h0)
        V_d   = self.v_d(y)
        A_d   = self.a_d(y)
        q_d   = V_d + (A_d - A_d.mean(dim=-1, keepdim=True))

        # ── rho-head: input depends on rho_input_mode ────────────────────────
        if self.rho_input_mode == "gru":
            # A2 ablation: use full GRU output as rho-head input
            rho_in = y                                             # [B, T, hidden]
        else:
            # tau_only and tau_gru_size: stateless — only current tau scalar
            rho_in = obs[:, :, self.tau_obs_idx:self.tau_obs_idx+1]  # [B, T, 1]
            # Explicit noise flag — only active during TD loss, never during eval
            if add_tau_noise:
                rho_in = rho_in + 0.02 * torch.randn_like(rho_in)
        V_rho  = self.v_rho(rho_in)                               # [B, T, 1]
        A_rho  = self.a_rho(rho_in)                               # [B, T, n_rho]
        q_rho  = V_rho + (A_rho - A_rho.mean(dim=-1, keepdim=True))

        return q_d, q_rho, hN


# ── Feedforward follower net (A0 ablation) ───────────────────────────────────
class FeedforwardFollowerQNet(nn.Module):
    """Pure MLP follower — no recurrence whatsoever.

    A0 ablation baseline: removes GRU entirely to isolate the contribution
    of recurrence. Joint 30-action space (6d × 5rho), shared reward — identical
    to A1 except no hidden state.

    Architecture:
        obs[6] → Linear(6→128) → ReLU → Linear(128→128) → ReLU
               → Dueling head → Q[n_d × n_rho]
    """
    def __init__(self, obs_dim: int, n_d: int, n_rho: int, hidden: int = 128):
        super().__init__()
        self.n_d    = n_d
        self.n_rho  = n_rho
        act_dim     = n_d * n_rho  # joint action space: 6×5=30

        self.feat = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden),  nn.ReLU(),
        )
        # Dueling head over joint action space
        self.v = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.a = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, act_dim))

    def init_hidden(self, batch_size: int, device: torch.device):
        """Returns None — no hidden state needed. Signature kept for compatibility."""
        return None

    def forward(self, obs: torch.Tensor, h0=None, **kwargs):
        """
        Args:
            obs: [B, T, obs_dim]
            h0:  ignored (no recurrence)
        Returns:
            Q:   [B, T, n_d*n_rho]
            h0:  None (no state to pass forward)
        """
        x = self.feat(obs)                                 # [B, T, hidden]
        V = self.v(x)                                      # [B, T, 1]
        A = self.a(x)                                      # [B, T, act_dim]
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q, None                                     # no hidden state
