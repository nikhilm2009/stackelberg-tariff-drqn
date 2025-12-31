from dataclasses import dataclass
import torch
import torch.nn as nn
from recurrent_q_net import RecurrentQNet

@dataclass
class DRQNConfig:
    obs_dim: int
    act_dim: int
    hidden: int = 128
    rnn_type: str = "gru"        # "gru" | "lstm"
    dueling: bool = True
    gamma: float = 0.97
    lr: float = 3e-4
    clip_grad: float = 1.0
    target_tau: float = 0.005     # soft update
    burn_in: int = 10
    unroll: int = 32

class DRQNAgent:
    def __init__(self, cfg:DRQNConfig, device:torch.device):
        self.cfg = cfg
        self.device = device
        self.online = RecurrentQNet(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.rnn_type, cfg.dueling).to(device)
        self.target = RecurrentQNet(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.rnn_type, cfg.dueling).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.huber = nn.SmoothL1Loss(reduction='none')

    @torch.no_grad()
    def select_action(self, obs:torch.Tensor, h, epsilon:float):
        # obs: [1,1,obs_dim] (batch=1, T=1)
        if torch.rand(()) < epsilon:
            return int(torch.randint(0, self.cfg.act_dim, (1,)).item()), h
        q, h2 = self.online(obs, h)
        a = int(q.squeeze(0).squeeze(0).argmax().item())
        return a, h2

    def _soft_update(self):
        tau = self.cfg.target_tau
        for t_params, s_params in zip(self.target.parameters(), self.online.parameters()):
            t_params.data.copy_(tau * s_params.data + (1 - tau) * t_params.data)

    def sequence_td_loss(self, batch):
        gamma = self.cfg.gamma
        burn, unroll = self.cfg.burn_in, self.cfg.unroll
        obs, acts, rews, dones, masks = batch  # obs [N,B+U+1,obs]
        N = obs.size(0)
        # Split into burn-in and learning segments
        obs_burn = obs[:, :burn, :]
        obs_main = obs[:, burn:, :]
        s_t   = obs_main[:, :-1, :]  # [N,U,obs]
        s_tp1 = obs_main[:, 1:,  :]
        # Initialize hidden
        h0 = self.online.init_hidden(N, self.device)
        with torch.no_grad():
            if burn > 0:
                _, h0 = self.online(obs_burn, h0)  # advance hidden (no loss)
        # Online Q at t
        Q_t, h_t = self.online(s_t, h0)   # [N,U,A]
        # Next state Q for Double-DQN
        with torch.no_grad():
            Q_tp1_online, _ = self.online(s_tp1, h_t)        # [N,U,A]
            a_star = Q_tp1_online.argmax(dim=-1)             # [N,U]
            Q_tp1_target, _ = self.target(s_tp1, h_t)
            q_tp1 = Q_tp1_target.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)  # [N,U]
            y = rews + gamma * (1.0 - dones) * q_tp1
        q_pred = Q_t.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
        loss = self.huber(q_pred, y)
        loss = (loss * masks).sum() / (masks.sum() + 1e-8)
        return loss

    def learn(self, batch):
        loss = self.sequence_td_loss(batch)
        self.opt.zero_grad()
        loss.backward()
        if self.cfg.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.clip_grad)
        self.opt.step()
        self._soft_update()
        return float(loss.item())
    
    def eval_q(self, obs_t, h):
        """Return Q-values (no epsilon), next hidden; obs_t: [1,1,obs]."""
        with torch.no_grad():
            q, h2 = self.online(obs_t, h)
        return q.squeeze(0).squeeze(0).detach().cpu().numpy(), h2
