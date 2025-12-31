import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentQNet(nn.Module):
    """Encoder -> {GRU|LSTM} -> (Dueling) Q head.
    Forward accepts [B, T, obs_dim] and an initial hidden state h0.
    Returns Q-values [B, T, act_dim] and next hidden state.
    """
    def __init__(self, obs_dim:int, act_dim:int, hidden:int=128, rnn_type:str="gru", dueling:bool=True):
        super().__init__()
        self.dueling = dueling
        self.is_lstm = (rnn_type.lower() == "lstm")
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden), nn.ReLU(),
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
        # obs: [B, T, obs_dim]
        x = self.feat(obs)
        y, hN = self.rnn(x, h0)
        if self.dueling:
            V = self.v(y)                     # [B,T,1]
            A = self.a(y)                     # [B,T,A]
            Q = V + (A - A.mean(dim=-1, keepdim=True))
        else:
            Q = self.q(y)
        return Q, hN