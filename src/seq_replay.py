import random
import numpy as np
import torch

class SequenceReplay:
    """Episode-wise buffer; samples contiguous chunks with burn-in + unroll."""
    def __init__(self, capacity_episodes:int=1000):
        self.capacity = capacity_episodes
        self.episodes = []  # each = dict(obs, act, rew, done)

    def push_episode(self, ep:dict):
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def sample(self, batch_size:int, burn_in:int, unroll:int, device:torch.device):
        assert len(self.episodes) > 0, "Replay is empty"
        B, U = burn_in, unroll
        O, A, R, D, M = [], [], [], [], []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            L = len(ep['obs'])
            assert L >= (B + U + 1), "Episode too short for requested chunk"
            start = random.randint(0, L - (B + U + 1))
            sl = slice(start, start + B + U + 1)
            o = np.stack(ep['obs'][sl])                 # [B+U+1, obs]
            a = np.array(ep['act'][start+B:start+B+U])  # [U]
            r = np.array(ep['rew'][start+B:start+B+U])  # [U]
            d = np.array(ep['done'][start+B:start+B+U]) # [U]
            m = np.ones_like(r, dtype=np.float32)       # [U]
            O.append(o); A.append(a); R.append(r); D.append(d); M.append(m)
        obs = torch.tensor(np.stack(O), dtype=torch.float32, device=device)      # [N, B+U+1, obs]
        acts= torch.tensor(np.stack(A), dtype=torch.long, device=device)         # [N, U]
        rews= torch.tensor(np.stack(R), dtype=torch.float32, device=device)      # [N, U]
        dones=torch.tensor(np.stack(D), dtype=torch.float32, device=device)      # [N, U]
        masks=torch.tensor(np.stack(M), dtype=torch.float32, device=device)      # [N, U]
        return obs, acts, rews, dones, masks