# stackelberg_q3_tariff_econ_sim_drqn.py
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import random
from collections import Counter

SIM_VERSION = "econ-sim v3 — deferred follower + (d_last, X_momentum) in follower state"

# -------------------------
# Economic parameter bundle
# -------------------------
@dataclass
class EconomicParams:
    M0: float = 300.0
    X0: float = 300.0
    demand_elast: float = 1.5
    supply_elast: float = 1.2
    kappa: float = 0.4
    lam: float = 0.35
    delta: float = 0.97
    horizon: int = 1
    phi_inflation: float = 0.20
    leader_cost_w: float = 0.0
    follower_cost_w: float = 0.0
    tau_max: float = 0.35
    d_max: float = 0.35

# -------------------------
# Economic environment
# -------------------------
class EconomicEnvironment:
    def __init__(self, params: EconomicParams):
        self.p = params
        self.reset()

    def reset(self):
        self.M = self.p.M0
        self.X = self.p.X0
        # for momentum (set/updated by orchestrator)
        self.prev_X = self.X

    def _M_star(self, tau: float) -> float:
        # Import demand falls with tariffs
        return max(0.0, self.p.M0 * (1.0 - self.p.demand_elast * tau))

    def _X_star(self, d: float) -> float:
        # Exports respond to depreciation
        return self.p.X0 * (1.0 + self.p.supply_elast * d)

    def _transition(self, tau: float, d: float):
        M_star = self._M_star(tau)
        X_star = self._X_star(d)
        self.M = (1 - self.p.kappa) * self.M + self.p.kappa * M_star
        self.X = (1 - self.p.lam)   * self.X + self.p.lam   * X_star

    def leader_period_payoff_with_components(self, tau: float):
        rev = tau * self.M
        cons_loss = 0.5 * tau * (self.p.M0 - self.M)
        action_cost_L = self.p.leader_cost_w * (tau ** 2)
        L = rev - cons_loss - action_cost_L
        return L, rev, cons_loss, action_cost_L

    def follower_period_payoff_with_components(self, d: float):
        export_gain = (self.X - self.p.X0)
        infl_cost   = self.p.phi_inflation * d * self.p.M0
        action_cost_F = self.p.follower_cost_w * (d ** 2)
        F = export_gain - infl_cost - action_cost_F
        return F, export_gain, infl_cost, action_cost_F

    def step(self, tau: float, d: float):
        L, rev, cons_loss, action_cost_L = self.leader_period_payoff_with_components(tau)
        F, export_gain, infl_cost, action_cost_F = self.follower_period_payoff_with_components(d)
        self._transition(tau, d)
        diag = {
            "M": self.M, "X": self.X,
            "rev": rev, "cons_loss": cons_loss, "action_cost_L": action_cost_L,
            "export_gain": export_gain, "infl_cost": infl_cost, "action_cost_F": action_cost_F
        }
        return L, F, diag

    # For BR curve (post-transition effect of d on X)
    def evaluate_follower_payoff(self, tau: float, d: float, timing: str = "post") -> float:
        M_bak, X_bak = self.M, self.X
        prev_bak = getattr(self, "prev_X", self.X)
        try:
            if timing == "pre":
                F, *_ = self.follower_period_payoff_with_components(d)
                return F
            elif timing == "post":
                self._transition(tau, d)
                F, *_ = self.follower_period_payoff_with_components(d)
                return F
            else:
                raise ValueError("timing must be 'pre' or 'post'")
        finally:
            self.M, self.X = M_bak, X_bak
            self.prev_X = prev_bak

# -------------------------
# Utility: uniform bins
# -------------------------
def make_bins(n: int, v_min: float, v_max: float) -> List[float]:
    if n <= 1:
        return [v_min]
    step = (v_max - v_min) / (n - 1)
    return [v_min + i * step for i in range(n)]

# -------------------------------------------
# Best-Response follower (over depreciation d)
# -------------------------------------------
class BestResponseFollowerEconomic:
    def __init__(self, d_bins: List[float]):
        self.d_bins = d_bins
        self.last_action = 0.0

    def respond(self, tau: float, env: EconomicEnvironment) -> float:
        best_d = self.d_bins[0]
        best_val = float("-inf")
        for d in self.d_bins:
            F = env.evaluate_follower_payoff(tau, d, timing="post")
            if F > best_val:
                best_val = F
                best_d = d
        self.last_action = best_d
        return best_d

# ------------------------------------------
# Q3-binned Leader: state = (last 3 d-bins, last τ, Mbin, Xbin)
# ------------------------------------------
class Q3BinnedLeader:
    def __init__(self, tau_bins: List[float], epsilon=0.15, gamma=0.95, alpha=0.2, q_init: float = 0.0):
        self.tau_bins = tau_bins
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.history_f = [f"{0.00:.2f}"] * 3   # follower d-bins as strings
        self.tau_last = f"{self.tau_bins[0]:.2f}"  # include last τ in state
        self.q_init = q_init  # optimistic init (0.0 keeps old behavior)

        # Q-table keyed by (state=(d1,d2,d3,τ_last,Mbin,Xbin), action)
        self.q: Dict[Tuple[Tuple[str, str, str, str, str, str], float], float] = {}
        self.last_state: Optional[Tuple[str, str, str, str, str, str]] = None
        self.last_action: Optional[float] = None
        self.total_payoff = 0.0
        self.env = None  # set by orchestrator

        # Diagnostics
        self.state_visits = Counter()
        self.greedy_flags: List[bool] = []
        self.epsilon_trace: List[float] = []
        self.qgap_trace: List[float] = []

        # Per-round Q snapshots (for plots)
        self.q_max_trace: List[float] = []
        self.q_avg_trace: List[float] = []
        self.state_trace: List[Tuple[str, str, str, str, str, str]] = []

        # Adaptive step sizes per (s,a)
        self.sa_visits: Dict[Tuple[Tuple[str, str, str, str, str, str], float], int] = {}

    # --- coarse binning for M and X (relative to baselines) ---
    @staticmethod
    def _coarse_bin(val: float, base: float) -> str:
        # thresholds tuned to help with corner/transition regimes
        ratio = 0.0 if base == 0 else val / base
        if ratio < 0.8:   return "L"
        if ratio < 1.2:   return "M"
        return "H"

    def set_env(self, env) -> None:
        self.env = env

    def _state(self) -> Tuple[str, str, str, str, str, str]:
        if self.env is None:
            Mbin = "?"
            Xbin = "?"
        else:
            Mbin = self._coarse_bin(self.env.M, self.env.p.M0)
            Xbin = self._coarse_bin(self.env.X, self.env.p.X0)
        return tuple(self.history_f) + (self.tau_last, Mbin, Xbin)

    def _q_vals_for(self, s: Tuple[str, str, str, str, str, str]) -> Dict[float, float]:
        default = self.q_init
        return {a: self.q.get((s, a), default) for a in self.tau_bins}

    def decide_tariff(self, follower_last_d: Optional[float] = None) -> float:
        # incorporate the observed last follower d into memory BEFORE choosing τ_t
        if follower_last_d is not None:
            self.history_f.pop(0)
            self.history_f.append(f"{follower_last_d:.2f}")

        s = self._state()
        self.state_visits[s] += 1

        vals = self._q_vals_for(s)

        # Snapshot Q BEFORE selecting action (for time series)
        if vals:
            max_q_snapshot = max(vals.values())
            avg_q_snapshot = sum(vals.values()) / len(vals)
        else:
            max_q_snapshot = 0.0
            avg_q_snapshot = 0.0
        self.state_trace.append(s)
        self.q_max_trace.append(max_q_snapshot)
        self.q_avg_trace.append(avg_q_snapshot)

        # ε-greedy choice
        if vals:
            max_q = max(vals.values())
            argmax_actions = [a for a, v in vals.items() if v == max_q]
        else:
            argmax_actions = self.tau_bins[:]

        explore = (random.random() < self.epsilon)
        if explore:
            a = random.choice(self.tau_bins)
        else:
            a = random.choice(argmax_actions)

        # diagnostics
        self.epsilon_trace.append(self.epsilon)
        self.greedy_flags.append(a in argmax_actions)
        if len(vals) >= 2:
            sorted_q = sorted(vals.values(), reverse=True)
            self.qgap_trace.append(sorted_q[0] - sorted_q[1])
        else:
            self.qgap_trace.append(0.0)

        # remember chosen τ in state
        self.tau_last = f"{a:.2f}"
        # decay epsilon
        self.epsilon = max(0.02, self.epsilon * 0.995)

        self.last_state = s
        self.last_action = a
        return a

    def update(self, follower_new_d: float, reward: float):
        s = self.last_state
        a = self.last_action

        # incorporate new follower action into next state
        self.history_f.pop(0)
        self.history_f.append(f"{follower_new_d:.2f}")
        s_next = self._state()

        old_q = self.q.get((s, a), self.q_init)
        max_next = max(self.q.get((s_next, ap), self.q_init) for ap in self.tau_bins)

        key = (s, a)
        self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
        alpha = 1.0 / (self.sa_visits[key] ** 0.5)
        alpha = max(0.02, alpha)  # consider 0.05 for very wide action ranges

        new_q = old_q + alpha * (reward + self.gamma * max_next - old_q)
        self.q[(s, a)] = new_q
        self.total_payoff += reward

# ---------------------------------------------
# Q3-binned Follower
# State = (last 3 τ, Xbin, d_last_bin, X_momentum)
# Optional Double-Q to reduce overestimation
# ---------------------------------------------
class Q3BinnedFollower:
    def __init__(self, d_bins: List[float], epsilon=0.15, gamma=0.95, alpha=0.2,
                 double_q: bool = False, q_init_f: float = 0.0):
        self.d_bins = d_bins
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.q_init_f = q_init_f
        self.history_l = [f"{0.00:.2f}"] * 3  # leader tau-bins as strings
        self.last_state: Optional[Tuple[str, str, str, str, str, str]] = None
        self.last_action: Optional[float] = None
        self.total_payoff = 0.0
        self.last_d_str = f"{self.d_bins[0]:.2f}"  # for state

        self.double_q = double_q
        if self.double_q:
            self.qA: Dict[Tuple[Tuple[str, str, str, str, str, str], float], float] = {}
            self.qB: Dict[Tuple[Tuple[str, str, str, str, str, str], float], float] = {}
            self.q:  Dict[Tuple[Tuple[str, str, str, str, str, str], float], float] = {}  # public average
        else:
            self.q:  Dict[Tuple[Tuple[str, str, str, str, str, str], float], float] = {}

        # Diagnostics
        self.state_visits = Counter()
        self.greedy_flags: List[bool] = []
        self.epsilon_trace: List[float] = []
        self.qgap_trace: List[float] = []

        # Per-round Q snapshots (for plots)
        self.q_max_trace: List[float] = []
        self.q_avg_trace: List[float] = []
        self.state_trace: List[Tuple[str, str, str, str, str, str]] = []

        # Adaptive step sizes per (s,a)
        self.sa_visits: Dict[Tuple[Tuple[str, str, str, str, str, str], float], int] = {}

        # env handle for X-binning & momentum (set by orchestrator)
        self.env = None

    # ----- helpers -----
    @staticmethod
    def _coarse_bin(val: float, base: float) -> str:
        # widen the 'M' band so states are not almost always 'H'
        ratio = 0.0 if base == 0 else val / base
        if ratio < 0.90:   return "L"
        if ratio < 1.20:   return "M"
        return "H"

    def set_env(self, env) -> None:
        self.env = env

    def _x_momentum(self) -> str:
        if self.env is None or not hasattr(self.env, "prev_X"):
            return "UNK"
        dx = self.env.X - self.env.prev_X
        if dx > 1e-6:
            return "UP"
        if dx < -1e-6:
            return "DN"
        return "FLAT"

    def _state(self) -> Tuple[str, str, str, str, str, str]:
        if self.env is None:
            Xbin = "?"
            mom = "UNK"
        else:
            Xbin = self._coarse_bin(self.env.X, self.env.p.X0)
            mom = self._x_momentum()
        return tuple(self.history_l) + (Xbin, self.last_d_str, mom)

    def _q_val(self, s, a: float) -> float:
        if not self.double_q:
            return self.q.get((s, a), self.q_init_f)
        return 0.5 * (self.qA.get((s, a), self.q_init_f) + self.qB.get((s, a), self.q_init_f))

    def _q_vals_for(self, s) -> Dict[float, float]:
        return {a: self._q_val(s, a) for a in self.d_bins}

    # ----- policy -----
    def respond(self, leader_tau: float) -> float:
        # advance memory with the NEW leader action
        self.history_l.pop(0)
        self.history_l.append(f"{leader_tau:.2f}")
        s = self._state()
        self.state_visits[s] += 1

        vals = self._q_vals_for(s)

        # Snapshot Q BEFORE selecting action (for time series)
        if vals:
            max_q_snapshot = max(vals.values())
            avg_q_snapshot = sum(vals.values()) / len(vals)
        else:
            max_q_snapshot = 0.0
            avg_q_snapshot = 0.0
        self.state_trace.append(s)
        self.q_max_trace.append(max_q_snapshot)
        self.q_avg_trace.append(avg_q_snapshot)

        # ε-greedy
        if vals:
            max_q = max(vals.values())
            argmax_actions = [a for a, v in vals.items() if v == max_q]
        else:
            argmax_actions = self.d_bins[:]

        explore = (random.random() < self.epsilon)
        if explore:
            a = random.choice(self.d_bins)
        else:
            a = random.choice(argmax_actions)

        # diagnostics
        self.epsilon_trace.append(self.epsilon)
        self.greedy_flags.append(a in argmax_actions)
        if len(vals) >= 2:
            sorted_q = sorted(vals.values(), reverse=True)
            self.qgap_trace.append(sorted_q[0] - sorted_q[1])
        else:
            self.qgap_trace.append(0.0)

        # remember last chosen d (as bin string) for state
        self.last_d_str = f"{a:.2f}"
        # decay epsilon
        self.epsilon = max(0.02, self.epsilon * 0.995)

        self.last_state = s
        self.last_action = a
        return a

    # ----- learning -----
    def update(self, leader_next_tau: float, reward: float):
        # Called one step later by orchestrator; env.X has progressed to X_t and leader_next_tau = τ_t
        s = self.last_state
        a = self.last_action

        if self.env is None:
            Xbin = "?"
            mom = "UNK"
        else:
            Xbin = self._coarse_bin(self.env.X, self.env.p.X0)
            mom = self._x_momentum()
        # s_next = (τ_{t-1}, τ_t, τ_{t+1}, Xbin_t, d_last, Xmom_t)
        s_next = (self.history_l[1], self.history_l[2], f"{leader_next_tau:.2f}", Xbin, self.last_d_str, mom)

        if not self.double_q:
            # Standard Q-learning with optimistic default
            old_q = self.q.get((s, a), self.q_init_f)
            max_next = max(self.q.get((s_next, ap), self.q_init_f) for ap in self.d_bins)

            key = (s, a)
            self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
            alpha = 1.0 / (self.sa_visits[key] ** 0.5)
            alpha = max(0.05, alpha)  # stronger floor for follower

            new_q = old_q + alpha * (reward + self.gamma * max_next - old_q)
            self.q[(s, a)] = new_q
        else:
            # Double Q-learning (Hasselt) with optimistic defaults & floor
            if random.random() < 0.5:
                argmax_a = max(self.d_bins, key=lambda ap: self.qA.get((s_next, ap), self.q_init_f))
                target   = reward + self.gamma * self.qB.get((s_next, argmax_a), self.q_init_f)
                old_q    = self.qA.get((s, a), self.q_init_f)

                key = (s, a)
                self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
                alpha = 1.0 / (self.sa_visits[key] ** 0.5)
                alpha = max(0.05, alpha)

                self.qA[(s, a)] = old_q + alpha * (target - old_q)
            else:
                argmax_a = max(self.d_bins, key=lambda ap: self.qB.get((s_next, ap), self.q_init_f))
                target   = reward + self.gamma * self.qA.get((s_next, argmax_a), self.q_init_f)
                old_q    = self.qB.get((s, a), self.q_init_f)

                key = (s, a)
                self.sa_visits[key] = self.sa_visits.get(key, 0) + 1
                alpha = 1.0 / (self.sa_visits[key] ** 0.5)
                alpha = max(0.05, alpha)

                self.qB[(s, a)] = old_q + alpha * (target - old_q)

            # Keep public average table in sync for diagnostics/plots
            self.q[(s, a)] = 0.5 * (self.qA.get((s, a), self.q_init_f) + self.qB.get((s, a), self.q_init_f))

        self.total_payoff += reward

# ------------------------------------
# Orchestrator (economic Stackelberg)
# ------------------------------------
class StackelbergTariffGameEconomic:
    def __init__(self, env: EconomicEnvironment, leader, follower, track: bool = True):
        self.env = env
        self.leader = leader
        self.follower = follower
        # give agents env handles for binning if supported
        if hasattr(self.leader, "set_env"):
            self.leader.set_env(self.env)
        if hasattr(self.follower, "set_env"):
            self.follower.set_env(self.env)
        self.track = track
        self.results = {"rounds": []} if track else None
        self.t = 0
        # stash follower reward to update after next τ is chosen (deferred update)
        self._pending_f_reward = None
        # track previous X for follower momentum feature
        self.env.prev_X = self.env.X

    def run(self, rounds: int = 300):
        for _ in range(rounds):
            self.step()

    def step(self):
        self.t += 1
        follower_last_d = getattr(self.follower, "last_action", None)

        # Leader moves (observes last d)
        tau_t = self.leader.decide_tariff(follower_last_d)

        # Complete follower's update for previous step using τ_t as "next τ"
        if isinstance(self.follower, Q3BinnedFollower) and self._pending_f_reward is not None:
            self.follower.update(leader_next_tau=tau_t, reward=self._pending_f_reward)
            self._pending_f_reward = None

        # Follower responds (BR needs env, Q-agent just τ)
        if isinstance(self.follower, BestResponseFollowerEconomic):
            d_t = self.follower.respond(tau_t, self.env)
        else:
            d_t = self.follower.respond(tau_t)

        # Environment transition and payoffs (now at time t)
        # keep momentum reference BEFORE step updates X
        prev_X_for_mom = self.env.X
        L_pay, F_pay, diag = self.env.step(tau_t, d_t)
        # expose X momentum info to follower for s and s'
        self.env.prev_X = prev_X_for_mom

        # Learn
        if isinstance(self.leader, Q3BinnedLeader):
            self.leader.update(follower_new_d=d_t, reward=L_pay)

        # Defer follower update to next step so that s' includes τ_t
        if isinstance(self.follower, Q3BinnedFollower):
            self._pending_f_reward = F_pay

        # Logging
        if self.track:
            self.results["rounds"].append({
                "round": self.t,
                "tau": tau_t, "d": d_t,
                "M": diag.get("M"), "X": diag.get("X"),
                "leader_pay": L_pay, "follower_pay": F_pay,
                "rev": diag.get("rev"), "cons_loss": diag.get("cons_loss"),
                "action_cost_L": diag.get("action_cost_L"),
                "export_gain": diag.get("export_gain"),
                "infl_cost": diag.get("infl_cost"),
                "action_cost_F": diag.get("action_cost_F"),
            })

def make_env(episode_len: int = 256):
    """
    Gym-style wrapper so DRQN can train on EconomicEnvironment with fixed horizon.
    The leader acts (τ), follower plays BR. Returns obs, reward, done, info.
    """
    from stackelberg_q3_tariff_econ_sim_drqn import (
        EconomicParams, EconomicEnvironment, make_bins, BestResponseFollowerEconomic
    )

    p = EconomicParams()
    p.horizon = episode_len  # ensure env runs long enough
    base = EconomicEnvironment(p)

    class TariffEnv:
        def __init__(self, env: EconomicEnvironment):
            self.env = env
            self.t = 0
            self._last_tau = 0.0
            self._last_d = 0.0
            self._d_bins = make_bins(6, 0.0, self.env.p.d_max)
            self._br = BestResponseFollowerEconomic(self._d_bins)

        def _obs(self):
            return [self.env.M, self.env.X, self._last_tau, self._last_d]

        def reset(self, seed=None):
            if seed is not None:
                import random, numpy as np
                random.seed(seed); np.random.seed(seed)
            self.env.reset()
            self.t = 0
            self._last_tau = 0.0
            self._last_d = 0.0
            return self._obs()

        def step(self, tau: float):
            d = self._br.respond(tau, self.env)
            L_pay, F_pay, diag = self.env.step(tau, d)
            self._last_tau, self._last_d = tau, d
            self.t += 1
            done = (self.t >= self.env.p.horizon)
            return self._obs(), float(L_pay), bool(done), diag

    return TariffEnv(base)