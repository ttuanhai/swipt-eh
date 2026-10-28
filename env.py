"""SWIPT-EH CIoT environment (Section II and reward eq. (4) of the paper)."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config


class SwiptCIoTEnv(gym.Env):
    """One CIoT Tx-Rx pair sharing a channel with one PU Tx-Rx pair (underlay).

    State  s_t = [B_t, e_{t-1}, omega_p, g_ps, g_sp, g_ss]
    Action a_t = [rho_t, P_s], encoded as idx = i_rho * len(p_levels) + i_p
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # The paper writes lambda_ij = d_ij^-alpha as the exponential rate, so the mean
        # gain is d_ij^alpha. This reading reproduces the ASR scale of Fig. 2a/2b.
        self.mean_ps = cfg.d_ps ** cfg.alpha
        self.mean_sp = cfg.d_sp ** cfg.alpha
        self.mean_ss = cfg.d_ss ** cfg.alpha
        self.mean_e = cfg.mu * cfg.gamma_k * cfg.gamma_beta
        self.n_p = len(cfg.p_levels)

        self.observation_space = spaces.Box(0.0, np.inf, shape=(cfg.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(cfg.n_actions)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self.cfg
        self.t = 0
        self.B = cfg.B0
        self.e_prev = 0.0
        # PU occupies L of the T slots; positions are not given in the paper, drawn per episode.
        self.omega = np.zeros(cfg.T, dtype=np.int64)
        self.omega[self.np_random.choice(cfg.T, size=cfg.L, replace=False)] = 1
        self._sample_gains()
        return self._state(), {}

    def decode(self, idx: int) -> tuple[float, float]:
        return self.cfg.rho_levels[idx // self.n_p], self.cfg.p_levels[idx % self.n_p]

    def step(self, action: int):
        cfg = self.cfg
        rho, P = self.decode(int(action))
        omega = self.omega[self.t]
        e_t = cfg.mu * self.np_random.gamma(cfg.gamma_k, cfg.gamma_beta)

        tx_energy = (1.0 - rho) * P * cfg.tau
        transmits = tx_energy > 0
        interference = omega == 1 and transmits and P * self.g_sp > cfg.I_th
        battery_short = tx_energy > self.B
        violation = interference or battery_short

        if interference:
            # Transmission is blocked: no rate, no transmit energy spent.
            rate, tx_energy = 0.0, 0.0
        else:
            if battery_short:
                # Not enough energy for the whole (1 - rho) * tau window: transmit at P until the battery is empty.
                tx_energy = self.B
            tx_time = tx_energy / P if transmits else 0.0
            sinr = P * self.g_ss / (omega * cfg.P_p * self.g_ps + cfg.sigma2)
            rate = tx_time * np.log2(1.0 + sinr)
        # Any violation is penalised (eq. (4)); data sent before the battery ran out still counts in the sum rate.
        reward = -cfg.phi if violation else rate

        self.B = min(self.B + rho * e_t * cfg.tau - tx_energy, cfg.B_max)
        self.e_prev = e_t
        self.t += 1
        # The task has a fixed horizon of T slots, so its end is a true termination.
        terminated = self.t >= cfg.T
        if not terminated:
            self._sample_gains()
        info = {"rate": float(rate), "violation": bool(violation)}
        return self._state(), float(reward), terminated, False, info

    def _sample_gains(self):
        self.g_ps = self.np_random.exponential(self.mean_ps)
        self.g_sp = self.np_random.exponential(self.mean_sp)
        self.g_ss = self.np_random.exponential(self.mean_ss)

    def _state(self) -> np.ndarray:
        omega = self.omega[min(self.t, self.cfg.T - 1)]
        s = np.array([self.B, self.e_prev, omega, self.g_ps, self.g_sp, self.g_ss], dtype=np.float32)
        if self.cfg.normalize_state:
            s /= np.array([self.cfg.B_max, self.mean_e, 1.0, self.mean_ps, self.mean_sp, self.mean_ss],
                          dtype=np.float32)
        return s
