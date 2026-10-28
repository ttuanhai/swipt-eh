"""Hyperparameters from Section IV of Abdolkhani et al., IEEE Commun. Lett. 2025."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    # --- Channel ---
    sigma2: float = 1e-3        # noise variance
    alpha: float = 4.0          # path-loss exponent
    d_sp: float = 1.8           # CIoT Tx -> PU Rx (m)
    d_ps: float = 1.8           # PU Tx -> CIoT Rx (m)
    d_ss: float = 1.5           # CIoT Tx -> CIoT Rx (m)

    # --- Time slots / primary user ---
    T: int = 30                 # slots per episode
    tau: float = 1.0            # slot duration (s)
    L: int = 18                 # slots occupied by the PU
    P_p: float = 0.2            # PU transmit power (W)
    I_th: float = 0.1           # interference threshold

    # --- Battery / energy harvesting ---
    B_max: float = 0.5
    # Initial battery. Not stated for Fig. 2a; its DDQN-UCB curve matches the B0 = 0, tau = 1 curve of Fig. 2c.
    B0: float = 0.0
    gamma_k: float = 0.5        # ambient energy ~ Gamma(k, beta)
    gamma_beta: float = 1.0
    mu: float = 0.9             # energy conversion efficiency

    # --- Action space: a = [rho, P_s] ---
    rho_levels: np.ndarray = field(default_factory=lambda: np.round(np.linspace(0.0, 1.0, 11), 2))
    p_levels: np.ndarray = field(default_factory=lambda: np.round(np.linspace(0.0, 0.1, 11), 3))

    phi: float = 7.0            # constraint-violation penalty

    # --- DDQN ---
    hidden1: int = 512
    hidden2: int = 128
    lr: float = 2e-4
    lr_step_episodes: int = 500  # lr halved every 500 episodes
    lr_decay: float = 0.5
    episodes: int = 2500
    batch_size: int = 80
    buffer_size: int = 10000     # paper: kappa = 333, which is unstable across seeds
    target_update: int = 200     # iterations between target syncs
    gamma: float = 0.99
    ucb_c: float = 2.5           # c'

    # --- Exploration ---
    exploration: str = "ucb"     # "ucb" (proposed), "egreedy", "greedy" or "random" (benchmarks)
    # The paper does not give its epsilon schedule; eps = max(eps_end, eps_start * eps_decay ** episode).
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995     # per episode

    normalize_state: bool = True
    seed: int = 0
    device: str = "cpu"

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def n_actions(self) -> int:
        return len(self.rho_levels) * len(self.p_levels)
