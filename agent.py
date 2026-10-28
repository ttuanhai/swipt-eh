"""DDQN agent with UCB exploration (Section III-B, eqs. (5)-(7), Algorithm 1)."""
import numpy as np
import torch
import torch.nn as nn

from config import Config


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, h1: int, h2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    @property
    def full(self) -> bool:
        return self.size == self.capacity

    def add(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


class DDQNUCBAgent:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.device = torch.device(cfg.device)
        self.n_actions = cfg.n_actions

        self.online = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden1, cfg.hidden2).to(self.device)
        self.target = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden1, cfg.hidden2).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.episodes_done = 0
        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.state_dim, rng)

        # UCB statistics: C_a and the running reward sum behind r_hat_a (eq. (7)).
        self.count = np.zeros(cfg.n_actions, dtype=np.int64)
        self.reward_sum = np.zeros(cfg.n_actions, dtype=np.float64)
        self.global_step = 0   # frame number * T + t
        self.train_iters = 0
        self.epsilon = cfg.eps_start

    def act(self, state: np.ndarray) -> int:
        self.global_step += 1
        if not self.buffer.full or self.cfg.exploration == "random":
            return int(self.rng.integers(self.n_actions))
        # Checked first so UCB runs draw no extra random numbers and stay reproducible.
        if self.cfg.exploration == "egreedy" and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        with torch.no_grad():
            q = self.online(torch.as_tensor(state, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        if self.cfg.exploration in ("egreedy", "greedy"):
            return int(np.argmax(q))

        # Eq. (6): U_a = r_hat_a + sqrt(c' ln t / C_a); untried actions get priority.
        tried = self.count > 0
        u = np.full(self.n_actions, np.inf)
        r_hat = self.reward_sum[tried] / self.count[tried]
        u[tried] = r_hat + np.sqrt(self.cfg.ucb_c * np.log(self.global_step) / self.count[tried])
        return int(np.argmax(q + u))

    def observe(self, action: int, reward: float):
        self.count[action] += 1
        self.reward_sum[action] += reward

    def learn(self) -> float | None:
        if not self.buffer.full or self.cfg.exploration == "random":
            return None
        s, a, r, s2, done = (torch.as_tensor(x, device=self.device)
                             for x in self.buffer.sample(self.cfg.batch_size))

        # Eq. (5): action chosen by the online net, evaluated by the target net.
        with torch.no_grad():
            a_star = self.online(s2).argmax(dim=1, keepdim=True)
            q_next = self.target(s2).gather(1, a_star).squeeze(1)
            y = r + self.cfg.gamma * (1.0 - done) * q_next
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_iters += 1
        if self.train_iters % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()

    def end_episode(self):
        # Step schedule: lr halved every `lr_step_episodes` episodes, counted from episode 1.
        self.episodes_done += 1
        lr = self.cfg.lr * self.cfg.lr_decay ** (self.episodes_done // self.cfg.lr_step_episodes)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.epsilon = max(self.cfg.eps_end, self.cfg.eps_start * self.cfg.eps_decay ** self.episodes_done)
