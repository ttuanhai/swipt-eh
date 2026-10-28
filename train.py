"""Train the DDQN-UCB agent (Algorithm 1) and reproduce the proposed curve of Fig. 2a."""
import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import DDQNUCBAgent
from config import Config
from env import SwiptCIoTEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=2500)
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--L", type=int, default=18)
    p.add_argument("--B0", type=float, default=0.0, help="initial battery (0.5 = B_max, 0.25 = B_max/2)")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--buffer", type=int, default=10000, help="replay buffer size (paper: kappa = 333)")
    p.add_argument("--explore", choices=("ucb", "egreedy", "greedy", "random"), default="ucb", help="exploration strategy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    return p.parse_args()


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot(metrics: dict, cfg: Config, path: Path, window: int = 50):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in [("sum_rate", "Sum rate (no penalty)"), ("return", "Return (with penalty)")]:
        y = moving_average(np.asarray(metrics[key]), window)
        ax.plot(np.arange(len(y)) + window, y, label=label)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Sum Rate (bit/s/Hz)")
    name = {"ucb": "DDQN-UCB", "egreedy": "DDQN-ε-greedy", "greedy": "DDQN-greedy", "random": "Random"}[cfg.exploration]
    ax.set_title(f"{name} (T={cfg.T}, L={cfg.L}, B0={cfg.B0}, tau={cfg.tau})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(cfg: Config, out: Path):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = SwiptCIoTEnv(cfg)
    agent = DDQNUCBAgent(cfg, rng)

    metrics = {k: [] for k in ("episode", "return", "sum_rate", "violations", "loss", "lr")}
    start = time.time()
    for ep in range(1, cfg.episodes + 1):
        s, _ = env.reset(seed=cfg.seed if ep == 1 else None)
        ep_return = ep_rate = 0.0
        violations = 0
        losses = []
        done = False
        while not done:
            a = agent.act(s)
            s2, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            agent.observe(a, r)
            agent.buffer.add(s, a, r, s2, float(terminated))
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            ep_return += r
            ep_rate += info["rate"]
            violations += info["violation"]
            s = s2
        lr = agent.optimizer.param_groups[0]["lr"]
        agent.end_episode()

        metrics["episode"].append(ep)
        metrics["return"].append(ep_return)
        metrics["sum_rate"].append(ep_rate)
        metrics["violations"].append(violations)
        metrics["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        metrics["lr"].append(lr)

        if ep % 100 == 0 or ep == cfg.episodes:
            last = slice(-100, None)
            print(f"ep {ep:5d} | sum rate {np.mean(metrics['sum_rate'][last]):7.2f} "
                  f"| return {np.mean(metrics['return'][last]):7.2f} "
                  f"| violations {np.mean(metrics['violations'][last]):5.2f} "
                  f"| lr {lr:.1e} | {time.time() - start:6.1f}s", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(metrics.keys())
        w.writerows(zip(*metrics.values()))
    torch.save(agent.online.state_dict(), out / "ddqn_ucb.pt")
    plot(metrics, cfg, out / "asr.png")
    print(f"Saved results to {out.resolve()}")
    return metrics


def main():
    args = parse_args()
    cfg = Config(episodes=args.episodes, T=args.T, L=args.L, B0=args.B0, tau=args.tau,
                 buffer_size=args.buffer, exploration=args.explore, seed=args.seed, device=args.device)
    train(cfg, Path(args.out))


if __name__ == "__main__":
    main()
