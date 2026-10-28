"""Compare groups of training runs (e.g. buffer sizes), each run with several seeds.

Writes two figures:
  <out>_seeds.png  one panel per group, one line per seed
  <out>_mean.png   each group's mean over seeds, with a min-max band
                   (--report: +-1 std band, no title, legend = group label only, 300 dpi)
Curves are smoothed with a moving average, or with --ema BETA a zero-initialised EMA as in the paper.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import moving_average

# Categorical slots in fixed order: seed i (and group i) always gets slot i, in every panel.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
METRICS = {"sum_rate": "Average Sum Rate (bit/s/Hz)", "return": "Return with penalties (bit/s/Hz)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", nargs="+", action="append", required=True, metavar="LABEL RUN_DIR",
                   help="a label followed by its run directories (each holding metrics.csv); repeat per group")
    p.add_argument("--out", required=True, help="output prefix, e.g. results/compare")
    p.add_argument("--metric", choices=METRICS, default="sum_rate")
    p.add_argument("--title", default="DDQN-UCB")
    p.add_argument("--window", type=int, default=50, help="moving-average window (ignored with --ema)")
    p.add_argument("--ema", type=float, metavar="BETA", help="smooth with a zero-initialised EMA instead, e.g. 0.99")
    p.add_argument("--report", action="store_true", help="clean mean figure for a report")
    args = p.parse_args()
    for g in args.group:
        if len(g) < 2:
            p.error(f"--group {g[0]} needs at least one run directory")
    if len(args.group) > len(SERIES):
        p.error(f"at most {len(SERIES)} groups")
    return args


def load(run: Path) -> dict[str, np.ndarray]:
    with open(run / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("sum_rate", "return", "violations")}


def smooth(y, window, ema=None):
    """(episodes, values) of the smoothed curve."""
    if ema is None:
        y = moving_average(y, window)
        return np.arange(len(y)) + window, y
    # The EMA starts from 0 at episode 0, so the curve rises from the origin like the paper's Fig. 2.
    s = np.zeros(len(y) + 1)
    for t, v in enumerate(y, start=1):
        s[t] = ema * s[t - 1] + (1 - ema) * v
    return np.arange(len(s)), s


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_2)
    ax.set_xlabel("Episodes", color=INK_2)


def print_table(groups):
    print(f"{'group':14s} {'run':8s} {'sum rate':>9s} {'return':>8s} {'viol':>6s}   (last 500 episodes)")
    for label, runs in groups:
        last = {k: np.array([m[k][-500:].mean() for _, m in runs]) for k in ("sum_rate", "return", "violations")}
        for i, (run, _) in enumerate(runs):
            print(f"{label:14s} {run.name:8s} {last['sum_rate'][i]:9.2f} {last['return'][i]:8.2f} "
                  f"{last['violations'][i]:6.2f}")
        print(f"{label:14s} {'mean':8s} {last['sum_rate'].mean():9.2f} {last['return'].mean():8.2f} "
              f"{last['violations'].mean():6.2f}")
        print(f"{label:14s} {'std':8s} {last['sum_rate'].std():9.2f} {last['return'].std():8.2f} "
              f"{last['violations'].std():6.2f}")


def plot_seeds(groups, metric, window, title, path, ema=None):
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4.3), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (label, runs) in zip(axes[0], groups):
        style(ax)
        # Too many seeds to tell apart by color: draw them all in one muted color, unlabelled.
        colored = len(runs) <= len(SERIES)
        for i, (run, m) in enumerate(runs):
            x, y = smooth(m[metric], window, ema)
            if colored:
                ax.plot(x, y, color=SERIES[i], linewidth=1.3, label=run.name)
            else:
                ax.plot(x, y, color=MUTED, linewidth=0.8, alpha=0.6)
        ax.set_title(label if colored else f"{label} ({len(runs)} seeds)", color=INK)
    axes[0][0].set_ylabel(METRICS[metric], color=INK_2)
    if axes[0][0].get_legend_handles_labels()[0]:
        axes[0][0].legend(loc="lower right", frameon=False, labelcolor=INK_2)
    smoothing = f"EMA β = {ema}" if ema is not None else f"moving average {window}"
    fig.suptitle(f"{title}: per-seed curves ({smoothing})", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_mean(groups, metric, window, title, path, report=False, ema=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    x_end = 0
    end_labels = []
    for gi, (label, runs) in enumerate(groups):
        smoothed = [smooth(m[metric], window, ema) for _, m in runs]
        x = min((sx for sx, _ in smoothed), key=len)
        curves = np.stack([sy[:len(x)] for _, sy in smoothed])
        mean = curves.mean(axis=0)
        if report:
            std = curves.std(axis=0)
            band = (mean - std, mean + std)
        else:
            band = (curves.min(axis=0), curves.max(axis=0))
        ax.fill_between(x, *band, color=SERIES[gi], alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=SERIES[gi], linewidth=1.6,
                label=label if report else f"{label} (mean of {len(runs)} seeds)")
        final = np.mean([m[metric][-500:].mean() for _, m in runs])
        end_labels.append([mean[-1], f"{final:.1f}"])
        x_end = max(x_end, x[-1])

    # Push end labels apart (keeping their order) so close lines don't print on top of each other.
    lo, hi = ax.get_ylim()
    end_labels.sort(key=lambda e: e[0])
    for prev, cur in zip(end_labels, end_labels[1:]):
        cur[0] = max(cur[0], prev[0] + 0.05 * (hi - lo))
    for y, text in end_labels:
        ax.annotate(text, (x_end, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK_2)

    ax.set_xlim(right=x_end * 1.08)
    ax.set_ylabel(METRICS[metric], color=INK_2)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2)
    if not report:
        ax.set_title(f"{title}\nmean over seeds, band = min-max, end labels = mean of last 500 episodes",
                     color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300 if report else 150, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()
    groups = [(g[0], [(Path(r), load(Path(r))) for r in g[1:]]) for g in args.group]
    print_table(groups)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds_path = out.with_name(out.name + "_seeds.png")
    mean_path = out.with_name(out.name + "_mean.png")
    plot_seeds(groups, args.metric, args.window, args.title, seeds_path, ema=args.ema)
    plot_mean(groups, args.metric, args.window, args.title, mean_path, report=args.report, ema=args.ema)
    print(f"Saved {seeds_path.resolve()}\nSaved {mean_path.resolve()}")


if __name__ == "__main__":
    main()
