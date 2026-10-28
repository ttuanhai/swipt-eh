"""Compare our Random strategy with the Random curve of the paper's Fig. 2a, both smoothed by EMA.

The paper curve is digitised from docs/paper.pdf: Fig. 2 is a single raster image, and the
Random strategy is the only near-black curve inside the Fig. 2a axes.

Writes two figures:
  random_vs_paper.png        our mean over seeds (+-1 std band) against the paper's curve
  random_vs_paper_seeds.png  every seed's curve, and each seed's final value, against the paper's
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

from plot_seeds import INK_2, METRICS, MUTED, SERIES, SURFACE, load, smooth, style

PAPER = Path("docs/paper.pdf")
RUNS = [Path(f"results/random/seed{s}") for s in range(20)]
OUT = Path("results/random_vs_paper.png")
OUT_SEEDS = Path("results/random_vs_paper_seeds.png")
EMA = 0.99  # fitted to the rise of the paper's Random curve

# Fig. 2 image (page 5, xref 297, 7838 x 2509 px); Fig. 2a axes calibrated on its spines and grid lines.
FIG_XREF = 297
X0, PX_PER_EP = 539.3, 417.3 / 500     # column of episode 0, pixels per episode
Y0, PX_PER_UNIT = 2090.3, 303.0 / 20   # row of 0 bit/s/Hz, pixels per bit/s/Hz
COLS, ROWS = (545, 2620), (420, 2083)  # inside the plot area, clear of the spines and the legend


def paper_random() -> tuple[np.ndarray, np.ndarray]:
    """(episodes, sum rate) of the paper's Random curve: median row of the near-black pixels per column."""
    pix = pymupdf.Pixmap(pymupdf.open(PAPER), FIG_XREF)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    dark = img.max(axis=2) < 90
    ep, val = [], []
    for c in range(*COLS):
        rows = np.nonzero(dark[ROWS[0]:ROWS[1], c])[0] + ROWS[0]
        if len(rows):
            ep.append((c - X0) / PX_PER_EP)
            val.append((Y0 - np.median(rows)) / PX_PER_UNIT)
    return np.array(ep), np.array(val)


def plot_per_seed(runs, ep, val, paper_end):
    """Left: every seed's EMA curve against the paper's. Right: each seed's mean of the last 500 episodes."""
    finals = np.array([y[-500:].mean() for y in runs])
    best = int(finals.argmax())
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, ax2):
        style(a)

    others = [i for i in range(len(runs)) if i != best]
    for i in others:
        x, y = smooth(runs[i], None, EMA)
        ax.plot(x, y, color=MUTED, linewidth=0.8, alpha=0.6,
                label=f"Random (ours, {len(runs)} seeds)" if i == others[0] else None)
    x, y = smooth(runs[best], None, EMA)
    ax.plot(x, y, color=SERIES[0], linewidth=1.6, label=f"Best seed (seed {best})")
    ax.plot(ep, val, color=SERIES[1], linewidth=1.6, linestyle="--", label="Random (paper, Fig. 2a)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(METRICS["sum_rate"], color=INK_2)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2)

    seeds = np.arange(len(runs))
    ax2.axhline(paper_end, color=SERIES[1], linewidth=1.6, linestyle="--")
    ax2.axhline(finals.mean(), color=SERIES[0], linewidth=1.0)
    ax2.plot(seeds, finals, "o", color=SERIES[0], markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax2.annotate(f"paper {paper_end:.1f}", (seeds[-1], paper_end), xytext=(0, 4), textcoords="offset points",
                 ha="right", va="bottom", color=INK_2)
    ax2.annotate(f"mean of seeds {finals.mean():.1f}", (seeds[-1], finals.mean()), xytext=(0, -8),
                 textcoords="offset points", ha="right", va="top", color=INK_2)
    ax2.set_xticks(seeds[::2])
    ax2.set_xlim(-1, len(runs))
    ax2.set_xlabel("Seed (mean of last 500 episodes)", color=INK_2)

    fig.tight_layout()
    fig.savefig(OUT_SEEDS, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"best seed: seed{best} {finals[best]:.2f}, {paper_end - finals[best]:.2f} below the paper")
    print(f"Saved {OUT_SEEDS.resolve()}")


def main():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor(SURFACE)
    style(ax)

    runs = [load(r)["sum_rate"] for r in RUNS]
    smoothed = [smooth(y, None, EMA) for y in runs]
    x = smoothed[0][0]
    curves = np.stack([sy for _, sy in smoothed])
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    ax.fill_between(x, mean - std, mean + std, color=SERIES[0], alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=SERIES[0], linewidth=1.6, label=f"Random (ours, {len(runs)} seeds)")
    ours_end = np.mean([y[-500:].mean() for y in runs])

    ep, val = paper_random()
    ax.plot(ep, val, color=SERIES[1], linewidth=1.6, linestyle="--", label="Random (paper, Fig. 2a)")
    paper_end = val[ep >= 2000].mean()

    x_end = x[-1]
    for y, end in ((mean[-1], ours_end), (val[-1], paper_end)):
        ax.annotate(f"{end:.1f}", (x_end, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK_2)
    ax.set_xlim(right=x_end * 1.08)
    ax.set_ylabel(METRICS["sum_rate"], color=INK_2)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"ours: mean of last 500 episodes {ours_end:.2f} (std over seeds "
          f"{np.std([y[-500:].mean() for y in runs]):.2f})")
    print(f"paper: digitised curve, mean over episodes 2000-2500 {paper_end:.2f}")
    print(f"Saved {OUT.resolve()}")
    plot_per_seed(runs, ep, val, paper_end)


if __name__ == "__main__":
    main()
