#!/usr/bin/env python3
"""
t-channel close-up — Figure 3 (Section 4.2).

Zooms in on the two sharp kinematic cut-offs of the MC-POM ``t`` channel
(near ``t = -0.4`` and ``t = -1.0``) and overlays generated vs. ground-truth
distributions plus a Gen/Truth ratio panel, to illustrate the residual flow
deviations characteristic of continuous flow-based models at hard boundaries.

Inputs:
    <run_dir>/dataset_cache.npz         — Cached truth (``pre_transform_data``).
    <run_dir>/generated_samples_best.npz — Pre-generated samples from PREDICT.
    <run_dir>/.hydra/config.yaml         — Used to reproduce the test split.

Outputs:
    <run_dir>/t_channel_closeup_highres.pdf

Usage:
    python scripts/3_t_channel_closeup.py <run_dir> [--split {test,full}]

Examples:
    python scripts/3_t_channel_closeup.py outputs/generation/mcpom/mcpom_gen

Notes:
    Defaults to evaluating on the held-out TEST split reproduced from the
    seeded ``torch.random_split`` recorded in ``.hydra/config.yaml``. Pass
    ``--split full`` to use the entire cached dataset instead.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scatterprism.utils import (  # noqa: E402
    COLOR_TRUTH, COLOR_GENERATED, COLOR_CONTEXT, COLOR_RATIO, COLOR_BAND,
    COLOR_MARKER, find_best_samples, reproduce_test_indices,
)

# ── JINST-compatible matplotlib settings ─────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 300,
})

def _plot_panel(ax_main, ax_ratio, truth, gen, xrange, marker_x, n_bins=200):
    """Plot one close-up panel with main histogram and Gen/Truth ratio."""
    ct, edges = np.histogram(truth, bins=n_bins, range=xrange, density=True)
    cg, _     = np.histogram(gen,   bins=n_bins, range=xrange, density=True)

    ax_main.fill_between(edges[:-1], ct, step="post", alpha=0.6,
                         color=COLOR_TRUTH, label="Ground truth")
    ax_main.step(edges[:-1], cg, where="post", color=COLOR_GENERATED,
                 linewidth=1.5, label="Generated")
    ax_main.axvline(marker_x, color=COLOR_MARKER, linestyle="--", alpha=0.8)
    ax_main.set_xlim(xrange)
    ax_main.set_ylabel("Density")
    ax_main.legend()
    ax_main.grid(True, linestyle=":", alpha=0.5)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ct > 0, cg / ct, np.nan)
    ax_ratio.step(edges[:-1], ratio, where="post",
                  color=COLOR_RATIO, linewidth=1.0)
    ax_ratio.axhline(1.0, color=COLOR_CONTEXT, linestyle="--", linewidth=0.8)
    ax_ratio.fill_between(edges[:-1], 0.9, 1.1, alpha=0.2,
                          color=COLOR_BAND, step="post")
    ax_ratio.set_xlabel(r"$t$")
    ax_ratio.set_ylabel("Gen/Truth")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, linestyle=":", alpha=0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Plot t-channel cut-off close-ups for an MC-POM run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--split", choices=["test", "full"], default="test",
                        help="Truth split to evaluate against (default: test)")
    parser.add_argument("--all-available-ckpts", action="store_true",
                        help="Allow fallback to generated_samples_last.npz if "
                             "generated_samples_best.npz is missing (default: "
                             "strict best, error if missing).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    print(f"Run directory: {run_dir}")

    gen_npz = find_best_samples(run_dir, allow_fallback=args.all_available_ckpts)
    print(f"Loading generated samples from {gen_npz}")
    gen_t = np.load(gen_npz)["samples"][:, 0]

    cache_path = run_dir / "dataset_cache.npz"
    print(f"Loading truth from {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    truth_t_full = cache["pre_transform_data"][:, 0]

    # Slice truth to the chosen split. Generated samples are produced from
    # noise (no 1-to-1 mapping to events), so we trim them to match length.
    if args.split == "test":
        test_idx = reproduce_test_indices(run_dir, len(truth_t_full))
        truth_t = truth_t_full[test_idx]
    else:
        truth_t = truth_t_full
    n = min(len(truth_t), len(gen_t))
    truth_t, gen_t = truth_t[:n], gen_t[:n]
    print(f"  Truth t: N={len(truth_t):,} | Gen t: N={len(gen_t):,}")

    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.0, wspace=0.25)

    ax_main_u = fig.add_subplot(gs[0, 0])
    ax_ratio_u = fig.add_subplot(gs[1, 0], sharex=ax_main_u)
    _plot_panel(ax_main_u, ax_ratio_u, truth_t, gen_t,
                xrange=(-0.42, -0.38), marker_x=-0.4)

    ax_main_l = fig.add_subplot(gs[0, 1])
    ax_ratio_l = fig.add_subplot(gs[1, 1], sharex=ax_main_l)
    _plot_panel(ax_main_l, ax_ratio_l, truth_t, gen_t,
                xrange=(-1.02, -0.98), marker_x=-1.0)

    plt.tight_layout()

    out_pdf = run_dir / "t_channel_closeup_highres.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
