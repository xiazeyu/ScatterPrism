#!/usr/bin/env python3
"""
Checkpoint-evolution grid — Figure 14 (Appendix C).

Loads a list of saved checkpoints, regenerates samples from each, and lays
the resulting density histograms out in a chronological grid to visualise
how the learned distribution converges as training progresses.

Inputs:
    <run_dir>                          — Run directory containing checkpoints.
    <run_dir>/checkpoints/epoch_*.ckpt — Saved intermediate checkpoints.
    <run_dir>/.hydra/config.yaml       — Used by the transform loader.

Outputs:
    <run_dir>/checkpoint_evolution_grid_custom.pdf

Usage:
    python scripts/7_checkpoint_evolution.py <run_dir> --epochs E1 E2 E3 ...

Examples:
    # Figure 14 — noise_10spikes evolution
    python scripts/7_checkpoint_evolution.py outputs/generation/mock/noise_10spikes \\
        --epochs 9 19 29 39 49 59 69 79 109 209 309 409 509 609 709

Notes:
    Each panel plots ONLY generated samples (no truth overlay), so no data
    split is involved. Samples are drawn from the base Gaussian and
    integrated through each checkpoint's learned ODE.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

from scatterprism import utils  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Render a grid of generated distributions across training epochs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--epochs", nargs="+", type=int, required=True,
                        help="Epochs to include (e.g., --epochs 9 19 29 109 209)")
    parser.add_argument("--n_generate", type=int, default=100_000,
                        help="Samples per checkpoint (default: 100000)")
    parser.add_argument("--bins", type=int, default=200,
                        help="Histogram bin count (default: 200)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_path = Path(args.run_dir)

    all_checkpoints = utils.find_checkpoints(run_path)
    if not all_checkpoints:
        log.error(f"No checkpoints found in {run_path}")
        return

    epoch_set = set(args.epochs)
    filtered = [(e, p) for e, p in all_checkpoints if e in epoch_set]
    if not filtered:
        log.error(f"No checkpoints matched epochs {args.epochs}")
        log.info(f"Available epochs: {[e for e, _ in all_checkpoints]}")
        return

    log.info(f"Plotting evolution for {len(filtered)} checkpoints: "
             f"epochs {[e for e, _ in filtered]}")

    # Locate the checkpoints directory for transform loading.
    actual_ckpt_dir = run_path
    if not list(run_path.glob("epoch_*.ckpt")) and (run_path / "checkpoints").is_dir():
        actual_ckpt_dir = run_path / "checkpoints"

    transform = utils.load_checkpoint_transform(actual_ckpt_dir, device)

    utils.plot_checkpoint_evolution_grid(
        checkpoints=filtered,
        n_generate=args.n_generate,
        device=device,
        output_path=str(run_path / "checkpoint_evolution_grid_custom.pdf"),
        transform=transform,
        bins=args.bins,
    )

    log.info(f"Saved plot to {run_path}")


if __name__ == "__main__":
    main()
