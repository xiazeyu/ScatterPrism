#!/usr/bin/env python3
"""
Denoising / detector-unfolding comparison — Figure 5 (Section 4.2) and
Figures 16–17 (Appendix E).

Three-way overlay (Original / Detector / Unfolded) with Unfolded/Original
ratio sub-panels, per kinematic channel, for paired denoising runs at
different smearing intensities (mcpom_sigma_0.5, mcpom_sigma_1.0,
mcpom_sigma_2.0).

Inputs:
    <run_dir>/dataset_cache.npz          — Truth (``pre_transform_data``) +
                                            detector observations
                                            (``detector_data``, transformed).
    <run_dir>/generated_samples_best.npz — Unfolded samples from PREDICT.
    <run_dir>/.hydra/config.yaml         — Used to reproduce the test split.
    <run_dir>/{final_model,checkpoints/best,checkpoints/last}.ckpt
                                         — For the inverse transform applied
                                            to detector_data.

Outputs:
    <run_dir>/distributions_comparison_<suffix>.pdf       (full split)
    <run_dir>/distributions_comparison_<suffix>_test.pdf  (default test split)

Usage:
    python scripts/5_denoise_comparison.py <run_dir> [<run_dir> ...] [options]

Examples:
    # Figure 5 (sigma=1.0)
    python scripts/5_denoise_comparison.py outputs/denoise/mcpom/mcpom_sigma_1.0

    # Figures 16-17 (sigma=2.0 and sigma=0.5)
    python scripts/5_denoise_comparison.py \\
        outputs/denoise/mcpom/mcpom_sigma_2.0 outputs/denoise/mcpom/mcpom_sigma_0.5

Notes:
    Defaults to evaluating on the held-out TEST split. Since these are paired
    runs, the detector / unfolded / truth arrays share a common ordering, so
    all three are sliced with the same test indices. Pass ``--split full`` to
    use the entire cached dataset.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scatterprism.transforms import BaseTransform  # noqa: E402
from scatterprism.utils import (  # noqa: E402
    _MCPOM_COLS_LIST, _MCPOM_SCALES, _MCPOM_LABELS, _MCPOM_COL_INDEX,
    COLOR_TRUTH, COLOR_GENERATED, COLOR_DETECTOR, COLOR_CONTEXT, COLOR_BAND,
    find_best_checkpoint, find_best_samples, reproduce_test_indices,
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

DENOISE_COLORS = {
    "Original": COLOR_TRUTH,
    "Detector": COLOR_DETECTOR,
    "Unfolded": COLOR_GENERATED,
}


def load_transform_from_checkpoint(run_dir: Path, *, allow_fallback: bool = False):
    """Load the dataset transform persisted inside the checkpoint."""
    ckpt_path = find_best_checkpoint(run_dir, allow_fallback=allow_fallback)
    print(f"  Loading transform from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "transform_state" in ckpt:
        transform = BaseTransform.deserialize(ckpt["transform_state"])
        if transform is not None:
            return transform
    print("  WARNING: no transform found in checkpoint")
    return None


def plot_distributions_multiple_with_ratio(datasets, output_filepath, *,
                                           cols=None, reference_label="Original"):
    """Plot N labelled datasets per channel with ratio panels vs. reference."""
    if cols is None:
        cols = {name: _MCPOM_COL_INDEX[name] for name in _MCPOM_COLS_LIST}

    n_plots = len(cols)
    ncols_grid = 4
    nrows_grid = math.ceil(n_plots / ncols_grid)

    fig = plt.figure(figsize=(20, 4 * nrows_grid))
    outer_gs = GridSpec(nrows_grid, ncols_grid, figure=fig,
                        hspace=0.33, wspace=0.2,
                        bottom=0.08, top=0.95, left=0.05, right=0.98)

    scales = _MCPOM_SCALES
    n_bins = 200
    main_axes = []

    if reference_label not in datasets:
        reference_label = next(iter(datasets.keys()))
    ref_data = datasets[reference_label]

    for i, (col, data_idx) in enumerate(cols.items()):
        r, c = divmod(i, ncols_grid)
        inner_gs = outer_gs[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main  = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
        main_axes.append(ax_main)

        if col in scales and "xlim" in scales[col]:
            bin_range = scales[col]["xlim"]
        else:
            all_d = [d[:, data_idx] for d in datasets.values() if len(d)]
            if not all_d:
                bin_range = (0, 1)
            else:
                lo = min(np.min(d) for d in all_d)
                hi = max(np.max(d) for d in all_d)
                bin_range = (lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)
        bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

        ref_counts, edges = np.histogram(ref_data[:, data_idx], bins=bins, density=True)

        for label, data in datasets.items():
            color = DENOISE_COLORS.get(label, COLOR_CONTEXT)
            d = data[:, data_idx]
            if len(d) == 0:
                continue
            counts, _ = np.histogram(d, bins=bins, density=True)
            ls = "-" if label == reference_label else "--"
            ax_main.fill_between(edges[:-1], counts, step="post", alpha=0.15, color=color)
            ax_main.step(edges[:-1], counts, where="post", color=color,
                         linewidth=1.5, linestyle=ls, label=label)

            if label != reference_label:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(ref_counts > 0, counts / ref_counts, np.nan)
                ax_ratio.step(edges[:-1], ratio, where="post",
                              color=color, linewidth=1.0,
                              label=f"{label}/{reference_label}")

        ax_main.set_title(_MCPOM_LABELS.get(col, col))
        ax_main.set_ylabel("Density")
        ax_main.grid(True, linestyle=":", linewidth=0.5)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        if col in scales and "xlim" in scales[col]:
            ax_main.set_xlim(scales[col]["xlim"])

        ax_ratio.axhline(1.0, color=COLOR_CONTEXT, linestyle="--", linewidth=0.8)
        ax_ratio.fill_between(edges[:-1], 0.9, 1.1, alpha=0.2,
                              color=COLOR_BAND, step="post")
        ax_ratio.set_xlabel("Value")
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(True, linestyle=":", linewidth=0.5)

    handles, labels = main_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(datasets), 4), bbox_to_anchor=(0.5, 0.02))

    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    plt.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot with ratio saved as {output_filepath}")


def plot_distributions_1d_multiple_with_ratio(datasets, output_filepath,
                                              reference_label="Original"):
    """1-D version of the three-way comparison plot."""
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    flat = {k: np.asarray(v).flatten() for k, v in datasets.items()}
    combined = np.concatenate(list(flat.values()))
    vmin, vmax = np.percentile(combined, [0.5, 99.5])
    bins = np.linspace(vmin, vmax, 201)

    if reference_label not in flat:
        reference_label = next(iter(flat.keys()))
    ref_data = flat[reference_label]
    ref_counts, edges = np.histogram(ref_data, bins=bins, density=True)

    for label, data in flat.items():
        color = DENOISE_COLORS.get(label, COLOR_CONTEXT)
        counts, _ = np.histogram(data, bins=bins, density=True)
        ls = "-" if label == reference_label else "--"
        ax_main.fill_between(edges[:-1], counts, step="post", alpha=0.15, color=color)
        ax_main.step(edges[:-1], counts, where="post", color=color,
                     linewidth=1.5, linestyle=ls, label=label)

        if label != reference_label:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(ref_counts > 0, counts / ref_counts, np.nan)
            ax_ratio.step(edges[:-1], ratio, where="post",
                          color=color, linewidth=1.0)

    ax_main.set_ylabel("Density")
    ax_main.set_title("Distribution Comparison")
    ax_main.legend()
    ax_main.grid(True, linestyle=":")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    ax_ratio.axhline(1.0, color=COLOR_CONTEXT, linestyle="--", linewidth=0.8)
    ax_ratio.fill_between(edges[:-1], 0.9, 1.1, alpha=0.2,
                          color=COLOR_BAND, step="post")
    ax_ratio.set_xlabel("Value")
    ax_ratio.set_ylabel(f"X/{reference_label}")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, linestyle=":")

    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    plt.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 1D comparison plot with ratio to {output_filepath}")


def process_run(run_dir: Path, split: str, *, allow_fallback: bool = False):
    print(f"\n{'='*60}\nProcessing: {run_dir}\n{'='*60}")

    try:
        gen_path = find_best_samples(run_dir, allow_fallback=allow_fallback)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return
    suffix = gen_path.stem.replace("generated_samples_", "")
    print(f"Loading generated samples from {gen_path}...")
    samples = np.load(gen_path)["samples"]
    print(f"  Generated shape: {samples.shape}")

    cache_path = run_dir / "dataset_cache.npz"
    if not cache_path.exists():
        print(f"  Error: dataset cache not found at {cache_path}")
        return
    print(f"Loading dataset cache from {cache_path}...")
    cache = np.load(cache_path, allow_pickle=True)

    if "pre_transform_data" in cache.files:
        truth_full = cache["pre_transform_data"]
    else:
        print("  Warning: pre_transform_data not found, using original_data")
        truth_full = cache["original_data"]

    if "detector_data" not in cache.files:
        print("  Error: detector_data not found in cache (run must be paired)")
        return
    detector_full_transformed = cache["detector_data"]

    # Inverse-transform detector_data back to physical units for direct
    # overlay with truth (which is in physical units already).
    try:
        transform = load_transform_from_checkpoint(run_dir, allow_fallback=allow_fallback)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return
    detector_full_physical = (
        transform.inverse_transform(detector_full_transformed)
        if transform is not None else detector_full_transformed
    )

    # Paired runs: gen / truth / detector share the same ordering, so all
    # three are sliced with the same test indices.
    if split == "test":
        test_idx = reproduce_test_indices(run_dir, len(truth_full))
        n = min(len(test_idx), len(samples))
        idx = test_idx[:n]
        truth    = truth_full[idx]
        detector = detector_full_physical[idx]
        unfolded = samples[idx]
    else:
        n = min(len(truth_full), len(samples))
        truth    = truth_full[:n]
        detector = detector_full_physical[:n]
        unfolded = samples[:n]
    print(f"  Aligned: truth={len(truth):,}, detector={len(detector):,}, "
          f"unfolded={len(unfolded):,}")

    comparison = {"Original": truth, "Detector": detector, "Unfolded": unfolded}

    is_mcpom = samples.ndim == 2 and samples.shape[1] == 24
    tag = "" if split == "full" else "_test"
    out_path = str(run_dir / f"distributions_comparison_{suffix}{tag}.pdf")

    print("Generating PDF plot...")
    if is_mcpom:
        plot_distributions_multiple_with_ratio(comparison, out_path,
                                                reference_label="Original")
    else:
        n_cols = samples.shape[1] if samples.ndim == 2 else 1
        if n_cols > 1:
            cols = {f"Feature {i}": i for i in range(n_cols)}
            plot_distributions_multiple_with_ratio(comparison, out_path,
                                                    cols=cols,
                                                    reference_label="Original")
        else:
            plot_distributions_1d_multiple_with_ratio(comparison, out_path,
                                                      reference_label="Original")


def main():
    parser = argparse.ArgumentParser(
        description="Three-way Original/Detector/Unfolded distribution plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", type=str, nargs="+",
                        help="One or more denoising run directories")
    parser.add_argument("--all-available-ckpts", action="store_true",
                        help="Allow fallback for both samples "
                             "(generated_samples_best.npz -> ..._last.npz) and "
                             "checkpoint loading (final_model.ckpt -> "
                             "checkpoints/best.ckpt -> checkpoints/last.ckpt). "
                             "Default: strict best, error if missing.")
    parser.add_argument("--split", choices=["test", "full"], default="test",
                        help="Split to evaluate against (default: test)")
    args = parser.parse_args()

    for d in args.run_dirs:
        process_run(Path(d.rstrip("/")), args.split,
                    allow_fallback=args.all_available_ckpts)


if __name__ == "__main__":
    main()
