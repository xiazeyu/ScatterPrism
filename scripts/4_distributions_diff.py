#!/usr/bin/env python3
"""
Distribution comparison plots — Figure 4 (Section 4.2) and Figures 6–11
(Appendix C, synthetic benchmarks).

Overlays generated vs. ground-truth distributions for every kinematic
channel and adds a Gen/Truth ratio sub-panel. Automatically detects
MC-POM (24-column) vs. mock (1-D) runs and lays out the figure accordingly.

Inputs:
    <run_dir>/dataset_cache.npz          — Cached truth (``pre_transform_data``).
    <run_dir>/generated_samples_best.npz — Pre-generated samples from PREDICT.
    <run_dir>/.hydra/config.yaml         — Used to reproduce the test split.

Outputs:
    <run_dir>/distributions_diff_<suffix>.pdf   (full split, --split full)
    <run_dir>/distributions_diff_<suffix>_test.pdf (default test split)

Usage:
    python scripts/4_distributions_diff.py <run_dir> [options]

Examples:
    # MC-POM (Figure 4)
    python scripts/4_distributions_diff.py outputs/generation/mcpom/mcpom_gen

    # Synthetic benchmarks (Figures 6-11)
    python scripts/4_distributions_diff.py outputs/generation/mock/triple_mixed_scale_5

Notes:
    Defaults to evaluating on the held-out TEST split reproduced from the
    seeded ``torch.random_split`` recorded in ``.hydra/config.yaml``. Pass
    ``--split full`` to use the entire cached dataset instead. Generated
    samples are drawn from the prior (no event-level alignment with truth),
    so we trim them to match the test-set length.
"""

import argparse
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

from scatterprism.utils import (  # noqa: E402
    _MCPOM_COLS_LIST, _MCPOM_SCALES, _MCPOM_LABELS, _MCPOM_COL_INDEX,
    COLOR_TRUTH, COLOR_GENERATED, COLOR_CONTEXT, COLOR_RATIO, COLOR_BAND,
    find_best_samples, reproduce_test_indices,
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

def plot_distributions_diff_1d_pdf(truth, generated, output_path,
                                    label_a="Ground truth", label_b="Generated"):
    """Overlay 1-D truth vs. generated histograms with Gen/Truth ratio."""
    truth_flat = np.asarray(truth).flatten()
    gen_flat   = np.asarray(generated).flatten()

    combined = np.concatenate([truth_flat, gen_flat])
    bins = np.linspace(np.min(combined), np.max(combined), 201)

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    counts_t, _ = np.histogram(truth_flat, bins=bins, density=True)
    counts_g, _ = np.histogram(gen_flat,   bins=bins, density=True)

    ax_main.fill_between(bins[:-1], counts_t, step="post", alpha=0.5,
                         color=COLOR_TRUTH, label=label_a)
    ax_main.step(bins[:-1], counts_g, where="post", color=COLOR_GENERATED,
                 linewidth=1.5, label=label_b)
    ax_main.set_ylabel("Density")
    ax_main.legend()
    ax_main.grid(True, linestyle=":")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(counts_t > 0, counts_g / counts_t, np.nan)
    ax_ratio.step(bins[:-1], ratio, where="post",
                  color=COLOR_RATIO, linewidth=1.2)
    ax_ratio.axhline(1.0, color=COLOR_CONTEXT, linestyle="--", linewidth=1)
    ax_ratio.fill_between(bins[:-1], 0.9, 1.1, alpha=0.2, color=COLOR_BAND)
    ax_ratio.set_xlabel("Value")
    ax_ratio.set_ylabel("Gen / Truth")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, linestyle=":")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, format="pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved 1D PDF plot to {output_path}")


def plot_distributions_diff_pdf(dataset_a, dataset_b, output_filepath,
                                 label_a="Ground truth", label_b="Generated"):
    """Overlay MC-POM truth vs. generated for all 20 displayed channels."""
    cols_to_plot = _MCPOM_COLS_LIST
    scales = _MCPOM_SCALES
    n_bins = 200
    nrows, ncols = 5, 4

    fig = plt.figure(figsize=(20, 18))
    outer_gs = GridSpec(nrows, ncols, figure=fig, hspace=0.33, wspace=0.2,
                        bottom=0.08, top=0.95, left=0.05, right=0.98)
    main_axes = []

    for i, col in enumerate(cols_to_plot):
        r, c = divmod(i, ncols)
        inner_gs = outer_gs[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main  = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)
        main_axes.append(ax_main)

        data_idx = _MCPOM_COL_INDEX[col]
        a, b = dataset_a[:, data_idx], dataset_b[:, data_idx]

        if col in scales and "xlim" in scales[col]:
            bin_range = scales[col]["xlim"]
        else:
            bin_range = (min(np.min(a), np.min(b)), max(np.max(a), np.max(b)))
        bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

        ct, edges = np.histogram(a, bins=bins, density=True)
        cg, _     = np.histogram(b, bins=bins, density=True)

        ax_main.fill_between(edges[:-1], ct, step="post", alpha=0.5,
                             color=COLOR_TRUTH, label=label_a)
        ax_main.step(edges[:-1], cg, where="post", color=COLOR_GENERATED,
                     linewidth=1.5, linestyle="-", label=label_b)
        ax_main.fill_between(edges[:-1], cg, step="post", alpha=0.15,
                             color=COLOR_GENERATED)
        ax_main.set_title(_MCPOM_LABELS.get(col, col))
        ax_main.set_ylabel("Density")
        ax_main.grid(True, linestyle=":", linewidth=0.5)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        if col in scales and "xlim" in scales[col]:
            ax_main.set_xlim(scales[col]["xlim"])

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ct > 0, cg / ct, np.nan)
        ax_ratio.step(edges[:-1], ratio, where="post",
                      color=COLOR_RATIO, linewidth=1.0)
        ax_ratio.axhline(1.0, color=COLOR_CONTEXT, linestyle="--", linewidth=0.8)
        ax_ratio.fill_between(edges[:-1], 0.9, 1.1, alpha=0.2,
                              color=COLOR_BAND, step="post")
        ax_ratio.set_xlabel("Value")
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(True, linestyle=":", linewidth=0.5)

    handles, labels = main_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.02))

    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    plt.savefig(output_filepath, format="pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PDF plot to {output_filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot generated vs. ground-truth distributions for a run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument("--all-available-ckpts", action="store_true",
                        help="Allow fallback to generated_samples_last.npz if "
                             "generated_samples_best.npz is missing (default: "
                             "strict best, error if missing).")
    parser.add_argument("--split", choices=["test", "full"], default="test",
                        help="Truth split to evaluate against (default: test)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output PDF path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir.rstrip("/"))

    gen_path = find_best_samples(run_dir, allow_fallback=args.all_available_ckpts)
    suffix = gen_path.stem.replace("generated_samples_", "")
    print(f"Loading generated samples from {gen_path}...")
    generated = np.load(gen_path)["samples"]
    print(f"  Generated shape: {generated.shape}")

    cache_path = run_dir / "dataset_cache.npz"
    if not cache_path.exists():
        sys.exit(f"Error: Dataset cache not found at {cache_path}")
    print(f"Loading truth from {cache_path}...")
    cache = np.load(cache_path, allow_pickle=True)
    truth_full = cache["pre_transform_data"]
    print(f"  Truth (full) shape: {truth_full.shape}")

    # Slice truth to the chosen split.
    if args.split == "test":
        test_idx = reproduce_test_indices(run_dir, len(truth_full))
        truth = truth_full[test_idx]
    else:
        truth = truth_full

    # For paired runs, generated has 1-to-1 correspondence with detector_data
    # (same ordering as truth); slice by test_idx too. For unpaired generation
    # the samples are i.i.d. from noise, so trim to the truth length.
    is_paired = "detector_data" in cache.files
    if is_paired and args.split == "test" and len(generated) == len(truth_full):
        generated = generated[test_idx]
    else:
        n = min(len(truth), len(generated))
        truth, generated = truth[:n], generated[:n]
    print(f"  Aligned: truth={len(truth):,}, generated={len(generated):,}")

    if args.output:
        output_path = args.output
    else:
        tag = "" if args.split == "full" else "_test"
        output_path = str(run_dir / f"distributions_diff_{suffix}{tag}.pdf")

    is_1d = (generated.ndim == 1) or (generated.shape[1] == 1)
    print("Generating PDF plot...")
    if is_1d:
        plot_distributions_diff_1d_pdf(truth, generated, output_path)
    else:
        plot_distributions_diff_pdf(truth, generated, output_path)


if __name__ == "__main__":
    main()
