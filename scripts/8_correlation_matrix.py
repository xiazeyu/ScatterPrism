#!/usr/bin/env python3
"""
Pearson correlation-matrix comparison — Figure 15 (Appendix D).

Side-by-side Pearson correlation heatmaps for the MC-POM truth vs.
generated samples, plus a difference plot that quantifies residual
discrepancies in the multivariate dependency structure.

Inputs:
    <run_dir>/dataset_cache.npz          — Truth (``pre_transform_data``) +
                                            column names.
    <run_dir>/generated_samples_best.npz — Pre-generated samples from PREDICT.
    <run_dir>/.hydra/config.yaml         — Used to reproduce the test split.

Outputs:
    <run_dir>/correlation_matrix_comparison{,_test}.{pdf,png}
    <run_dir>/correlation_difference_detailed{,_test}.{pdf,png}

Usage:
    python scripts/8_correlation_matrix.py <run_dir> [options]

Examples:
    python scripts/8_correlation_matrix.py outputs/generation/mcpom/mcpom_gen

Notes:
    Defaults to evaluating on the held-out TEST split. The four
    identically-zero channels (photon p_y, target-proton p_y, recoil-proton
    p_x/p_y) are excluded because their correlation entries are undefined.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scatterprism.utils import reproduce_test_indices  # noqa: E402

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


# Display labels (LaTeX) for each raw column name.
COLUMN_RENAME = {
    # Derived observables
    "t": r"$t$", "mpipi": r"$M_{\pi\pi}$",
    "costh": r"$\cos\theta$", "phi": r"$\phi$",
    # Incident photon (γ)
    "q0": r"$E_{\gamma}$",  "q1": r"$p_{\gamma x}$",
    "q2": r"$p_{\gamma y}$", "q3": r"$p_{\gamma z}$",
    # Target proton (p1)
    "p10": r"$E_{p_1}$",  "p11": r"$p_{1x}$",
    "p12": r"$p_{1y}$",   "p13": r"$p_{1z}$",
    # Positive pion (π+)
    "k10": r"$E_{\pi^+}$",  "k11": r"$p_{\pi^+ x}$",
    "k12": r"$p_{\pi^+ y}$","k13": r"$p_{\pi^+ z}$",
    # Negative pion (π−)
    "k20": r"$E_{\pi^-}$",  "k21": r"$p_{\pi^- x}$",
    "k22": r"$p_{\pi^- y}$","k23": r"$p_{\pi^- z}$",
    # Recoil proton (p2)
    "p20": r"$E_{p_2}$",  "p21": r"$p_{2x}$",
    "p22": r"$p_{2y}$",   "p23": r"$p_{2z}$",
}

# Identically-zero (delta) columns excluded from the correlation heatmaps.
DELTA_COLUMNS = {"q2", "p12", "p21", "p22"}


def filter_and_rename_columns(data: np.ndarray, columns: list):
    """Drop identically-zero columns and replace raw names with LaTeX labels."""
    keep_idx = [i for i, c in enumerate(columns) if c not in DELTA_COLUMNS]
    filtered = data[:, keep_idx]
    renamed  = [COLUMN_RENAME.get(columns[i], columns[i]) for i in keep_idx]
    removed  = [c for c in columns if c in DELTA_COLUMNS]
    if removed:
        print(f"  Removed delta columns: {removed}")
    return filtered, renamed


def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix with NaN (from constant cols) → 0."""
    corr = np.corrcoef(data, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


def plot_correlation_heatmaps(truth_corr, gen_corr, columns,
                              output_path: Path, title_prefix=""):
    """Side-by-side truth vs. generated correlation matrices + difference."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    vmin, vmax, cmap = -1, 1, "RdBu_r"

    im0 = axes[0].imshow(truth_corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title(f"{title_prefix}Truth Correlation Matrix", fontweight="bold")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(gen_corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axes[1].set_title(f"{title_prefix}Generated Correlation Matrix", fontweight="bold")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    diff = gen_corr - truth_corr
    max_diff = np.max(np.abs(diff))
    im2 = axes[2].imshow(diff, cmap=cmap, vmin=-max_diff, vmax=max_diff, aspect="equal")
    axes[2].set_title(
        f"{title_prefix}Difference (Gen − Truth)\nMax |diff| = {max_diff:.4f}",
        fontweight="bold",
    )
    plt.colorbar(im2, ax=axes[2], shrink=0.8, label="Correlation Difference")

    for ax in axes:
        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right")
        ax.set_yticklabels(columns)

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")
    print(f"  Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)
    return diff


def plot_detailed_difference(diff, columns, output_path: Path, title_prefix=""):
    """Annotated correlation-difference heatmap with numeric overlay."""
    fig, ax = plt.subplots(figsize=(14, 12))
    max_diff = np.max(np.abs(diff))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-max_diff, vmax=max_diff, aspect="equal")

    # Numeric labels for entries with |diff| > 0.05.
    for i in range(len(columns)):
        for j in range(len(columns)):
            v = diff[i, j]
            if abs(v) > 0.05:
                color = "white" if abs(v) > max_diff * 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color=color)

    ax.set_title(
        f"{title_prefix}Correlation Difference (Gen − Truth)\nMax |diff| = {max_diff:.4f}",
        fontweight="bold",
    )
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    plt.colorbar(im, ax=ax, shrink=0.8).set_label("Correlation Difference")

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")
    print(f"  Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def compute_correlation_metrics(truth_corr, gen_corr):
    """Summary scalars comparing two correlation matrices."""
    diff = gen_corr - truth_corr
    upper = diff[np.triu_indices_from(diff, k=1)]
    return {
        "max_abs_diff":     float(np.max(np.abs(diff))),
        "mean_abs_diff":    float(np.mean(np.abs(upper))),
        "rmse":             float(np.sqrt(np.mean(upper ** 2))),
        "max_positive":     float(np.max(upper)),
        "max_negative":     float(np.min(upper)),
        "n_diff_gt_0.1":    int(np.sum(np.abs(upper) > 0.1)),
        "n_diff_gt_0.05":   int(np.sum(np.abs(upper) > 0.05)),
        "frobenius_norm":   float(np.linalg.norm(diff, "fro")),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pearson correlation heatmaps for MC-POM generation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=str, help="Path to MC-POM run directory")
    parser.add_argument("--split", choices=["test", "full"], default="test",
                        help="Truth split to evaluate against (default: test)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Subsample size for faster computation (default: use all)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    truth_path = run_dir / "dataset_cache.npz"
    gen_path   = run_dir / "generated_samples_best.npz"
    print(f"Loading truth from {truth_path}")
    truth_cache = np.load(truth_path, allow_pickle=True)
    print(f"Loading generated samples from {gen_path}")
    gen_samples_full = np.load(gen_path)["samples"]

    truth_full = truth_cache["pre_transform_data"]
    columns    = list(truth_cache["columns"])
    print(f"  Truth (full) shape: {truth_full.shape}")
    print(f"  Generated shape:    {gen_samples_full.shape}")
    print(f"  Columns: {columns}")

    if args.split == "test":
        test_idx = reproduce_test_indices(run_dir, len(truth_full))
        truth_samples = truth_full[test_idx]
        n = min(len(test_idx), len(gen_samples_full))
        gen_samples   = gen_samples_full[:n]
        truth_samples = truth_samples[:n]
    else:
        n = min(len(truth_full), len(gen_samples_full))
        truth_samples = truth_full[:n]
        gen_samples   = gen_samples_full[:n]

    truth_samples, display_cols = filter_and_rename_columns(truth_samples, columns)
    gen_samples,   _            = filter_and_rename_columns(gen_samples,   columns)

    if args.sample_size and args.sample_size < len(truth_samples):
        print(f"  Subsampling to {args.sample_size} events")
        rng = np.random.default_rng(42)
        sel = rng.choice(len(truth_samples), size=args.sample_size, replace=False)
        truth_samples, gen_samples = truth_samples[sel], gen_samples[sel]

    print("Computing correlation matrices...")
    truth_corr = compute_correlation_matrix(truth_samples)
    gen_corr   = compute_correlation_matrix(gen_samples)

    print("\nCorrelation matrix comparison metrics:")
    for k, v in compute_correlation_metrics(truth_corr, gen_corr).items():
        print(f"  {k}: {v}")

    tag = "" if args.split == "full" else "_test"
    print("\nGenerating plots...")
    diff = plot_correlation_heatmaps(
        truth_corr, gen_corr, display_cols,
        run_dir / f"correlation_matrix_comparison{tag}",
    )
    plot_detailed_difference(
        diff, display_cols,
        run_dir / f"correlation_difference_detailed{tag}",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
