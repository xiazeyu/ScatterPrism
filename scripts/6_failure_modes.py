#!/usr/bin/env python3
"""
Common flow-matching failure modes — Figure 13 (Appendix C).

Two-panel didactic figure illustrating typical CFM artefacts before full
convergence:

  (a) Underfitting artefacts — spurious sharp spikes, blended/smeared peaks.
  (b) Boundary smearing      — intrinsic CFM leakage past hard kinematic cuts.

By default the panels are produced from FIXED synthetic distributions so the
figure is fully reproducible and decoupled from any training run. The
``--underfitting-run`` and ``--smearing-run`` options allow regenerating the
same panels from an actual early-stopped checkpoint.

Inputs:
    None (synthetic mode, default).
    Optional: <run_dir>/dataset_cache.npz + generated_samples_<epoch>.npz
              when ``--underfitting-run`` / ``--smearing-run`` are given.

Outputs:
    <output> (default: figures/failure_modes.pdf)

Usage:
    python scripts/6_failure_modes.py [--output figures/failure_modes.pdf]

Examples:
    # Reproduce Figure 13 from synthetic exemplars (default)
    python scripts/6_failure_modes.py --output figures/failure_modes.pdf

    # Regenerate from a real early-epoch checkpoint
    python scripts/6_failure_modes.py \\
        --underfitting-run outputs/generation/mock/triple_mixed_scale_5 \\
        --underfitting-epoch 9

Notes:
    These exemplars are illustrative only — they are deliberately produced
    from fixed deterministic synthetic distributions and do not depend on
    any train/val/test split.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scatterprism.utils import (  # noqa: E402
    COLOR_TRUTH, COLOR_GENERATED, COLOR_GEN_FILL, COLOR_CONTEXT,
)

# ============================================================
# JINST-compatible Matplotlib settings
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.dpi": 300,
})


def generate_synthetic_underfitting_example(n_samples=100000, seed=42):
    """Generate synthetic data demonstrating underfitting failure mode.
    
    Based on triple_mixed config: peaks: [[0.0, 1.0, 0.7], [2.5, 0.8, 0.3], [5.0, 1.5, 0.34]]
    
    Creates:
    - Ground truth: trimodal distribution with well-separated peaks
    - Generated: underfitted version showing:
      * Smearing effect: second and third peaks blended/merged together
      * Spurious mode: sharp spike(s) that don't exist in truth
    
    Returns:
        truth: Ground truth samples
        generated: Simulated underfitted samples
    """
    rng = np.random.default_rng(seed)
    
    # Ground truth: trimodal distribution with distinct peaks
    # Peak 1 at 0, Peak 2 at 3, Peak 3 at 5.5
    weights = [0.45, 0.30, 0.25]
    means = [0.0, 3.0, 5.5]
    stds = [0.7, 0.5, 0.6]
    
    truth_samples = []
    for w, m, s in zip(weights, means, stds):
        n = int(w * n_samples)
        truth_samples.append(rng.normal(m, s, n))
    truth = np.concatenate(truth_samples)
    
    # Simulated underfitted output:
    # 1. First peak: slightly smeared but present
    # 2. Second and third peaks: BLENDED together into one broad bump
    # 3. Spurious sharp spikes between peak 1 and the blended region (lower height)
    
    gen_samples = []
    
    # Peak 1: smeared version
    gen_samples.append(rng.normal(0.0, 0.85, int(n_samples * 0.45)))
    
    # Peaks 2+3 blended: one broad distribution centered between them
    gen_samples.append(rng.normal(4.0, 1.2, int(n_samples * 0.51)))
    
    # Spurious sharp spikes - ~HALF height of first peak
    # Use smaller sample counts to reduce spike heights
    gen_samples.append(rng.normal(1.5, 0.12, int(n_samples * 0.022)))  # Sharp spike 1
    gen_samples.append(rng.normal(2.2, 0.10, int(n_samples * 0.018)))  # Sharp spike 2
    
    generated = np.concatenate(gen_samples)
    
    return truth, generated


def generate_synthetic_smearing_example(n_samples=100000, seed=123):
    """Generate synthetic data demonstrating boundary smearing.
    
    Creates:
    - Ground truth: Gaussian(0, 1) with smooth falloff near threshold 1.0
    - Generated: FM output with additional boundary leakage
    
    Returns:
        truth: Ground truth samples with smooth cutoff
        generated: Simulated FM output with more leakage
    """
    rng = np.random.default_rng(seed)
    
    # Ground truth: Gaussian(0, 1) with HARD vertical cutoff at threshold 1.0
    mean, std = 0.0, 1.0
    threshold = 1.0
    
    # Generate samples with hard rejection at threshold
    raw = rng.normal(mean, std, int(n_samples * 2))
    truth = raw[raw <= threshold][:n_samples]
    
    # FM-generated version:
    # Similar smooth falloff but with more leakage past threshold
    gen_raw = rng.normal(mean, std, int(n_samples * 3))
    
    # Softer cutoff (smaller k) = more leakage
    k_gen = 40  # Sharper cutoff for generated
    keep_prob_gen = 1.0 / (1.0 + np.exp(k_gen * (gen_raw - threshold)))
    keep_mask_gen = rng.random(len(gen_raw)) < keep_prob_gen
    generated = gen_raw[keep_mask_gen][:n_samples]
    
    return truth[:n_samples], generated[:n_samples]


def load_from_checkpoint(run_dir, epoch_suffix='best', dataset_key='pre_transform_data'):
    """Load generated samples and ground truth from a run directory.
    
    Args:
        run_dir: Path to run directory
        epoch_suffix: Suffix for samples file (e.g., 'best', 'last', '009')
        dataset_key: Key in dataset_cache.npz for truth data
        
    Returns:
        truth: Ground truth array
        generated: Generated samples array
    """
    gen_path = os.path.join(run_dir, f'generated_samples_{epoch_suffix}.npz')
    cache_path = os.path.join(run_dir, 'dataset_cache.npz')
    
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"Generated samples not found: {gen_path}")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Dataset cache not found: {cache_path}")
    
    gen_data = np.load(gen_path)
    generated = gen_data['samples']
    
    cache_data = np.load(cache_path, allow_pickle=True)
    truth = cache_data[dataset_key]
    
    return truth, generated


def plot_failure_modes_figure(
    truth_underfit, gen_underfit,
    truth_smear, gen_smear,
    output_path,
    show_annotations=True
):
    """Create publication-quality two-panel failure modes figure.
    
    Args:
        truth_underfit: Ground truth for underfitting panel
        gen_underfit: Generated samples for underfitting panel
        truth_smear: Ground truth for smearing panel
        gen_smear: Generated samples for smearing panel
        output_path: Output PDF path
        show_annotations: Whether to add explanatory annotations
    """
    # Publication settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': False,  # Set True if LaTeX available
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # =========================================================================
    # Panel (a): Underfitting - Spurious modes and missing peaks
    # =========================================================================
    truth_flat = np.asarray(truth_underfit).flatten()
    gen_flat = np.asarray(gen_underfit).flatten()
    
    # Compute shared bins
    vmin = min(np.percentile(truth_flat, 0.5), np.percentile(gen_flat, 0.5))
    vmax = max(np.percentile(truth_flat, 99.5), np.percentile(gen_flat, 99.5))
    bins = np.linspace(vmin, vmax, 201)
    
    # Compute histograms
    counts_t, _ = np.histogram(truth_flat, bins=bins, density=True)
    counts_g, _ = np.histogram(gen_flat, bins=bins, density=True)
    
    # Plot ground truth (filled)
    ax1.fill_between(bins[:-1], counts_t, step="post", 
                     alpha=0.6, color=COLOR_TRUTH, label='Ground truth')
    
    # Plot generated (outline + light fill)
    ax1.fill_between(bins[:-1], counts_g, step="post",
                     alpha=0.35, color=COLOR_GEN_FILL)
    ax1.step(bins[:-1], counts_g, where="post", 
             color=COLOR_GENERATED, linewidth=1.8, label='Generated')
    
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('(a) Underfitting Artifacts', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_xlim(vmin, vmax)
    ax1.set_ylim(bottom=0)
    
    # Add annotations for underfitting panel
    if show_annotations:
        # Annotation style
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.9)
        arrow_props = dict(arrowstyle='->', color=COLOR_GENERATED, lw=1.5)
        
        # Spurious sharp spikes annotation - point to first spike around x=1.5
        ax1.annotate('Spurious\nspikes', 
                     xy=(1.5, 0.10), 
                     xytext=(1.0, 0.02),
                     fontsize=14, ha='center',
                     bbox=bbox_props,
                     arrowprops=arrow_props)
        
        # Blended peaks annotation (peaks 2 and 3 merged)
        ax1.annotate('Blended peaks\n(modes merged)', 
                     xy=(4.0, 0.12), 
                     xytext=(4.5, 0.03),
                     fontsize=14, ha='center',
                     bbox=bbox_props,
                     arrowprops=arrow_props)
        
        # Smeared peaks annotation - keep inside plot area
        ax1.annotate('Smeared peak', 
                     xy=(0.1, 0.20), 
                     xytext=(-0.5, 0.10),
                     fontsize=14, ha='center',
                     bbox=bbox_props,
                     arrowprops=arrow_props)
    
    # =========================================================================
    # Panel (b): Boundary Smearing - Intrinsic FM limitation
    # =========================================================================
    truth_flat2 = np.asarray(truth_smear).flatten()
    gen_flat2 = np.asarray(gen_smear).flatten()
    
    # Bins focused on the boundary region - fixed range [-0.5, 1.5]
    vmin2, vmax2 = 0.5, 1.5
    bins2 = np.linspace(vmin2, vmax2, 201)
    
    counts_t2, _ = np.histogram(truth_flat2, bins=bins2, density=True)
    counts_g2, _ = np.histogram(gen_flat2, bins=bins2, density=True)
    
    # Plot
    ax2.fill_between(bins2[:-1], counts_t2, step="post",
                     alpha=0.6, color=COLOR_TRUTH, label='Ground truth')
    ax2.fill_between(bins2[:-1], counts_g2, step="post",
                     alpha=0.35, color=COLOR_GEN_FILL)
    ax2.step(bins2[:-1], counts_g2, where="post",
             color=COLOR_GENERATED, linewidth=1.8, label='Generated')
    
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('(b) Boundary Smearing Effect', fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_xlim(vmin2, vmax2)
    ax2.set_ylim(bottom=0)
    
    # Add vertical line at cutoff threshold
    cutoff = 1.0
    ax2.axvline(x=cutoff, color=COLOR_CONTEXT, linestyle='--', linewidth=1.5, alpha=0.7)
    
    if show_annotations:
        # Boundary leakage annotation - FM intrinsic error
        ax2.annotate('Boundary leakage\n(intrinsic FM error)', 
                     xy=(1.1, 0.08),
                     xytext=(1.25, 0.25),
                     fontsize=14, ha='center',
                     bbox=bbox_props,
                     arrowprops=arrow_props)
        
        # Hard cutoff label
        ax2.annotate('Hard cutoff', 
                     xy=(cutoff, 0.18),
                     xytext=(0.75, 0.35),
                     fontsize=14, ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', 
                              edgecolor='gray', alpha=0.9),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
        
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved failure modes figure to {output_path}")


def plot_failure_modes_three_panel(
    truth_underfit, gen_underfit,
    truth_smear, gen_smear,
    output_path,
    show_annotations=True
):
    """Alternative three-panel layout showing additional failure modes."""
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.9)
    arrow_props = dict(arrowstyle='->', color=COLOR_GENERATED, lw=1.5)
    
    # =========================================================================
    # Panel (a): Mode collapse / spurious modes
    # =========================================================================
    ax = axes[0]
    
    truth_flat = np.asarray(truth_underfit).flatten()
    gen_flat = np.asarray(gen_underfit).flatten()
    
    vmin = min(np.percentile(truth_flat, 0.5), np.percentile(gen_flat, 0.5))
    vmax = max(np.percentile(truth_flat, 99.5), np.percentile(gen_flat, 99.5))
    bins = np.linspace(vmin, vmax, 151)
    
    counts_t, _ = np.histogram(truth_flat, bins=bins, density=True)
    counts_g, _ = np.histogram(gen_flat, bins=bins, density=True)
    
    ax.fill_between(bins[:-1], counts_t, step="post", 
                    alpha=0.6, color=COLOR_TRUTH, label='Ground truth')
    ax.fill_between(bins[:-1], counts_g, step="post", alpha=0.3, color=COLOR_GEN_FILL)
    ax.step(bins[:-1], counts_g, where="post", color=COLOR_GENERATED, linewidth=1.5, 
            label='Generated')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('(a) Multimodal Underfitting', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(bottom=0)
    
    if show_annotations:
        ax.annotate('Spurious spikes', xy=(1.85, 0.10), xytext=(1.0, 0.02),
                   fontsize=14, ha='center', bbox=bbox_props, arrowprops=arrow_props)
        ax.annotate('Blended peaks', xy=(4.0, 0.10), xytext=(5.2, 0.03),
                   fontsize=14, ha='center', bbox=bbox_props, arrowprops=arrow_props)
    
    # =========================================================================
    # Panel (b): Boundary smearing
    # =========================================================================
    ax = axes[1]
    
    truth_flat2 = np.asarray(truth_smear).flatten()
    gen_flat2 = np.asarray(gen_smear).flatten()
    
    vmin2 = min(np.percentile(truth_flat2, 0.5), np.percentile(gen_flat2, 0.5))
    vmax2 = max(np.percentile(truth_flat2, 99.5), np.percentile(gen_flat2, 99.5))
    bins2 = np.linspace(vmin2, vmax2, 151)
    
    counts_t2, _ = np.histogram(truth_flat2, bins=bins2, density=True)
    counts_g2, _ = np.histogram(gen_flat2, bins=bins2, density=True)
    
    ax.fill_between(bins2[:-1], counts_t2, step="post",
                    alpha=0.6, color=COLOR_TRUTH, label='Ground truth')
    ax.fill_between(bins2[:-1], counts_g2, step="post", alpha=0.3, color=COLOR_GEN_FILL)
    ax.step(bins2[:-1], counts_g2, where="post", color=COLOR_GENERATED, linewidth=1.5,
            label='Generated')
    
    ax.axvline(x=1.0, color=COLOR_CONTEXT, linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('(b) Sharp Boundary Smearing', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(vmin2, vmax2)
    ax.set_ylim(bottom=0)
    
    if show_annotations:
        ax.annotate('Boundary leakage\n(intrinsic FM error)', xy=(1.3, 0.12), xytext=(1.8, 0.25),
                   fontsize=14, ha='center', bbox=bbox_props, arrowprops=arrow_props)
        ax.annotate('Cutoff', xy=(1.0, 0.20), xytext=(0.3, 0.32),
                   fontsize=14, ha='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0f0',
                            edgecolor='gray', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color=COLOR_CONTEXT, lw=1.0))
    
    # =========================================================================
    # Panel (c): High-frequency features / fine structure
    # =========================================================================
    ax = axes[2]
    
    # Generate high-frequency example
    rng = np.random.default_rng(456)
    n = 80000
    
    # Ground truth: Gaussian with narrow spikes
    base = rng.normal(0, 1.5, int(n * 0.7))
    spike1 = rng.normal(-1.5, 0.08, int(n * 0.1))
    spike2 = rng.normal(0.8, 0.08, int(n * 0.1))
    spike3 = rng.normal(2.2, 0.08, int(n * 0.1))
    truth_hf = np.concatenate([base, spike1, spike2, spike3])
    
    # Generated: spikes are smoothed/widened
    base_g = rng.normal(0, 1.6, int(n * 0.75))
    spike1_g = rng.normal(-1.5, 0.25, int(n * 0.08))
    spike2_g = rng.normal(0.8, 0.25, int(n * 0.09))
    spike3_g = rng.normal(2.2, 0.25, int(n * 0.08))
    gen_hf = np.concatenate([base_g, spike1_g, spike2_g, spike3_g])
    
    vmin3 = -5
    vmax3 = 5
    bins3 = np.linspace(vmin3, vmax3, 201)
    
    counts_t3, _ = np.histogram(truth_hf, bins=bins3, density=True)
    counts_g3, _ = np.histogram(gen_hf, bins=bins3, density=True)
    
    ax.fill_between(bins3[:-1], counts_t3, step="post",
                    alpha=0.6, color=COLOR_TRUTH, label='Ground truth')
    ax.fill_between(bins3[:-1], counts_g3, step="post", alpha=0.3, color=COLOR_GEN_FILL)
    ax.step(bins3[:-1], counts_g3, where="post", color=COLOR_GENERATED, linewidth=1.5,
            label='Generated')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('(c) High-Frequency Smoothing', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(vmin3, vmax3)
    ax.set_ylim(bottom=0)
    
    if show_annotations:
        ax.annotate('Broadened\nspike', xy=(-1.5, 0.22), xytext=(-3.0, 0.35),
                   fontsize=14, ha='center', bbox=bbox_props, arrowprops=arrow_props)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved three-panel failure modes figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality failure modes figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate figure with synthetic examples
    python scripts/6_failure_modes.py --synthetic --output figures/failure_modes.pdf

    # Use data from an actual run
    python scripts/6_failure_modes.py \\
        --underfitting-run outputs/generation/mock/triple_mixed_scale_5 \\
        --underfitting-epoch 9 \\
        --output figures/failure_modes.pdf

    # Three-panel layout
    python scripts/6_failure_modes.py --synthetic --three-panel \\
        --output figures/failure_modes_3panel.pdf
"""
    )
    
    parser.add_argument('--synthetic', action='store_true', default=True,
                       help='Generate synthetic examples (default: True)')
    parser.add_argument('--underfitting-run', type=str, default=None,
                       help='Path to run directory with underfitting example')
    parser.add_argument('--underfitting-epoch', type=str, default='009',
                       help='Epoch suffix for early checkpoint (default: 009)')
    parser.add_argument('--smearing-run', type=str, default=None,
                       help='Path to run directory with smearing example')
    parser.add_argument('--smearing-epoch', type=str, default='best',
                       help='Epoch suffix for smearing example (default: best)')
    parser.add_argument('--output', type=str, default='figures/failure_modes.pdf',
                       help='Output PDF path')
    parser.add_argument('--three-panel', action='store_true',
                       help='Generate three-panel layout')
    parser.add_argument('--no-annotations', action='store_true',
                       help='Disable annotations')
    parser.add_argument('--n-samples', type=int, default=100000,
                       help='Number of samples for synthetic examples')
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.underfitting_run:
        print(f"Loading underfitting example from {args.underfitting_run}")
        truth_underfit, gen_underfit = load_from_checkpoint(
            args.underfitting_run, args.underfitting_epoch
        )
        # Flatten if multi-dimensional
        if len(truth_underfit.shape) > 1:
            # Select one interesting column (e.g., column 0)
            truth_underfit = truth_underfit[:, 0]
            gen_underfit = gen_underfit[:, 0]
    else:
        print("Generating synthetic underfitting example...")
        truth_underfit, gen_underfit = generate_synthetic_underfitting_example(
            n_samples=args.n_samples
        )
    
    if args.smearing_run:
        print(f"Loading smearing example from {args.smearing_run}")
        truth_smear, gen_smear = load_from_checkpoint(
            args.smearing_run, args.smearing_epoch
        )
        if len(truth_smear.shape) > 1:
            truth_smear = truth_smear[:, 0]
            gen_smear = gen_smear[:, 0]
    else:
        print("Generating synthetic smearing example...")
        truth_smear, gen_smear = generate_synthetic_smearing_example(
            n_samples=args.n_samples
        )
    
    # Generate figure
    print(f"Generating figure...")
    if args.three_panel:
        plot_failure_modes_three_panel(
            truth_underfit, gen_underfit,
            truth_smear, gen_smear,
            args.output,
            show_annotations=not args.no_annotations
        )
    else:
        plot_failure_modes_figure(
            truth_underfit, gen_underfit,
            truth_smear, gen_smear,
            args.output,
            show_annotations=not args.no_annotations
        )


if __name__ == '__main__':
    main()
