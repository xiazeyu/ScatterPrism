"""Plotting and checkpoint-handling utilities for ScatterPrism.

Provides:
    * MC-POM-aware distribution plots (1-D, 24-channel grids, per-group).
    * Flow-trajectory visualisation for trained CFM checkpoints.
    * Checkpoint-evolution grids and overlays for training-time monitoring.
    * Helpers for loading models and per-dimension Gaussian fits.
"""

import logging
import math
import os
import re
from pathlib import Path

import hydra.utils as _hu
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import norm as _norm
from torch.utils.data import random_split

from scatterprism.models import CFM
from scatterprism.transforms import BaseTransform

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CVD-safe color palette for consistent plotting across all figures
# ---------------------------------------------------------------------------
COLOR_TRUTH = '#0072B2'       # Blue - reference/truth distributions
COLOR_GENERATED = '#E69F00'   # Orange/amber - generated/model output
COLOR_DETECTOR = '#CC79A7'    # Pink/magenta - detector/degraded data
COLOR_CONTEXT = '#808080'     # Medium gray - reference lines
COLOR_GEN_FILL = '#F5C97A'    # Lighter orange - fill for generated
COLOR_MARKER = '#D55E00'      # Vermillion - boundary markers
COLOR_RATIO = '#404040'       # Dark gray - ratio line in ratio sub-panels
COLOR_BAND = '#CCCCCC'        # Light gray - +/-10% uncertainty band

# ---------------------------------------------------------------------------
# Shared MC-POM column / scale metadata used by all MC-POM plotting functions
# ---------------------------------------------------------------------------
# Delta function columns excluded: photon_y, target_proton_y, recoil_proton_x, recoil_proton_y
_MCPOM_COLS_LIST = [
    't', 'mpipi', 'costh', 'phi',
    'photon_t', 'photon_x', 'photon_z',
    'target_proton_t', 'target_proton_x', 'target_proton_z',
    'pi_plus_t', 'pi_plus_x', 'pi_plus_y', 'pi_plus_z',
    'pi_minus_t', 'pi_minus_x', 'pi_minus_y', 'pi_minus_z',
    'recoil_proton_t', 'recoil_proton_z',
]

# Full list including delta channels (for data loading/indexing)
_MCPOM_COLS_LIST_FULL = [
    't', 'mpipi', 'costh', 'phi',
    'photon_t', 'photon_x', 'photon_y', 'photon_z',
    'target_proton_t', 'target_proton_x', 'target_proton_y', 'target_proton_z',
    'pi_plus_t', 'pi_plus_x', 'pi_plus_y', 'pi_plus_z',
    'pi_minus_t', 'pi_minus_x', 'pi_minus_y', 'pi_minus_z',
    'recoil_proton_t', 'recoil_proton_x', 'recoil_proton_y', 'recoil_proton_z',
]

# Index mapping: column name -> index in full 24-column data
_MCPOM_COL_INDEX = {name: idx for idx, name in enumerate(_MCPOM_COLS_LIST_FULL)}

_MCPOM_SCALES: dict = {
    't':                 {'xlim': (-1.1, -0.3)},
    'mpipi':             {'xlim': (0.3, 1.5)},
    'costh':             {'xlim': (-1.0, 1.0)},
    'phi':               {'xlim': (0, 2 * np.pi)},
    'photon_t':          {'xlim': (0.6, 1.45)},
    'photon_x':          {'xlim': (-1.1, -0.6)},
    'photon_y':          {'xlim': (-0.1, 0.1)},
    'photon_z':          {'xlim': (-1.2, 0.6)},
    'target_proton_t':   {'xlim': (2, 8.2)},
    'target_proton_x':   {'xlim': (0.6, 1.1)},
    'target_proton_y':   {'xlim': (-0.1, 0.1)},
    'target_proton_z':   {'xlim': (-8.1, -1.5)},
    'pi_plus_t':         {'xlim': (0.15, 0.8)},
    'pi_plus_x':         {'xlim': (-0.8, 0.8)},
    'pi_plus_y':         {'xlim': (-0.8, 0.8)},
    'pi_plus_z':         {'xlim': (-0.8, 0.8)},
    'pi_minus_t':        {'xlim': (0.15, 0.8)},
    'pi_minus_x':        {'xlim': (-0.8, 0.8)},
    'pi_minus_y':        {'xlim': (-0.8, 0.8)},
    'pi_minus_z':        {'xlim': (-0.8, 0.8)},
    'recoil_proton_t':   {'xlim': (1.3, 8.5)},
    'recoil_proton_x':   {'xlim': (-0.1, 0.1)},
    'recoil_proton_y':   {'xlim': (-0.1, 0.1)},
    'recoil_proton_z':   {'xlim': (-8.5, -1.3)},
}

# LaTeX labels for MC-POM observables (for paper-quality figures)
_MCPOM_LABELS: dict = {
    # Derived Observables
    't': r'$t$',
    'mpipi': r'$M_{\pi\pi}$',
    'costh': r'$\cos\theta$',
    'phi': r'$\phi$',
    # Incident Photon (γ)
    'photon_t': r'$E_{\gamma}$',
    'photon_x': r'$p_{\gamma x}$',
    'photon_y': r'$p_{\gamma y}$',
    'photon_z': r'$p_{\gamma z}$',
    # Target Proton (p₁)
    'target_proton_t': r'$E_{p_1}$',
    'target_proton_x': r'$p_{1x}$',
    'target_proton_y': r'$p_{1y}$',
    'target_proton_z': r'$p_{1z}$',
    # Positive Pion (π⁺)
    'pi_plus_t': r'$E_{\pi^+}$',
    'pi_plus_x': r'$p_{\pi^+ x}$',
    'pi_plus_y': r'$p_{\pi^+ y}$',
    'pi_plus_z': r'$p_{\pi^+ z}$',
    # Negative Pion (π⁻)
    'pi_minus_t': r'$E_{\pi^-}$',
    'pi_minus_x': r'$p_{\pi^- x}$',
    'pi_minus_y': r'$p_{\pi^- y}$',
    'pi_minus_z': r'$p_{\pi^- z}$',
    # Recoil Proton (p₂)
    'recoil_proton_t': r'$E_{p_2}$',
    'recoil_proton_x': r'$p_{2x}$',
    'recoil_proton_y': r'$p_{2y}$',
    'recoil_proton_z': r'$p_{2z}$',
}


def plot_flatten_dataset_distribution(dataset, output_dir, dataset_name):
    """Plot the flattened (1-D) density of all values in a dataset.

    Extracts data from various dataset formats (pre-transform, .data, .x, or
    indexed), flattens to 1-D, and saves a single density histogram.

    Args:
        dataset:      Dataset object or array-like.  Tries ``pre_transform_data``,
                      ``data``, ``x`` attributes in order; falls back to indexing.
        output_dir:   Directory to save the plot (created if needed).
        dataset_name: Label used for the plot title and output filename.

    Saves:
        ``<output_dir>/<dataset_name>_distribution.png``
    """
    # Prefer pre-transform data (original scale) when available
    if hasattr(dataset, 'pre_transform_data') and dataset.pre_transform_data is not None:
        data = dataset.pre_transform_data
    elif hasattr(dataset, 'data'):
        data = dataset.data
    elif hasattr(dataset, 'x'):
        data = dataset.x
    else:
        data = [dataset[i] for i in range(min(len(dataset), 1000))]

    data = np.array(data)
    data_flat = data.flatten()

    plt.figure(figsize=(10, 6))
    # Style update for single plot as well
    counts, bin_edges = np.histogram(data_flat, bins=200, density=True)
    plt.fill_between(bin_edges[:-1], counts,
                     step="post", alpha=0.5, color=COLOR_TRUTH)

    plt.title(f'Distribution of {dataset_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, linestyle=':')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{dataset_name}_distribution.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    log.info(f"Saved distribution plot to {save_path}")


def plot_distributions_diff_1d(truth, generated, output_path, label_a='Truth', label_b='Generated'):
    """Plot overlaid 1-D distributions for truth vs generated (mock datasets).

    Creates a single plot with two overlaid histograms comparing truth and
    generated distributions, suitable for low-dimensional mock datasets.

    Args:
        truth:       Numpy array (will be flattened to 1-D).
        generated:   Numpy array (will be flattened to 1-D).
        output_path: Full path to save the plot.
        label_a:     Legend label for truth data.
        label_b:     Legend label for generated data.
    """
    truth_flat = np.array(truth).flatten()
    gen_flat = np.array(generated).flatten()

    # Compute shared bin edges based on combined range
    combined = np.concatenate([truth_flat, gen_flat])
    vmin, vmax = np.percentile(combined, [0.5, 99.5])
    bins = np.linspace(vmin, vmax, 201)

    plt.figure(figsize=(10, 6))

    # Plot truth
    counts_t, _ = np.histogram(truth_flat, bins=bins, density=True)
    plt.fill_between(bins[:-1], counts_t, step="post", alpha=0.5, color=COLOR_TRUTH, label=label_a)

    # Plot generated
    counts_g, _ = np.histogram(gen_flat, bins=bins, density=True)
    plt.step(bins[:-1], counts_g, where="post", color=COLOR_GENERATED, linewidth=1.5, label=label_b)

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{label_a} vs {label_b}')
    plt.legend()
    plt.grid(True, linestyle=':')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    log.info(f"Saved 1D diff plot to {output_path}")


def plot_distributions_mcpom(dataset, output_filepath):
    """Plot per-column distributions for a single MC-POM 24-column dataset.

    Produces a 6x4 grid of normalised histograms, one per MC-POM column.

    Args:
        dataset:         Numpy array of shape ``[N, 24]``.
        output_filepath: Path for the saved figure.
    """
    cols_to_plot = _MCPOM_COLS_LIST_FULL

    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 18))
    axes = axes.flatten()

    num_bins = 200

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        data = dataset[:, i]

        bin_range = (np.min(data), np.max(data))

        bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)

        counts, bin_edges = np.histogram(data, bins=bins, density=True)

        ax.fill_between(bin_edges[:-1], counts,
                        step="post", alpha=0.5, color=COLOR_TRUTH)

        ax.set_title(f'Distribution of {col}', fontsize=12)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10, color=COLOR_TRUTH)
        ax.grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Plot saved as {output_filepath}")


def plot_distributions_diff_mcpom(dataset_a, dataset_b, output_filepath, label_a='Original', label_b='Detector'):
    """Plot an overlay comparison of two MC-POM datasets across all 24 columns.

    Convenience wrapper around :func:`plot_distributions_multiple_mcpom` for
    exactly two datasets with distinct fill/line styling.

    Args:
        dataset_a:       First dataset, ``[N, 24]`` array.
        dataset_b:       Second dataset, ``[M, 24]`` array.
        output_filepath: Path for the saved figure.
        label_a:         Legend label for the first dataset.
        label_b:         Legend label for the second dataset.
    """
    cols_to_plot = _MCPOM_COLS_LIST_FULL

    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 18))
    axes = axes.flatten()

    scales = _MCPOM_SCALES
    num_bins = 200

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        data_a = dataset_a[:, i]
        data_b = dataset_b[:, i]

        if col in scales and 'xlim' in scales[col]:
            bin_range = scales[col]['xlim']
        else:
            combined_min = min(np.min(data_a), np.min(data_b))
            combined_max = max(np.max(data_a), np.max(data_b))
            bin_range = (combined_min, combined_max)

        bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)

        counts_a, bin_edges = np.histogram(data_a, bins=bins, density=True)
        counts_b, _ = np.histogram(data_b, bins=bins, density=True)

        ax.fill_between(bin_edges[:-1], counts_a, step="post",
                        alpha=0.5, color=COLOR_TRUTH, label=label_a)
        ax.step(bin_edges[:-1], counts_b, where="post",
                color=COLOR_GENERATED, linewidth=1.5, linestyle='--', label=label_b)
        ax.fill_between(bin_edges[:-1], counts_b, step="post",
                        alpha=0.15, color=COLOR_GENERATED)

        ax.set_title(f'Distribution for {col}', fontsize=12)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Normalized Frequency', fontsize=10)
        ax.grid(True, linestyle=':', linewidth=0.5)

        if col in scales and 'xlim' in scales[col]:
            ax.set_xlim(scales[col]['xlim'])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               fontsize=14, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_filepath) or '.', exist_ok=True)
    plt.savefig(output_filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Comparison plot with colored difference saved as {output_filepath}")


def plot_distributions_multiple_mcpom(datasets, output_filepath, *,
                                      cols: dict[str, int] | None = None,
                                      extra_text: list[str] | None = None):
    """Plot and compare distributions of multiple datasets across MC-POM columns.

    Produces a grid of normalised histogram overlays (one subplot per column)
    with a shared legend.  The first dataset is drawn with a solid line; all
    subsequent datasets use dashed lines.

    Args:
        datasets:        ``{label: array}`` dict.  Each array is ``[N, 24]``.
        output_filepath: Path to save the output image.
        cols:            Optional ``{col_name: data_column_index}`` mapping.
                         When *None* (default), all 24 MC-POM columns are plotted
                         in a 6x4 grid.  Pass a custom dict to plot a subset
                         (grid size is computed automatically with 4 columns).
        extra_text:      Optional list of annotation strings indexed by *data
                         column index*.  When provided, the string for each
                         plotted column is shown below the subplot title.
    """
    if cols is None:
        cols = _MCPOM_COL_INDEX

    n_plots = len(cols)
    ncols_grid = 4
    nrows_grid = math.ceil(n_plots / ncols_grid)

    fig, axes = plt.subplots(nrows=nrows_grid, ncols=ncols_grid, figsize=(20, 3 * nrows_grid + 6))
    axes = np.array(axes).flatten()

    scales = _MCPOM_SCALES
    num_bins = 200
    colors = plt.get_cmap('tab10', len(datasets))

    for i, (col, data_idx) in enumerate(cols.items()):
        ax = axes[i]

        # Determine bin range across all datasets for this column
        if col in scales and 'xlim' in scales[col]:
            bin_range = scales[col]['xlim']
        else:
            all_data_for_col = [ds[:, data_idx] for ds in datasets.values()]
            if not all_data_for_col or all(len(d) == 0 for d in all_data_for_col):
                bin_range = (0, 1)
            else:
                combined_min = min(np.min(d) for d in all_data_for_col if len(d) > 0)
                combined_max = max(np.max(d) for d in all_data_for_col if len(d) > 0)
                if combined_min == combined_max:
                    bin_range = (combined_min - 0.5, combined_max + 0.5)
                else:
                    bin_range = (combined_min, combined_max)

        bins = np.linspace(bin_range[0], bin_range[1], num_bins + 1)

        for k, (label, data) in enumerate(datasets.items()):
            color = colors(k)
            column_data = data[:, data_idx]

            if len(column_data) > 0:
                counts, bin_edges = np.histogram(column_data, bins=bins, density=True)
                linestyle = '--' if k > 0 else '-'
                ax.fill_between(bin_edges[:-1], counts,
                                step="post", alpha=0.15, color=color)
                ax.step(bin_edges[:-1], counts, where="post", color=color,
                        linewidth=1.5, linestyle=linestyle, label=label)
            else:
                log.warning(f"Dataset '{label}' has no data for column '{col}'. Skipping.")

        title_pad = 35 if extra_text is not None else 6
        ax.set_title(f'Distribution for {col}', fontsize=12, pad=title_pad)
        if extra_text is not None and data_idx < len(extra_text):
            ax.text(0.5, 1.02, extra_text[data_idx],
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax.transAxes,
                    fontsize=11, color='black')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Normalized Frequency', fontsize=10)
        ax.grid(True, linestyle=':', linewidth=0.5)

        if col in scales and 'xlim' in scales[col]:
            ax.set_xlim(scales[col]['xlim'])

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(datasets), 4),
               fontsize=14, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Comparison plot saved as {output_filepath}")


# ============================================================
# Generated samples plotting (from plot_generated_samples.py)
# ============================================================

# Column metadata for the MC POM 24-column dataset
_MCPOM_COLUMNS = [
    (0,  "t",       r"$t$",                       r"GeV$^2$"),
    (1,  "mpipi",   r"$m_{\pi\pi}$",              r"GeV"),
    (2,  "costh",   r"$\cos\theta$",              ""),
    (3,  "phi",     r"$\phi$",                    "rad"),
    (4,  "q0",      r"$E_\gamma$",               "GeV"),
    (5,  "q1",      r"$p_x^\gamma$",             "GeV"),
    (6,  "q2",      r"$p_y^\gamma$",             "GeV"),
    (7,  "q3",      r"$p_z^\gamma$",             "GeV"),
    (8,  "p10",     r"$E_{p_1}$",               "GeV"),
    (9,  "p11",     r"$p_x^{p_1}$",             "GeV"),
    (10, "p12",     r"$p_y^{p_1}$",             "GeV"),
    (11, "p13",     r"$p_z^{p_1}$",             "GeV"),
    (12, "k10",     r"$E_{\pi^+}$",             "GeV"),
    (13, "k11",     r"$p_x^{\pi^+}$",           "GeV"),
    (14, "k12",     r"$p_y^{\pi^+}$",           "GeV"),
    (15, "k13",     r"$p_z^{\pi^+}$",           "GeV"),
    (16, "k20",     r"$E_{\pi^-}$",             "GeV"),
    (17, "k21",     r"$p_x^{\pi^-}$",           "GeV"),
    (18, "k22",     r"$p_y^{\pi^-}$",           "GeV"),
    (19, "k23",     r"$p_z^{\pi^-}$",           "GeV"),
    (20, "p20",     r"$E_{p_2}$",               "GeV"),
    (21, "p21",     r"$p_x^{p_2}$",             "GeV"),
    (22, "p22",     r"$p_y^{p_2}$",             "GeV"),
    (23, "p23",     r"$p_z^{p_2}$",             "GeV"),
]

_MCPOM_GROUPS = {
    "Physics variables":  [0, 1, 2, 3],
    "Photon (gamma)":     [4, 5, 6, 7],
    "Target proton (p1)": [8, 9, 10, 11],
    "pi+":                [12, 13, 14, 15],
    "pi-":                [16, 17, 18, 19],
    "Recoil proton (p2)": [20, 21, 22, 23],
}


def _get_plot_range(col_idx: int, gen: np.ndarray, truth=None):
    """Compute a clipped plotting range for a single column.

    Uses the 0.5th and 99.5th percentiles of the generated (and optionally
    truth) data, expanded by a 5 % margin on each side.

    Args:
        col_idx: Column index to evaluate.
        gen:     Generated samples array ``[N, D]``.
        truth:   Optional truth array ``[M, D]``.

    Returns:
        ``(lo, hi)`` float tuple suitable for histogram bin range.
    """
    data = gen[:, col_idx]
    lo, hi = np.percentile(data, 0.5), np.percentile(data, 99.5)
    if truth is not None:
        t = truth[:, col_idx]
        lo = min(lo, np.percentile(t, 0.5))
        hi = max(hi, np.percentile(t, 99.5))
    margin = 0.05 * (hi - lo) if hi > lo else 1e-6
    return lo - margin, hi + margin


def _plot_sample_group(group_name: str, col_indices: list, gen: np.ndarray,
                       truth, bins: int, save_path=None):
    """Plot one variable group: generated vs truth normalised histograms."""
    ncols = 2
    nrows = (len(col_indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    fig.suptitle(group_name, fontsize=14, fontweight="bold", y=1.01)

    for ax, col_idx in zip(axes, col_indices):
        _, name, label, unit = _MCPOM_COLUMNS[col_idx]
        xlabel = f"{label}  [{unit}]" if unit else label
        lo, hi = _get_plot_range(col_idx, gen, truth)
        bin_edges = np.linspace(lo, hi, bins + 1)

        if truth is not None:
            t_counts, _ = np.histogram(truth[:, col_idx], bins=bin_edges)
            t_norm = t_counts / t_counts.sum()
            ax.step(bin_edges[:-1], t_norm, where="post",
                    color=COLOR_TRUTH, linewidth=1.5, label="Ground truth")
            ax.fill_between(bin_edges[:-1], t_norm, step="post",
                            color=COLOR_TRUTH, alpha=0.25)

        g_counts, _ = np.histogram(gen[:, col_idx], bins=bin_edges)
        g_norm = g_counts / g_counts.sum()
        ax.step(bin_edges[:-1], g_norm, where="post",
                color=COLOR_GENERATED, linewidth=1.5, label="Generated", linestyle="--")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Normalised counts", fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    for ax in axes[len(col_indices):]:
        ax.set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        log.info(f"Saved group plot to {save_path}")
    plt.close(fig)


def plot_generated_vs_truth(gen: np.ndarray, truth, output_dir: str,
                             bins: int = 80, max_samples: int = 500_000):
    """Plot generated samples vs truth for all MC POM variable groups.

    Saves one PDF per group into output_dir/generated_vs_truth/.

    Args:
        gen:         Generated samples array [N, 24]
        truth:       Truth (pre-transform) array [M, 24], or None
        output_dir:  Directory to save plots
        bins:        Number of histogram bins
        max_samples: Subsample limit for performance
    """
    rng = np.random.default_rng(42)
    if max_samples and len(gen) > max_samples:
        gen = gen[rng.choice(len(gen), max_samples, replace=False)]
    if truth is not None and max_samples and len(truth) > max_samples:
        truth = truth[rng.choice(len(truth), max_samples, replace=False)]

    out_dir = os.path.join(output_dir, "generated_vs_truth")
    os.makedirs(out_dir, exist_ok=True)

    for group_name, col_indices in _MCPOM_GROUPS.items():
        safe_name = (group_name.replace(" ", "_").replace("(", "").replace(")", "")
                     .replace("/", ""))
        save_path = os.path.join(out_dir, f"gen_vs_truth_{safe_name}.pdf")
        log.info(f"Plotting group: {group_name}")
        _plot_sample_group(group_name, col_indices, gen, truth,
                           bins=bins, save_path=save_path)

    log.info(f"All generated-vs-truth figures saved to {out_dir}")


# ============================================================
# Flow trajectory analysis (from plot_flow_trajectory.py)
# ============================================================

def find_best_checkpoint(run_dir, *, allow_fallback: bool = False):
    """Locate the best model checkpoint inside a run directory.

    Default (``allow_fallback=False``): only ``checkpoints/best.ckpt`` is
    accepted; raises :class:`FileNotFoundError` if it does not exist.

    With ``allow_fallback=True``: search
    ``final_model.ckpt`` → ``checkpoints/best.ckpt`` → ``checkpoints/last.ckpt``
    (``final_model.ckpt`` is the in-training copy of best; ``last.ckpt`` is the
    final-epoch fallback for partial runs).

    Args:
        run_dir:        Path-like run directory.
        allow_fallback: Enable the three-way fallback search.

    Returns:
        ``pathlib.Path`` to the located checkpoint.

    Raises:
        FileNotFoundError: If no acceptable checkpoint exists.
    """
    run_dir = Path(run_dir)
    if allow_fallback:
        candidates = [
            run_dir / "final_model.ckpt",
            run_dir / "checkpoints" / "best.ckpt",
            run_dir / "checkpoints" / "last.ckpt",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir}; tried "
            f"final_model.ckpt, checkpoints/best.ckpt, checkpoints/last.ckpt"
        )
    best = run_dir / "checkpoints" / "best.ckpt"
    if not best.exists():
        raise FileNotFoundError(
            f"checkpoints/best.ckpt not found in {run_dir}. "
            f"Pass --all-available-ckpts to enable fallback search "
            f"(final_model.ckpt -> checkpoints/best.ckpt -> checkpoints/last.ckpt)."
        )
    return best


def find_best_samples(run_dir, *, allow_fallback: bool = False):
    """Locate the best-checkpoint generated samples ``.npz`` in a run directory.

    Default (``allow_fallback=False``): only ``generated_samples_best.npz`` is
    accepted; raises :class:`FileNotFoundError` if it does not exist.

    With ``allow_fallback=True``: search
    ``generated_samples_best.npz`` → ``generated_samples_last.npz``.

    Args:
        run_dir:        Path-like run directory.
        allow_fallback: Enable the two-way fallback search.

    Returns:
        ``pathlib.Path`` to the located samples file.

    Raises:
        FileNotFoundError: If no acceptable samples file exists.
    """
    run_dir = Path(run_dir)
    best = run_dir / "generated_samples_best.npz"
    if best.exists():
        return best
    if allow_fallback:
        last = run_dir / "generated_samples_last.npz"
        if last.exists():
            return last
        raise FileNotFoundError(
            f"No samples found in {run_dir}; tried "
            f"generated_samples_best.npz, generated_samples_last.npz"
        )
    raise FileNotFoundError(
        f"generated_samples_best.npz not found in {run_dir}. "
        f"Pass --all-available-ckpts to enable fallback search "
        f"(generated_samples_best.npz -> generated_samples_last.npz)."
    )


def load_model_from_checkpoint(checkpoint_path: str, device):
    """Load a trained model from a ``.ckpt`` file.

    Prefers the ``model_target`` injected by :class:`CheckpointMetadataCallback`;
    falls back to :class:`scatterprism.models.CFM` when missing.

    Returns:
        ``(model, transform)`` — eval-mode Lightning model on *device* and the
        deserialised transform (or ``None`` if none was stored).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_target' in checkpoint:
        model_target = checkpoint['model_target']
        model_class = _hu.get_class(model_target)
        log.info(f"Auto-detected model type from checkpoint: {model_target}")
    else:
        model_class = CFM
        log.info(f"Heuristic model detection: {model_class.__name__}")

    model = model_class.load_from_checkpoint(
        checkpoint_path, map_location=device, weights_only=False,
    )
    model.eval()
    model.to(device)

    transform = None
    if 'transform_state' in checkpoint:
        transform = BaseTransform.deserialize(checkpoint['transform_state'])
        if transform is not None:
            log.info("Loaded transform from checkpoint")

    return model, transform


def get_flow_trajectory(model, n_generate: int, num_steps: int, device):
    """Generate a flow trajectory from ``t=0`` (Gaussian) to ``t=1`` (target).

    Returns:
        ``(trajectory, t_values)`` — numpy arrays of shape
        ``[num_steps, n_generate, data_dim]`` and ``[num_steps]``.
    """
    x0 = torch.randn(n_generate, model.data_dim, device=device)
    trajectory = model.get_trajectory(x0, num_steps=num_steps)
    t_values = np.linspace(0, 1, num_steps)
    return trajectory.cpu().numpy(), t_values


def plot_flow_1d_density(trajectory, t_values, dim=0, save_path=None, num_bins=200):
    """Plot 1D flow evolution as a 2D density heatmap (time on x-axis, value on y-axis).

    Marginal distributions at t=0 and t=1 are shown on the left and right panels.
    """
    num_steps, _, _ = trajectory.shape
    v_min, v_max = -4, 4

    density_map = np.zeros((num_bins, num_steps))
    bin_edges = np.linspace(v_min, v_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for step_idx in range(num_steps):
        hist, _ = np.histogram(trajectory[step_idx, :, dim], bins=bin_edges, density=True)
        density_map[:, step_idx] = hist

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.12, 1, 0.12, 0.05], wspace=0.05)
    ax_left = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2])
    ax_cbar = fig.add_subplot(gs[3])

    im = ax_main.imshow(density_map, aspect='auto', origin='lower',
                        extent=[0, 1, v_min, v_max],
                        cmap='Blues', interpolation='bicubic')
    ax_main.set_xlabel(r'Time $t$')
    ax_main.set_yticklabels([])

    hist_t0 = density_map[:, 0]
    ax_left.fill_betweenx(bin_centers, 0, hist_t0, alpha=0.7, color=COLOR_TRUTH)
    ax_left.plot(hist_t0, bin_centers, color=COLOR_TRUTH, linewidth=1.5)
    ax_left.set_ylim(v_min, v_max)
    ax_left.set_xlim(ax_left.get_xlim()[::-1])
    ax_left.set_ylabel('Value')
    ax_left.set_xlabel('Density')
    ax_left.set_xticklabels([])
    ax_left.set_title(r'$t=0$')

    hist_t1 = density_map[:, -1]
    ax_right.fill_betweenx(bin_centers, 0, hist_t1, alpha=0.7, color=COLOR_TRUTH)
    ax_right.plot(hist_t1, bin_centers, color=COLOR_TRUTH, linewidth=1.5)
    ax_right.set_ylim(v_min, v_max)
    ax_right.set_xlabel('Density')
    ax_right.set_xticklabels([])
    ax_right.set_title(r'$t=1$')
    ax_right.set_yticklabels([])

    plt.colorbar(im, cax=ax_cbar, label='Density')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        log.info(f"Saved flow density plot to {save_path}")
    plt.close(fig)


def plot_flow_trajectory_for_checkpoint(checkpoint_path: str, output_dir: str,
                                         n_generate: int = 50000,
                                         num_steps: int = 100,
                                         dims=(0, 1),
                                         plot_type: str = 'all',
                                         device=None,
                                         save_format: str = 'png'):
    """Generate and save flow trajectory plots for a single checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        output_dir:      Directory to save figures
        n_generate:     Number of trajectory samples
        num_steps:       ODE integration time steps
        dims:            Two dimensions to use for 2-D scatter / marginal plots
        plot_type:       One of 'scatter', 'density', 'marginal', 'all'
        device:          torch.device or None (auto-detect)
        save_format:     Output format ('png' or 'pdf')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Loading model from {checkpoint_path} ...")
    model, _ = load_model_from_checkpoint(checkpoint_path, device)
    log.info(f"Model loaded. Data dimension: {model.data_dim}")

    log.info(f"Generating trajectory: {n_generate} samples, {num_steps} steps ...")
    trajectory, t_values = get_flow_trajectory(model, n_generate, num_steps, device)

    data_dim = model.data_dim
    is_1d = (data_dim == 1)
    # Accept 1-element dims for 1-D data; pad with 0 so the 2-D-only branches
    # still have two indices even though they will be skipped when is_1d.
    if len(dims) == 1:
        dim1 = dims[0]
        dim2 = dims[0]
    else:
        dim1, dim2 = dims[0], dims[1]

    if plot_type in ('density', 'all'):
        if is_1d:
            plot_flow_1d_density(
                trajectory, t_values, dim=0,
                save_path=os.path.join(output_dir, f'flow_density_1d.{save_format}'),
            )
        else:
            num_time_points = 6
            time_indices = np.linspace(0, trajectory.shape[0] - 1,
                                       num_time_points, dtype=int)
            ncols, nrows = 3, (num_time_points + 2) // 3
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            axes = np.array(axes).flatten()
            for ax, idx in zip(axes, time_indices):
                t = t_values[idx]
                ax.scatter(trajectory[idx, :, dim1], trajectory[idx, :, dim2],
                           alpha=0.3, s=5, c=COLOR_TRUTH, rasterized=True)
                ax.set_title(f't = {t:.2f}')
                ax.set_xlabel(f'Dim {dim1}')
                ax.set_ylabel(f'Dim {dim2}')
                ax.grid(True, alpha=0.3)
            for ax in axes[num_time_points:]:
                ax.set_visible(False)
            plt.tight_layout()
            sp = os.path.join(output_dir, f'flow_scatter.{save_format}')
            plt.savefig(sp, dpi=300, bbox_inches='tight', format=save_format)
            plt.close(fig)
            log.info(f"Saved flow scatter plot to {sp}")

    if plot_type in ('marginal', 'all'):
        valid_dims = list(dict.fromkeys(d for d in (dim1, dim2) if d < data_dim)) or [0]
        for dim in valid_dims:
            num_time_points = 6
            time_indices = np.linspace(0, trajectory.shape[0] - 1,
                                       num_time_points, dtype=int)
            all_values = trajectory[:, :, dim].flatten()
            x_min, x_max = np.percentile(all_values, [0.5, 99.5])
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for ax, idx in zip(axes, time_indices):
                t = t_values[idx]
                ax.hist(trajectory[idx, :, dim], bins=50, density=True,
                        alpha=0.7, color=COLOR_TRUTH, edgecolor='white')
                ax.set_title(f't = {t:.2f}')
                ax.set_xlim(x_min, x_max)
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            sp = os.path.join(output_dir, f'marginal_dim{dim}.{save_format}')
            plt.savefig(sp, dpi=300, bbox_inches='tight', format=save_format)
            plt.close(fig)
            log.info(f"Saved marginal plot to {sp}")

    log.info(f"Flow trajectory plots saved to {output_dir}")


# ============================================================
# Checkpoint evolution (from plot_checkpoint_evolution.py)
# ============================================================

def find_checkpoints(ckpt_dir) -> list:
    """Find all ``epoch_*.ckpt`` files and return sorted ``(epoch, Path)`` pairs.

    If *ckpt_dir* contains no ``.ckpt`` files but has a ``checkpoints/``
    sub-directory, that sub-directory is searched instead.  ``last.ckpt`` is
    appended after all epoch checkpoints with a synthetic epoch number one
    higher than the maximum found.

    Args:
        ckpt_dir: Directory (or parent of ``checkpoints/``) to search.

    Returns:
        List of ``(epoch_number, pathlib.Path)`` tuples sorted by epoch.
    """
    ckpt_dir = Path(ckpt_dir)
    candidates = list(ckpt_dir.glob("epoch_*.ckpt"))
    if not candidates and (ckpt_dir / "checkpoints").is_dir():
        ckpt_dir = ckpt_dir / "checkpoints"
        candidates = list(ckpt_dir.glob("epoch_*.ckpt"))

    pattern = re.compile(r"epoch_(\d+)\.ckpt")
    results = []
    for p in candidates:
        m = pattern.match(p.name)
        if m:
            results.append((int(m.group(1)), p))

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        max_epoch = max((e for e, _ in results), default=-1)
        results.append((max_epoch + 1, last))

    results.sort(key=lambda x: x[0])
    return results


def load_checkpoint_transform(ckpt_dir, device=None):
    """Load the stored transform from ``last.ckpt`` in *ckpt_dir*.

    Args:
        ckpt_dir: Directory containing ``last.ckpt``.
        device:   ``torch.device`` for checkpoint loading (default: CPU).

    Returns:
        Deserialized :class:`~scatterprism.transforms.BaseTransform` or ``None``.
    """
    ckpt_dir = Path(ckpt_dir)
    if device is None:
        device = torch.device('cpu')

    ckpt_path = ckpt_dir / "last.ckpt"
    if ckpt_path.exists():
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if "transform_state" in checkpoint:
            transform = BaseTransform.deserialize(checkpoint["transform_state"])
            if transform is not None:
                log.info(f"Loaded transform from {ckpt_path.name}")
                return transform
    return None


def _generate_checkpoint_samples(model, n_generate: int, device) -> np.ndarray:
    """Generate *n_generate* samples from an already-loaded, eval-mode model."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        samples = model.sample(n_generate, device=device)
    return samples.cpu().numpy()


def plot_checkpoint_evolution_grid(checkpoints: list, n_generate: int, device,
                                    output_path: str, transform=None,
                                    bins: int = 200,
                                    value_range=None):
    """Plot the generated (flattened) distribution at each checkpoint in a grid.

    Each subplot shows the density histogram of generated samples for one
    checkpoint, arranged chronologically.

    Args:
        checkpoints: List of ``(epoch, Path)`` tuples from :func:`find_checkpoints`.
        n_generate: Samples to generate per checkpoint.
        device:      ``torch.device`` for inference.
        output_path: File path for the saved figure.
        transform:   Optional inverse transform applied to generated samples.
        bins:        Number of histogram bins.
        value_range: ``(xmin, xmax)`` tuple; defaults to ``(-5, 5)``.
    """
    n = len(checkpoints)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    if value_range is None:
        value_range = (-5, 5)

    for idx, (epoch, ckpt_path) in enumerate(checkpoints):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        label = "last" if ckpt_path.name == "last.ckpt" else f"epoch {epoch}"
        log.info(f"  [{idx + 1}/{n}] Loading {ckpt_path.name} ...")

        try:
            model, _ = load_model_from_checkpoint(str(ckpt_path), device)
            samples = _generate_checkpoint_samples(model, n_generate, device)
            if transform is not None and hasattr(transform, "inverse_transform"):
                samples = transform.inverse_transform(samples)

            gen_counts, gen_edges = np.histogram(
                samples.flatten(), bins=bins, range=value_range, density=True)
            ax.fill_between(gen_edges[:-1], gen_counts,
                            step="post", alpha=0.5, color=COLOR_GENERATED)
            ax.set_title(label)
            ax.set_xlim(value_range)
            ax.grid(True, alpha=0.25, linestyle=":")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", color=COLOR_MARKER)
            ax.set_title(label)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.supxlabel("Value")
    fig.supylabel("Density")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    log.info(f"Saved checkpoint evolution grid to {output_path}")
    plt.close(fig)


def plot_checkpoint_evolution_overlay(checkpoints: list, n_generate: int, device,
                                       output_path: str, transform=None,
                                       bins: int = 200,
                                       value_range=None):
    """Overlay all checkpoint distributions on a single axis with an epoch colour gradient.

    Uses a ``Reds`` colour-map so earlier epochs are lighter and later epochs
    are darker, with a colour-bar indicating epoch number.

    Args:
        checkpoints: List of ``(epoch, Path)`` tuples.
        n_generate: Samples to generate per checkpoint.
        device:      ``torch.device`` for inference.
        output_path: File path for the saved figure.
        transform:   Optional inverse transform applied to generated samples.
        bins:        Number of histogram bins.
        value_range: ``(xmin, xmax)`` tuple; defaults to ``(-5, 5)``.
    """
    n = len(checkpoints)
    if value_range is None:
        value_range = (-5, 5)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.Reds
    epoch_nums = [e for e, _ in checkpoints]
    norm = plt.Normalize(vmin=min(epoch_nums), vmax=max(epoch_nums))

    for idx, (epoch, ckpt_path) in enumerate(checkpoints):
        label = "last" if ckpt_path.name == "last.ckpt" else f"epoch {epoch}"
        log.info(f"  [{idx + 1}/{n}] Loading {ckpt_path.name} ...")

        try:
            model, _ = load_model_from_checkpoint(str(ckpt_path), device)
            samples = _generate_checkpoint_samples(model, n_generate, device)
            if transform is not None and hasattr(transform, "inverse_transform"):
                samples = transform.inverse_transform(samples)

            gen_counts, gen_edges = np.histogram(
                samples.flatten(), bins=bins, range=value_range, density=True)
            color = cmap(norm(epoch))
            alpha = 0.3 + 0.5 * (idx / max(n - 1, 1))
            ax.step(gen_edges[:-1], gen_counts, where="post",
                    color=color, linewidth=0.8, alpha=alpha, label=label)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"  Error loading {ckpt_path.name}: {e}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Epoch", pad=0.02)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Generated Distribution Evolution Over Training")
    ax.set_xlim(value_range)
    ax.grid(True, alpha=0.25, linestyle=":")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    log.info(f"Saved checkpoint evolution overlay to {output_path}")
    plt.close(fig)


def run_checkpoint_evolution_plot(run_dir: str, device=None,
                                   n_generate: int = 50000,
                                   bins: int = 200,
                                   value_range=None,
                                   skip_last: bool = False,
                                   every_n: int = 1,
                                   overlay: bool = True):
    """Orchestrate checkpoint evolution plots for a single run directory.

    Finds all epoch_*.ckpt files, loads the saved transform, and saves a grid
    figure (and optionally an overlay figure) into *run_dir*.

    Args:
        run_dir:      Path to the run directory (parent of checkpoints/)
        device:       torch.device or None (auto-detect)
        n_generate:  Samples to generate per checkpoint
        bins:         Histogram bins
        value_range:  (xmin, xmax) or None (uses default +/-5)
        skip_last:    Exclude last.ckpt from the plot
        every_n:      Keep only every n-th checkpoint (first + last always kept)
        overlay:      Also produce the overlay plot
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_path = Path(run_dir)
    checkpoints = find_checkpoints(run_path)
    if not checkpoints:
        log.warning(f"No checkpoints found in {run_path}")
        return

    if skip_last:
        checkpoints = [(e, p) for e, p in checkpoints if p.name != "last.ckpt"]

    if every_n > 1:
        filtered = [checkpoints[0]]
        for i in range(1, len(checkpoints) - 1):
            if i % every_n == 0:
                filtered.append(checkpoints[i])
        if len(checkpoints) > 1:
            filtered.append(checkpoints[-1])
        checkpoints = filtered

    log.info(f"Plotting evolution for {len(checkpoints)} checkpoints in {run_path}")

    # Locate actual checkpoints directory for transform loading
    actual_ckpt_dir = run_path
    if not list(run_path.glob("epoch_*.ckpt")) and (run_path / "checkpoints").is_dir():
        actual_ckpt_dir = run_path / "checkpoints"

    transform = load_checkpoint_transform(actual_ckpt_dir, device)

    plot_checkpoint_evolution_grid(
        checkpoints=checkpoints,
        n_generate=n_generate,
        device=device,
        output_path=str(run_path / "checkpoint_evolution_grid.png"),
        transform=transform,
        bins=bins,
        value_range=value_range,
    )

    if overlay:
        plot_checkpoint_evolution_overlay(
            checkpoints=checkpoints,
            n_generate=n_generate,
            device=device,
            output_path=str(run_path / "checkpoint_evolution_overlay.png"),
            transform=transform,
            bins=bins,
            value_range=value_range,
        )


# ============================================================
# Gaussian fitting (from fit_gaussian.py)
# ============================================================

def fit_gaussian_per_dim(samples: np.ndarray) -> list:
    """Fit a Gaussian (mean, std) to each column via MLE.

    Args:
        samples: Array of shape ``[N, D]``.

    Returns:
        List of ``(mean, std)`` tuples, one per dimension.
    """
    return [_norm.fit(samples[:, d]) for d in range(samples.shape[1])]


def fit_and_compare_gaussian(samples: np.ndarray, truth,
                              output_dir: str,
                              col_names: list | None = None) -> list:
    """Fit Gaussians to each dimension of *samples* and compare with *truth*.

    Produces:
    - ``gaussian_fit.png`` -- per-dimension histograms with Gaussian overlays
      and truth distribution (if provided)
    - ``gaussian_fit_summary.txt`` -- table of fitted parameters and delta-sigma

    This is an exploratory/diagnostic tool.  Comparing sigma_gen with sigma_truth
    gives an estimate of the model's per-dimension error range.

    Args:
        samples:    Generated samples  [N, D]
        truth:      Ground-truth array [M, D] or None
        output_dir: Directory to save outputs
        col_names:  Optional column name list (length D)

    Returns:
        List of ``(mean, std)`` tuples for the generated samples.
    """
    fit_results = fit_gaussian_per_dim(samples)
    n_dims = samples.shape[1]
    n_cols = min(4, n_dims)
    n_rows = max(1, (n_dims + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_dims == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for dim in range(n_dims):
        row, col = divmod(dim, n_cols)
        ax = axes[row, col]
        data_col = samples[:, dim]
        mu, sigma = fit_results[dim]

        ax.hist(data_col, bins=100, density=True, alpha=0.7,
                label='Generated', color=COLOR_GENERATED)
        margin = max(3 * sigma, 0.1)
        x = np.linspace(mu - 4 * margin, mu + 4 * margin, 300)
        ax.plot(x, _norm.pdf(x, mu, sigma), color=COLOR_MARKER, linestyle='-', lw=2,
                label=f'Fit: mu={mu:.4f}\n     sigma={sigma:.4f}')

        if truth is not None and dim < truth.shape[1]:
            ax.hist(truth[:, dim], bins=100, density=True,
                    alpha=0.3, label='Ground truth', color=COLOR_TRUTH)
            mu_t, sigma_t = _norm.fit(truth[:, dim])
            x_t = np.linspace(mu_t - 4 * max(3 * sigma_t, 0.1),
                               mu_t + 4 * max(3 * sigma_t, 0.1), 300)
            ax.plot(x_t, _norm.pdf(x_t, mu_t, sigma_t), color=COLOR_TRUTH, linestyle='--', lw=1.5,
                    label=f'Ground truth fit: mu={mu_t:.4f}\n           sigma={sigma_t:.4f}')

        name = col_names[dim] if col_names and dim < len(col_names) else f'Dim {dim}'
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=6)

    for dim in range(n_dims, n_rows * n_cols):
        row, col = divmod(dim, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'gaussian_fit.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Gaussian fit plot saved to {plot_path}")

    # Summary table
    lines = []
    header = f"{'Dim':>5} {'Column':>15} {'mu_gen':>12} {'sigma_gen':>12}"
    if truth is not None:
        header += f"  {'mu_truth':>12} {'sigma_truth':>12} {'delta_sigma':>12}"
    lines.append(header)
    lines.append('-' * len(header))

    for dim, (mu, sigma) in enumerate(fit_results):
        name = col_names[dim] if col_names and dim < len(col_names) else str(dim)
        row_str = f"{dim:>5} {name:>15} {mu:>12.6f} {sigma:>12.6f}"
        if truth is not None and dim < truth.shape[1]:
            mu_t, sigma_t = _norm.fit(truth[:, dim])
            row_str += f"  {mu_t:>12.6f} {sigma_t:>12.6f} {sigma - sigma_t:>12.6f}"
        lines.append(row_str)

    summary_text = "\n".join(lines)
    log.info("\nGaussian fit summary:\n" + summary_text)

    summary_path = os.path.join(output_dir, 'gaussian_fit_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text + "\n")
    log.info(f"Gaussian fit summary saved to {summary_path}")

    return fit_results

def _reproduce_split_indices(run_dir, total_size: int, which: str) -> np.ndarray:
    """Re-run the seeded ``random_split`` and return indices for one partition.

    ``which`` is one of ``"train"``, ``"val"``, ``"test"``.  Reads
    ``split_ratios`` and ``random_seed`` from ``<run_dir>/.hydra/config.yaml``.
    Falls back to ``arange(total_size)`` when the requested partition is empty
    or the config cannot be located.
    """
    if which not in ("train", "val", "test"):
        raise ValueError(f"which must be 'train'|'val'|'test', got {which!r}")

    run_dir = Path(run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        log.warning(f"No .hydra/config.yaml in {run_dir}; falling back to full set")
        return np.arange(total_size, dtype=np.int64)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    ds = (cfg.get("dataset") or {})
    split_ratios = tuple(ds.get("split_ratios", (1.0, 0.0, 0.0)))
    raw_seed = ds.get("random_seed")

    train_size = int(split_ratios[0] * total_size)
    val_size   = int(split_ratios[1] * total_size)
    test_size  = total_size - train_size - val_size
    sizes = {"train": train_size, "val": val_size, "test": test_size}
    if sizes[which] <= 0:
        log.info(f"No {which} partition (split_ratios={split_ratios}); using full set")
        return np.arange(total_size, dtype=np.int64)

    if raw_seed is None:
        log.warning(
            f"{cfg_path} has no random_seed — split was non-deterministic "
            f"and cannot be reproduced; returning full set"
        )
        return np.arange(total_size, dtype=np.int64)
    seed = int(raw_seed)

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        list(range(total_size)), [train_size, val_size, test_size], generator=gen,
    )
    chosen = {"train": train_set, "val": val_set, "test": test_set}[which]
    idx = np.asarray(list(chosen.indices), dtype=np.int64)
    log.info(f"Reproduced {which} split: {len(idx):,} events (seed={seed}, split={split_ratios})")
    return idx


def reproduce_train_indices(run_dir, total_size: int) -> np.ndarray:
    """Reproduce the training-split indices from a run's seeded ``random_split``."""
    return _reproduce_split_indices(run_dir, total_size, "train")


def reproduce_test_indices(run_dir, total_size: int) -> np.ndarray:
    """Reproduce the held-out test indices from a run's seeded ``random_split``."""
    return _reproduce_split_indices(run_dir, total_size, "test")


def plot_distributions_multiple_1d(datasets: dict, output_filepath: str) -> None:
    """Overlay 1-D density histograms of several datasets on a single axis.

    Args:
        datasets:         ``{label: array}`` dict; each array is flattened to 1-D.
        output_filepath:  Path for the saved figure.
    """
    plt.figure(figsize=(10, 6))
    
    combined = np.concatenate([np.array(d).flatten() for d in datasets.values()])
    vmin, vmax = np.percentile(combined, [0.5, 99.5])
    bins = np.linspace(vmin, vmax, 201)
    
    for i, (label, data) in enumerate(datasets.items()):
        data_flat = np.array(data).flatten()
        counts, _ = np.histogram(data_flat, bins=bins, density=True)
        if i == 0:
            plt.fill_between(bins[:-1], counts, step="post", alpha=0.5, label=label)
        else:
            plt.step(bins[:-1], counts, where="post", linewidth=1.5, label=label)

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("1D Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    plt.savefig(output_filepath, dpi=300)
    plt.close()
    log.info(f"Saved 1D multi-distribution plot to {output_filepath}")

