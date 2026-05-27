#!/usr/bin/env python3
"""
Best-checkpoint evaluation — Table 1 (Section 4.2) and Table 2 (Appendix C).

Computes the full multi-metric protocol — chi^2, W_1, chi^2_2D, D_corr,
R_NN — on the BEST checkpoint of each run using pre-generated samples.
Designed to run on a CPU node (no model loading required).

Inputs (per run):
    <run_dir>/dataset_cache.npz          — Truth (``pre_transform_data``,
                                            ``data``) and column metadata.
    <run_dir>/generated_samples_best.npz — Pre-generated samples from PREDICT
                                            on ``best.ckpt``.
    <run_dir>/checkpoints/best.ckpt      — Source of the dataset transform
                                            (and best-epoch number).
    <run_dir>/.hydra/config.yaml         — Used to reproduce the test split.

Outputs:
    figures/best_checkpoint_metrics_{mcpom,mock}.json — Per-run metric summary
    (suffixed by ``--runs`` so the two row groups do not overwrite each other).
    Console table with each run's chi^2, W_1, chi^2_2D, D_corr, R_NN.

Usage:
    python scripts/9_eval_best_checkpoint.py --runs {mcpom,mock} [--split {test,full}]

Examples:
    # Table 1 — MC-POM generation + denoising (test split)
    python scripts/9_eval_best_checkpoint.py --runs mcpom

    # Table 2 — synthetic-benchmark sweep
    python scripts/9_eval_best_checkpoint.py --runs mock

    # Use the entire cached dataset
    python scripts/9_eval_best_checkpoint.py --runs mcpom --split full

Notes:
    * Defaults to the held-out TEST split reproduced from each run's seeded
      ``torch.random_split``. The R_NN metric, by construction
      (Section 3.3, Appendix B), compares generated samples against the
      training partition only (val/test rows are excluded) so it diagnoses
      memorisation of training data rather than generalisation to held-out
      events.
    * The ``RUNS_MCPOM`` / ``RUNS_MOCK`` mappings below are fixed for the
      manuscript runs; edit them to point at your own ``outputs/`` directories.
      Results are written to ``figures/best_checkpoint_metrics_<runs>.json``
      so the two row groups do not overwrite each other.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import ks_2samp, wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scatterprism.metric import (  # noqa: E402
    chi2_metric,
    compute_joint_distribution_metrics,
    compute_nn_memorization_metric,
)
from scatterprism.transforms import BaseTransform  # noqa: E402
from scatterprism.utils import reproduce_test_indices, reproduce_train_indices  # noqa: E402


# ── Runs to evaluate (edit for your local outputs/ tree) ─────────────────────
# Paths match the semantic layout produced by the JINST reproduction commands
# (see README → How to Reproduce JINST Paper Results). If you trained without
# the `hydra.run.dir=` overrides, point these at the timestamped run dirs
# under `outputs/YYYY-MM-DD/HH-MM-SS_<run_id>/` instead.

# MC-POM generation + denoising rows for Table 1
RUNS_MCPOM = {
    "mc_pom_gen": {
        "path": "outputs/generation/mcpom/mcpom_gen",
        "label": "Generation",
        "sigma": "---",
    },
    "mc_pom_denoise_sigma2.0": {
        "path": "outputs/denoise/mcpom/mcpom_sigma_2.0",
        "label": "Denoise",
        "sigma": "2.0",
    },
    "mc_pom_denoise_sigma1.0": {
        "path": "outputs/denoise/mcpom/mcpom_sigma_1.0",
        "label": "Denoise",
        "sigma": "1.0",
    },
    "mc_pom_denoise_sigma0.5": {
        "path": "outputs/denoise/mcpom/mcpom_sigma_0.5",
        "label": "Denoise",
        "sigma": "0.5",
    },
}

# Synthetic-benchmark rows for Table 2.
# `triple_mixed` uses the scale=5 variant (used for quantitative
# metrics); the scale=1 variant (`triple_mixed_scale_1`) is the Fig 1
# visualisation run.
RUNS_MOCK = {
    "bimodal_asym":        {"path": "outputs/generation/mock/bimodal_asym",        "label": "bimodal_asym",  "sigma": "---"},
    "delta_0":             {"path": "outputs/generation/mock/delta_0",             "label": "delta_0",       "sigma": "---"},
    "exponential_decay":   {"path": "outputs/generation/mock/exponential_decay",   "label": "exp_decay",     "sigma": "---"},
    "gauss_cutoff":        {"path": "outputs/generation/mock/gauss_cutoff",        "label": "gauss_cutoff",  "sigma": "---"},
    "narrow_wide_overlap": {"path": "outputs/generation/mock/narrow_wide_overlap", "label": "narrow_wide",   "sigma": "---"},
    "noise_3spikes":       {"path": "outputs/generation/mock/noise_3spikes",       "label": "noise_3spk",    "sigma": "---"},
    "noise_10spikes":      {"path": "outputs/generation/mock/noise_10spikes",      "label": "noise_10spk",   "sigma": "---"},
    "tall_flat_far":       {"path": "outputs/generation/mock/tall_flat_far",       "label": "tall_flat",     "sigma": "---"},
    "triple_flat_spread":  {"path": "outputs/generation/mock/triple_flat_spread",  "label": "triple_flat",   "sigma": "---"},
    "triple_mixed":        {"path": "outputs/generation/mock/triple_mixed_scale_5","label": "triple_mixed",  "sigma": "---"},
    "uniform_flat":        {"path": "outputs/generation/mock/uniform_flat",        "label": "uniform_flat",  "sigma": "---"},
}

RUN_GROUPS = {
    "mcpom": RUNS_MCPOM,
    "mock":  RUNS_MOCK,
}

# Evaluation settings matched to the training validation hook.
METRIC_SAMPLE_SIZE = 1_000_000
NN_SAMPLE_SIZE     = 80_000
NN_QUERY_BATCH     = 1_000
NN_REF_BATCH       = 50_000


def load_data(run_dir: Path, split: str):
    """Load pre-generated samples + cached truth, sliced to the chosen split."""
    samples_path = run_dir / "generated_samples_best.npz"
    if not samples_path.exists():
        raise FileNotFoundError(f"No generated_samples_best.npz in {run_dir}")
    samples_physical_full = np.load(str(samples_path))["samples"]

    # Best-checkpoint epoch + dataset transform.
    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    best_epoch = "?"
    transform = None
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        best_epoch = ckpt.get("epoch", "?")
        if "transform_state" in ckpt:
            transform = BaseTransform.deserialize(ckpt["transform_state"])
        del ckpt

    cache_path = run_dir / "dataset_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"No dataset_cache.npz in {run_dir}")
    cache = np.load(str(cache_path), allow_pickle=True)

    # Truth in physical units.
    if "pre_transform_data" in cache.files:
        truth_physical_full = cache["pre_transform_data"]
    elif "data" in cache.files and transform is not None:
        truth_physical_full = transform.inverse_transform(cache["data"])
    else:
        truth_physical_full = cache["data"]

    # Training data in transformed (model-output) space for the NN metric.
    # Restrict to the actual training partition so R_NN measures memorisation
    # of training events, not the full (train+val+test) cached manifold.
    training_transformed = cache["data"] if "data" in cache.files else None
    if training_transformed is not None:
        train_idx = reproduce_train_indices(run_dir, len(training_transformed))
        training_transformed = training_transformed[train_idx]

    # Whether this is a paired (denoising / unfolding) run.
    cfg_path = run_dir / ".hydra" / "config.yaml"
    is_paired = False
    if cfg_path.exists():
        with open(cfg_path) as f:
            is_paired = bool(((yaml.safe_load(f) or {}).get("dataset") or {}).get("paired", False))

    # Apply test split. For paired runs, gen has 1-to-1 ordering with truth
    # so it is sliced too; for unpaired runs the samples are i.i.d. from
    # noise and we just trim them to the truth length.
    if split == "test":
        test_idx = reproduce_test_indices(run_dir, len(truth_physical_full))
        truth_physical = truth_physical_full[test_idx]
        if is_paired and len(samples_physical_full) == len(truth_physical_full):
            samples_physical = samples_physical_full[test_idx]
        else:
            samples_physical = samples_physical_full
    else:
        truth_physical   = truth_physical_full
        samples_physical = samples_physical_full

    return (samples_physical, truth_physical, training_transformed,
            transform, is_paired, best_epoch)


def compute_metrics(run_name, run_info, device, split: str):
    """Compute all five metrics for a single run using pre-generated samples."""
    run_dir = PROJECT_ROOT / run_info["path"]
    print(f"\n{'='*60}")
    print(f"Evaluating: {run_name} ({run_dir.name})  [split={split}]")
    print(f"{'='*60}")

    (samples_physical, truth_physical, training_transformed,
     transform, is_paired, best_epoch) = load_data(run_dir, split)
    print(f"  Best checkpoint epoch: {best_epoch}")
    print(f"  Loaded samples: {samples_physical.shape}, truth: {truth_physical.shape}")

    n = min(METRIC_SAMPLE_SIZE, len(truth_physical), len(samples_physical))
    truth   = truth_physical[:n]
    samples = samples_physical[:n]
    print(f"  Using {n} samples for metrics")

    if truth.ndim == 1:
        truth = truth[:, None]
    if samples.ndim == 1:
        samples = samples[:, None]

    # ── Marginal metrics ────────────────────────────────────────────────────
    num_features = truth.shape[1]
    chi2_vals, ks_vals, w1_vals = [], [], []
    for i in range(num_features):
        t, g = truth[:, i], samples[:, i]
        chi2_vals.append(float(chi2_metric(t, g)))
        ks_vals.append(float(ks_2samp(t, g).statistic))
        w1_vals.append(float(wasserstein_distance(t, g)))
    chi2_mean = float(np.mean(chi2_vals))
    ks_mean   = float(np.mean(ks_vals))
    w1_mean   = float(np.mean(w1_vals))
    print(f"  chi2_mean={chi2_mean:.2f}, ks_mean={ks_mean:.6f}, w1_mean={w1_mean:.6f}")

    # ── Joint metrics (only meaningful for multi-feature data) ──────────────
    chi2_2d_mean = corr_dist = None
    if num_features > 1:
        joint = compute_joint_distribution_metrics(truth, samples)
        chi2_2d_mean = joint["chi2_2d_mean"]
        corr_dist    = joint["correlation_distance"]
        print(f"  chi2_2d_mean={chi2_2d_mean:.2f}, corr_dist={corr_dist:.6f}")

    # ── NN memorisation ratio (vs. the training partition only) ─────────────
    # ``training_transformed`` was already sliced to the seeded training
    # indices in ``load_data`` so the metric measures memorisation of training
    # events rather than the full (train+val+test) cached manifold.
    nn_ratio = d_gen_to_train = d_train_to_train = None
    if not is_paired and training_transformed is not None:
        if transform is not None and hasattr(transform, "transform"):
            samples_transformed = transform.transform(samples)
        else:
            samples_transformed = samples
        print(f"  Computing NN memorization metric ({NN_SAMPLE_SIZE} samples)...")
        nn_results = compute_nn_memorization_metric(
            generated=samples_transformed,
            training=training_transformed,
            nn_sample_size=NN_SAMPLE_SIZE,
            query_batch_size=NN_QUERY_BATCH,
            ref_batch_size=NN_REF_BATCH,
            device=device,
            rng_seed=42,
        )
        d_gen_to_train   = nn_results["D_gen_to_train_mean"]
        d_train_to_train = nn_results["D_train_to_train_mean"]
        nn_ratio = (d_gen_to_train / d_train_to_train
                    if d_train_to_train > 0 else float("nan"))
        print(f"  D_gen→train={d_gen_to_train:.6f}, "
              f"D_train→train={d_train_to_train:.6f}, R_NN={nn_ratio:.4f}")

    return {
        "run_name":         run_name,
        "label":            run_info["label"],
        "sigma":            run_info["sigma"],
        "split":            split,
        "best_epoch":       best_epoch,
        "chi2_mean":        chi2_mean,
        "ks_mean":          ks_mean,
        "w1_mean":          w1_mean,
        "chi2_2d_mean":     chi2_2d_mean,
        "corr_dist":        corr_dist,
        "d_gen_to_train":   d_gen_to_train,
        "d_train_to_train": d_train_to_train,
        "nn_ratio":         nn_ratio,
        "n_samples":        n,
    }


def format_sci(val):
    if val is None:
        return "---"
    exp = int(f"{val:.2e}".split("e")[1])
    return f"{val / (10 ** exp):.2f}e{exp}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the full multi-metric protocol on the best checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", choices=sorted(RUN_GROUPS.keys()), required=True,
                        help="Which row group to evaluate: 'mcpom' (Tab. 1) "
                             "or 'mock' (Tab. 2).")
    parser.add_argument("--split", choices=["test", "full"], default="test",
                        help="Split to evaluate against (default: test)")
    args = parser.parse_args()

    runs = RUN_GROUPS[args.runs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("NOTE: NN memorization metric will be slow on CPU. "
              "Consider running on a GPU node.")

    results = []
    for name, info in runs.items():
        results.append(compute_metrics(name, info, device, args.split))

    print(f"\n{'='*80}\nBEST CHECKPOINT EVALUATION RESULTS  "
          f"(runs={args.runs}, split={args.split})\n{'='*80}")
    header = (f"{'Task':<12} {'σ':>4} {'Epoch':>6} {'χ²':>8} {'W1':>12} "
              f"{'χ²_2D':>8} {'D_corr':>12} {'D_g→t':>12} {'D_t→t':>12} {'R_NN':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        nn_str      = f"{r['nn_ratio']:.2f}"     if r["nn_ratio"]     is not None else "---"
        chi2_2d_str = f"{r['chi2_2d_mean']:.1f}" if r["chi2_2d_mean"] is not None else "---"
        corr_str = format_sci(r["corr_dist"])        if r["corr_dist"]        is not None else "---"
        dgt_str  = format_sci(r["d_gen_to_train"])   if r["d_gen_to_train"]   is not None else "---"
        dtt_str  = format_sci(r["d_train_to_train"]) if r["d_train_to_train"] is not None else "---"
        print(
            f"{r['label']:<12} {r['sigma']:>4} {str(r['best_epoch']):>6} "
            f"{r['chi2_mean']:>8.1f} {format_sci(r['w1_mean']):>12} "
            f"{chi2_2d_str:>8} {corr_str:>12} {dgt_str:>12} {dtt_str:>12} {nn_str:>8}"
        )

    out_path = PROJECT_ROOT / "outputs" / f"best_checkpoint_metrics_{args.runs}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
