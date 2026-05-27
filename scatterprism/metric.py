"""Distribution-quality and memorization metrics for ScatterPrism.

Provides:
    * Per-feature distribution metrics (chi², KS, Wasserstein).
    * Joint distribution metrics (correlation/covariance Frobenius distance,
      pairwise 2-D chi²).
    * Nearest-neighbour memorization diagnostics.
    * A tiny wall-clock :class:`Timer` and a parameter counter for models.
"""

from dataclasses import dataclass
import logging
import numpy as np
import time

import torch

log = logging.getLogger(__name__)


def extract_feature(data: np.ndarray, feature_index: int) -> np.ndarray:
    """Return a single feature column from ``data``.

    Args:
        data:           Dataset of shape ``[N, D]``.
        feature_index:  Column index to extract.

    Returns:
        Numpy array of shape ``[N]``.
    """
    return data[:, feature_index]


def chi2_metric(expected: np.ndarray, observed: np.ndarray, n_bins: int = 50) -> float:
    """Binned chi-squared distance between two 1-D distributions.

    Bins are defined by the range of *expected* (truth).  The *observed*
    histogram is re-normalised to the same total count as *expected* before
    computing the statistic, so generated events landing outside the truth
    range do not bias the result.

    Args:
        expected: Truth samples (1-D array).
        observed: Generated samples (1-D array).
        n_bins:   Number of histogram bins (default 50).

    Returns:
        Chi-squared statistic over all non-empty truth bins.
    """

    bin_min = np.min(expected)
    bin_max = np.max(expected)

    bins = np.linspace(bin_min, bin_max, n_bins + 1)

    hist_expected, _ = np.histogram(expected, bins=bins)
    hist_observed, _ = np.histogram(observed, bins=bins)

    E = np.asarray(hist_expected, dtype=float)
    O = np.asarray(hist_observed, dtype=float)

    obs_total, exp_total = O.sum(), E.sum()
    if obs_total > 0 and exp_total > 0:
        O = O * (exp_total / obs_total)

    mask = E > 0
    return float(np.sum((O[mask] - E[mask]) ** 2 / E[mask]))


@dataclass
class Timer:
    """Simple wall-clock timer with CUDA-sync support.

    ``start()``/``stop()`` synchronise CUDA streams (if available) so the
    measurement covers GPU work as well as host code.
    """

    start_time: float | None = None
    end_time: float | None = None

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()

    def elapsed(self) -> float:
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer has not been started and stopped properly.")
        return self.end_time - self.start_time


def num_params(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def momentum_residuals(momentum_orig: np.ndarray, momentum_test: np.ndarray) -> np.ndarray:
    """Return the element-wise residuals ``momentum_orig - momentum_test``."""
    return momentum_orig - momentum_test


@torch.no_grad()
def nearest_neighbor_distances(
    query: torch.Tensor,
    ref: torch.Tensor,
    query_batch_size: int = 1_000,
    ref_batch_size: int = 50_000,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute the L2 distance from each query point to its nearest neighbor in ref.

    Operates in batches to avoid OOM for large datasets (e.g. 80 K * 8 M).

    Args:
        query:            float Tensor of shape [Q, D]  — points to look up.
        ref:              float Tensor of shape [R, D]  — reference (database) points.
        query_batch_size: Number of query points processed per outer iteration.
        ref_batch_size:   Number of reference points processed per inner iteration.
                          Memory per inner step ≈ query_batch_size * ref_batch_size * 4 bytes.
                          Default 50 000 keeps each step ≤ ~200 MB at query_batch_size=1 000.
        device:           Computation device.  Defaults to CUDA if available, else CPU.

    Returns:
        min_dists: float Tensor of shape [Q] — minimum L2 distance for every query point.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    Q = query.shape[0]
    R = ref.shape[0]

    # Accumulate results on CPU to keep GPU memory free for intermediate matrices.
    min_dists = torch.full((Q,), float("inf"))

    for qi_start in range(0, Q, query_batch_size):
        qi_end = min(qi_start + query_batch_size, Q)
        q_batch = query[qi_start:qi_end].float().to(device)      # [qb, D]
        batch_min = torch.full((q_batch.shape[0],), float("inf"), device=device)

        for ri_start in range(0, R, ref_batch_size):
            ri_end = min(ri_start + ref_batch_size, R)
            r_batch = ref[ri_start:ri_end].float().to(device)    # [rb, D]

            # torch.cdist returns [qb, rb]; take row-wise min immediately.
            dists = torch.cdist(q_batch, r_batch, p=2)            # [qb, rb]
            batch_min = torch.minimum(batch_min, dists.min(dim=1).values)

            del r_batch, dists

        min_dists[qi_start:qi_end] = batch_min.cpu()
        del q_batch, batch_min

    return min_dists


def compute_nn_memorization_metric(
    generated: np.ndarray,
    training: np.ndarray,
    nn_sample_size: int = 80_000,
    query_batch_size: int = 1_000,
    ref_batch_size: int = 50_000,
    device: torch.device | str | None = None,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Nearest-Neighbor memorization diagnostic.

    Computes two distance statistics that together indicate whether the generative
    model has simply memorised its training set:

    * **D_gen_to_train**: distance from each *generated* event to its single closest
      training neighbour.  Formally, take ``nn_sample_size`` generated events and
      compute the NN distance into the *whole* training set.

    * **D_train_to_train**: distance from each member of a random
      ``nn_sample_size``-event training sub-sample to its closest neighbour in the
      *remaining* training events (query indices excluded from the reference so that
      each point does not match itself).

    Interpretation:
        - If D_gen ≈ D_train  → the model generalises well.
        - If D_gen ≪ D_train  → the model copies training events (memorisation).

    Args:
        generated:        numpy array [N_gen, D] in *transformed* (model-output) space.
        training:         numpy array [N_train, D] in the same space.
        nn_sample_size:   Number of events to sample for each NN computation.
                          For mc_pom 1 % ≈ 80 000 events at 8 M training size.
        query_batch_size: Passed to ``nearest_neighbor_distances``.
        ref_batch_size:   Passed to ``nearest_neighbor_distances``.
        device:           Computation device (default: CUDA if available).
        rng_seed:         Random seed for reproducible sampling.

    Returns:
        dict with keys:
            D_gen_to_train_mean, D_gen_to_train_min,
            D_train_to_train_mean, D_train_to_train_min,
            nn_sample_size_gen, nn_sample_size_train, n_training_events.
    """
    rng = np.random.default_rng(rng_seed)
    N_train = len(training)
    N_gen = len(generated)

    # --- generated sample -------------------------------------------------------
    n_gen = min(nn_sample_size, N_gen)
    gen_idx = rng.choice(N_gen, size=n_gen, replace=False)
    gen_sample = torch.from_numpy(generated[gen_idx]).float()
    train_full = torch.from_numpy(training).float()

    log.info(
        f"[NN] D_gen_to_train: {n_gen} generated events vs {N_train} training events"
    )
    d_gen = nearest_neighbor_distances(
        gen_sample, train_full,
        query_batch_size=query_batch_size,
        ref_batch_size=ref_batch_size,
        device=device,
    )

    # --- training sample --------------------------------------------------------
    n_train = min(nn_sample_size, N_train)
    train_query_idx = rng.choice(N_train, size=n_train, replace=False)
    train_query = torch.from_numpy(training[train_query_idx]).float()

    # Exclude query indices so each point cannot match itself.
    mask = np.ones(N_train, dtype=bool)
    mask[train_query_idx] = False
    train_ref = torch.from_numpy(training[mask]).float()

    log.info(
        f"[NN] D_train_to_train: {n_train} query events vs {mask.sum()} reference events"
    )
    d_train = nearest_neighbor_distances(
        train_query, train_ref,
        query_batch_size=query_batch_size,
        ref_batch_size=ref_batch_size,
        device=device,
    )

    results = {
        "D_gen_to_train_mean":   float(d_gen.mean()),
        "D_gen_to_train_min":    float(d_gen.min()),
        "D_train_to_train_mean": float(d_train.mean()),
        "D_train_to_train_min":  float(d_train.min()),
        "nn_sample_size_gen":    int(n_gen),
        "nn_sample_size_train":  int(n_train),
        "n_training_events":     int(N_train),
    }

    log.info(
        f"[NN] D_gen_to_train  — mean={results['D_gen_to_train_mean']:.6f}, "
        f"min={results['D_gen_to_train_min']:.6f}"
    )
    log.info(
        f"[NN] D_train_to_train — mean={results['D_train_to_train_mean']:.6f}, "
        f"min={results['D_train_to_train_min']:.6f}"
    )

    return results


# ── Joint Distribution Metrics ──────────────────────────────────────────────

def correlation_matrix_distance(
    truth: np.ndarray,
    generated: np.ndarray,
) -> float:
    """Frobenius norm of the difference between Pearson correlation matrices.

    Measures how well the generative model reproduces pairwise linear
    correlations across all feature channels.  A value of 0 means the
    correlation structure is perfectly reproduced.

    Args:
        truth:     numpy array [N, D] — reference (validation) data.
        generated: numpy array [M, D] — generated data (same feature space).

    Returns:
        Frobenius norm  ||corr(truth) - corr(generated)||_F .
    """
    corr_truth = np.corrcoef(truth, rowvar=False)  # [D, D]
    corr_gen = np.corrcoef(generated, rowvar=False)  # [D, D]
    # Handle NaN (constant features → undefined correlation)
    corr_truth = np.nan_to_num(corr_truth, nan=0.0)
    corr_gen = np.nan_to_num(corr_gen, nan=0.0)
    return float(np.linalg.norm(corr_truth - corr_gen, ord='fro'))


def covariance_frobenius_distance(
    truth: np.ndarray,
    generated: np.ndarray,
) -> float:
    """Frobenius norm of the difference between covariance matrices.

    Unlike correlation_matrix_distance, this captures both the correlation
    structure *and* the per-feature variance simultaneously.

    Args:
        truth:     numpy array [N, D].
        generated: numpy array [M, D].

    Returns:
        ||cov(truth) - cov(generated)||_F .
    """
    cov_truth = np.cov(truth, rowvar=False)
    cov_gen = np.cov(generated, rowvar=False)
    return float(np.linalg.norm(cov_truth - cov_gen, ord='fro'))


def chi2_2d_metric(
    truth_x: np.ndarray,
    truth_y: np.ndarray,
    gen_x: np.ndarray,
    gen_y: np.ndarray,
    n_bins: int = 20,
) -> float:
    """2-D binned chi-squared between two joint distributions.

    Bins are defined by the range of *truth* features.  The generated
    histogram is re-normalised to the same total count as truth before
    computing the statistic.

    Args:
        truth_x, truth_y: 1-D truth arrays for two features.
        gen_x, gen_y:      1-D generated arrays for the same features.
        n_bins:            Number of bins per axis (total bins = n_bins²).

    Returns:
        Chi-squared statistic over all non-empty truth bins.
    """
    x_range = (np.min(truth_x), np.max(truth_x))
    y_range = (np.min(truth_y), np.max(truth_y))

    hist_truth, _, _ = np.histogram2d(
        truth_x, truth_y, bins=n_bins, range=[x_range, y_range]
    )
    hist_gen, _, _ = np.histogram2d(
        gen_x, gen_y, bins=n_bins, range=[x_range, y_range]
    )

    E = hist_truth.astype(float).ravel()
    O = hist_gen.astype(float).ravel()

    # Re-normalise O to same total as E
    if O.sum() > 0 and E.sum() > 0:
        O = O * (E.sum() / O.sum())

    mask = E > 0
    terms = (O[mask] - E[mask]) ** 2 / E[mask]
    return float(np.sum(terms))


def pairwise_chi2_2d(
    truth: np.ndarray,
    generated: np.ndarray,
    n_bins: int = 20,
) -> dict[str, float]:
    """Compute 2-D chi-squared for all unique feature pairs.

    Returns a dict with ``chi2_2d_mean`` and per-pair values keyed as
    ``chi2_2d_{i}_{j}``.

    Args:
        truth:     numpy array [N, D].
        generated: numpy array [M, D].
        n_bins:    Number of bins per axis.

    Returns:
        Dictionary with chi2_2d_mean and per-pair chi2 values.
    """
    D = truth.shape[1]
    values = []
    pair_results = {}
    for i in range(D):
        for j in range(i + 1, D):
            val = chi2_2d_metric(
                truth[:, i], truth[:, j],
                generated[:, i], generated[:, j],
                n_bins=n_bins,
            )
            pair_results[f"chi2_2d_{i}_{j}"] = val
            values.append(val)

    pair_results["chi2_2d_mean"] = float(np.mean(values)) if values else 0.0
    return pair_results


def compute_joint_distribution_metrics(
    truth: np.ndarray,
    generated: np.ndarray,
    n_bins_2d: int = 20,
) -> dict[str, float]:
    """Compute all joint distribution quality metrics.

    Combines correlation matrix distance, covariance distance, and pairwise
    2-D chi-squared into a single results dictionary.

    Args:
        truth:      numpy array [N, D] — reference data.
        generated:  numpy array [M, D] — generated data.
        n_bins_2d:  Number of bins per axis for 2-D chi-squared.

    Returns:
        Dictionary with all joint distribution metrics.
    """
    results: dict[str, float] = {}

    results["correlation_distance"] = correlation_matrix_distance(truth, generated)
    results["covariance_distance"] = covariance_frobenius_distance(truth, generated)

    chi2_2d_results = pairwise_chi2_2d(truth, generated, n_bins=n_bins_2d)
    results["chi2_2d_mean"] = chi2_2d_results["chi2_2d_mean"]

    log.info(
        f"[Joint] correlation_distance={results['correlation_distance']:.6f}, "
        f"covariance_distance={results['covariance_distance']:.6f}, "
        f"chi2_2d_mean={results['chi2_2d_mean']:.4f}"
    )

    return results
