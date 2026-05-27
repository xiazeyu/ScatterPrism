"""Dataset classes for ScatterPrism.

Provides the :class:`BaseDataset` abstraction (detector + transform pipeline
shared by all concrete datasets), an :class:`MCPom` loader for the real
physics parquet file, and a family of :class:`Synthetic` distributions used
for unit-style validation of the generative models.
"""

from abc import ABCMeta
from dataclasses import dataclass, field
import logging
import os

from hepunits.units import GeV
from omegaconf import MISSING
from torch.utils.data import Dataset, Subset, random_split
import numpy as np
import pandas as pd
import torch

from scatterprism.detectors import BaseDetector
from scatterprism.transforms import BaseTransform

log = logging.getLogger(__name__)


@dataclass
class BaseDataset(Dataset, metaclass=ABCMeta):
    """Common dataset base wrapping a detector + transform pipeline.

    Concrete subclasses populate a raw :class:`pandas.DataFrame` and call
    :meth:`_setup_data_from_df` to apply the optional detector and transform
    and produce the numpy arrays consumed by training.

    Attributes:
        paired: When True, ``__getitem__`` returns ``(original, detector)``
            tensor pairs and ``original_data`` / ``detector_data`` are
            populated.  When False, only ``data`` is populated.
        data:               Final (transformed) data, only set in unpaired mode.
        original_data:      Particle-level data (paired mode).
        detector_data:      Detector-level data (paired mode).
        pre_transform_data: Copy of the data before the transform was applied;
            used for physical-units plotting/metrics.
    """

    batch_size: int = MISSING
    shuffle: bool = MISSING
    num_workers: int = MISSING
    split_ratios: tuple[float, float, float] = MISSING
    random_seed: int | None = MISSING
    paired: bool = MISSING
    detector: BaseDetector | None = field(default=MISSING, repr=False)
    transform: BaseTransform | None = field(default=MISSING, repr=False)
    data_dir: str = field(default=MISSING, repr=False)

    data: np.ndarray | None = field(default=None, repr=False)
    original_data: np.ndarray | None = field(default=None, repr=False)
    detector_data: np.ndarray | None = field(default=None, repr=False)
    pre_transform_data: np.ndarray | None = field(default=None, repr=False)

    data_dim: int | None = None

    def __len__(self) -> int:
        if self.paired:
            assert self.original_data is not None
            return len(self.original_data)
        else:
            assert self.data is not None
            return len(self.data)

    def __getitem__(self, index):
        if self.paired:
            assert self.original_data is not None and self.detector_data is not None
            return torch.from_numpy(self.original_data[index]).float(), torch.from_numpy(self.detector_data[index]).float()
        else:
            assert self.data is not None
            return torch.from_numpy(self.data[index]).float()

    def get_splits(self) -> tuple[Subset, Subset, Subset]:
        split_ratios = self.split_ratios

        total_size = len(self)
        train_size = int(split_ratios[0] * total_size)
        val_size = int(split_ratios[1] * total_size)
        test_size = total_size - train_size - val_size

        # ``None`` ⇒ non-deterministic: ``Generator.seed()`` self-seeds from
        # std::random_device (``manual_seed`` rejects ``None``).
        rng = torch.Generator()
        if self.random_seed is None:
            rng.seed()
        else:
            rng.manual_seed(self.random_seed)

        if val_size == 0 and test_size == 0:
            # No holdout: train on the full dataset.
            # Build a small random validation subset (capped at 50 000 events) so
            # that distribution metrics (chi2, KS, Wasserstein) remain fast and the
            # val-data buffer does not bloat RAM.
            val_cap = min(total_size, 50_000)
            perm = torch.randperm(total_size, generator=rng)
            train_set = Subset(self, list(range(total_size)))
            val_set = Subset(self, perm[:val_cap].tolist())
            test_set = Subset(self, [])
            log.info(
                f"No holdout split — full {total_size} events for training, "
                f"{val_cap} random events for validation metrics."
            )
            return train_set, val_set, test_set

        train_set, val_set, test_set = random_split(
            self, [train_size, val_size, test_size], generator=rng)
        return train_set, val_set, test_set

    def save_data(self, save_dir: str) -> str:
        """Save dataset arrays to disk for reproducibility.

        Returns the path to the saved cache file.
        """
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, 'dataset_cache.npz')
        save_dict = {}
        if self.data is not None:
            save_dict['data'] = self.data
        if self.original_data is not None:
            save_dict['original_data'] = self.original_data
        if self.detector_data is not None:
            save_dict['detector_data'] = self.detector_data
        if self.pre_transform_data is not None:
            save_dict['pre_transform_data'] = self.pre_transform_data
        if hasattr(self, 'columns') and self.columns is not None:
            save_dict['columns'] = np.array(self.columns, dtype=object)
        if self.data_dim is not None:
            save_dict['data_dim'] = np.array(self.data_dim)
        log.info(f"Saving dataset cache to {cache_path} (compressing {len(save_dict)} arrays)...")
        np.savez_compressed(cache_path, **save_dict)
        log.info(f"Saved dataset cache to {cache_path}")
        return cache_path

    def load_cached_data(self, cache_path: str) -> None:
        """Load dataset arrays from a previously saved cache file."""
        loaded = np.load(cache_path, allow_pickle=True)
        if 'data' in loaded:
            self.data = loaded['data']
        if 'original_data' in loaded:
            self.original_data = loaded['original_data']
        if 'detector_data' in loaded:
            self.detector_data = loaded['detector_data']
        if 'pre_transform_data' in loaded:
            self.pre_transform_data = loaded['pre_transform_data']
        if 'data_dim' in loaded:
            self.data_dim = int(loaded['data_dim'])
        log.info(f"Loaded dataset cache from {cache_path}")

    # Maximum rows to pass to detector.apply() at once.  Keeps peak memory
    # manageable for heavy detectors (e.g. MomentumSmearing with vector.array
    # intermediates) on large datasets (8M+ rows).
    _DETECTOR_BATCH_SIZE: int = 500_000

    def _setup_data_from_df(self, df: pd.DataFrame) -> None:
        """Apply detector, transform, and split into arrays from a raw DataFrame.

        Populates ``self.data`` / ``self.original_data`` / ``self.detector_data`` /
        ``self.pre_transform_data`` and sets ``self.data_dim``.  The logic is
        identical for all concrete dataset classes, so it lives here in the base.

        The detector is applied in batches of ``_DETECTOR_BATCH_SIZE`` rows.
        Numpy arrays are collected per-batch and concatenated at the end so that
        no full-size intermediate DataFrame (from ``pd.concat``) is ever created.
        This mirrors how the DataLoader feeds mini-batches during training and
        keeps peak memory well below the ~5 GB spike that occurs when
        ``MomentumSmearing`` processes 8M rows at once.

        Paired mode (``self.paired=True``):
            Applies detector to a copy of *df*, stores original + detector
            arrays, then fits and applies ``self.transform`` (if any) to both.

        Unpaired mode (``self.paired=False``):
            Optionally applies detector to *df*, then fits and applies
            ``self.transform`` (if any).
        """
        batch_size = self._DETECTOR_BATCH_SIZE
        n = len(df)

        if self.paired:
            assert self.detector is not None, "Paired mode requires a detector"

            # Process detector in batches and collect numpy arrays directly
            # to avoid keeping full-size intermediate DataFrames in memory.
            original_arrs: list[np.ndarray] = []
            detector_arrs: list[np.ndarray] = []

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                chunk_df = df.iloc[start:end]
                det_df = self.detector.apply(chunk_df.copy())

                # Keep only events that survive detector cuts
                accepted = det_df.index
                original_arrs.append(chunk_df.loc[accepted].values)
                detector_arrs.append(det_df.values)

                if n > batch_size:
                    log.info(
                        f"Detector batch {start:,}–{end:,} of {n:,} "
                        f"({len(det_df):,} events after cuts)"
                    )
                del det_df

            self.original_data = (
                np.concatenate(original_arrs) if len(original_arrs) > 1
                else original_arrs[0]
            )
            del original_arrs
            self.detector_data = (
                np.concatenate(detector_arrs) if len(detector_arrs) > 1
                else detector_arrs[0]
            )
            del detector_arrs

            n_cut = n - len(self.original_data)
            if n_cut > 0:
                log.info(f"Detector cuts removed {n_cut} events ({100*n_cut/n:.2f}%)")

            log.info(f"Applied detector effects using {self.detector.__class__.__name__}")
            log.info(f"Data shape after detector effects: {self.detector_data.shape}")

            self.pre_transform_data = self.original_data.copy()

            if self.transform is not None:
                self.transform.fit(self.original_data)
                self.original_data = self.transform.transform(self.original_data)
                self.detector_data = self.transform.transform(self.detector_data)
                log.info(f"Applied transformations using {self.transform.__class__.__name__}")
                log.info(f"Data shape after transformations: {self.original_data.shape}")

            self.data_dim = self.original_data.shape[1]
        else:
            if self.detector is not None:
                data_arrs: list[np.ndarray] = []
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    det_df = self.detector.apply(df.iloc[start:end].copy())
                    data_arrs.append(det_df.values)
                    if n > batch_size:
                        log.info(
                            f"Detector batch {start:,}–{end:,} of {n:,} "
                            f"({len(det_df):,} events after cuts)"
                        )
                    del det_df
                self.data = (
                    np.concatenate(data_arrs) if len(data_arrs) > 1
                    else data_arrs[0]
                )
                del data_arrs
                log.info(f"Applied detector effects using {self.detector.__class__.__name__}")
                log.info(f"Data shape after detector effects: {self.data.shape}")
            else:
                self.data = df.values

            self.pre_transform_data = self.data.copy()

            if self.transform is not None:
                self.transform.fit(self.data)
                self.data = self.transform.transform(self.data)
                log.info(f"Applied transformations using {self.transform.__class__.__name__}")
                log.info(f"Data shape after transformations: {self.data.shape}")

            self.data_dim = self.data.shape[1]


@dataclass
class MCPom(BaseDataset):
    """The MC Pomeron γ p → p π⁺ π⁻ dataset (24-column parquet)."""

    file_name: str = MISSING
    sample_num: int | None = MISSING
    unit = GeV

    columns: list[str] = field(default_factory=lambda: [
        't', 'mpipi', 'costh', 'phi', 'q0', 'q1', 'q2', 'q3', 'p10', 'p11',
        'p12', 'p13', 'k10', 'k11', 'k12', 'k13', 'k20', 'k21', 'k22', 'k23',
        'p20', 'p21', 'p22', 'p23'
    ], repr=False)

    column_name: dict[str, str] = field(default_factory=lambda: {
        't': 't',
        'mpipi': 'mpipi',
        'costh': 'costh',
        'phi': 'phi',
        'q0': 'photon_t',
        'q1': 'photon_x',
        'q2': 'photon_y',
        'q3': 'photon_z',
        'p10': 'target_proton_t',
        'p11': 'target_proton_x',
        'p12': 'target_proton_y',
        'p13': 'target_proton_z',
        'p20': 'recoil_proton_t',
        'p21': 'recoil_proton_x',
        'p22': 'recoil_proton_y',
        'p23': 'recoil_proton_z',
        'k10': 'pi_plus_t',
        'k11': 'pi_plus_x',
        'k12': 'pi_plus_y',
        'k13': 'pi_plus_z',
        'k20': 'pi_minus_t',
        'k21': 'pi_minus_x',
        'k22': 'pi_minus_y',
        'k23': 'pi_minus_z',
    }, repr=False)

    def __post_init__(self):
        self.filepath = os.path.join(self.data_dir, self.file_name)
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset file not found: {self.filepath}\n"
                "Please download 'mc_pom_v2.parquet' from "
                "https://doi.org/10.5281/zenodo.19277777 "
                "and place it under data/mc_pom_v2.parquet"
            )

        df = pd.read_parquet(self.filepath).astype(np.float32)
        log.info(f"Loaded data from {self.filepath} with shape {df.shape}")

        if self.sample_num is not None:
            df = df.sample(n=self.sample_num,
                           random_state=self.random_seed).reset_index(drop=True)
            log.info(f"Sampled {self.sample_num} entries from the dataset")

        self._setup_data_from_df(df)


@dataclass
class Synthetic(BaseDataset):
    """Abstract base for synthetically generated datasets.

    Subclasses override :meth:`generate_data` to return a numpy array of
    shape ``(sample_num, dim)``.
    """

    sample_num: int = MISSING
    dim: int = MISSING

    columns: list[str] | None = field(default_factory=lambda: [])
    column_name: dict[str, str] | None = field(default=None, repr=False)

    def __post_init__(self):
        raw_data = self.generate_data()

        # Ensure shape is (N, D)
        if len(raw_data.shape) == 1:
            raw_data = raw_data.reshape(-1, 1)

        # Convert to DataFrame for Detector compatibility
        # If columns are not provided, generate dummy names "0", "1", ...
        if not self.columns:
            self.columns = [str(i) for i in range(raw_data.shape[1])]
            self.column_name = {col: col for col in self.columns}

        assert raw_data.shape[1] == len(
            self.columns), "Number of columns must match data dimension"

        df = pd.DataFrame(raw_data, columns=self.columns)
        log.info(f"Generated synthetic data with shape {df.shape}")

        self._setup_data_from_df(df)

    def generate_data(self) -> np.ndarray:
        """Subclasses must implement this to return a numpy array of shape (sample_num, dim)"""
        raise NotImplementedError


@dataclass
class Gaussian(Synthetic):
    """Plain ``N(mean, std)`` distribution."""

    mean: float = MISSING
    std: float = MISSING

    def generate_data(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        return rng.normal(loc=self.mean, scale=self.std, size=(self.sample_num, self.dim))


@dataclass
class HighCut(Synthetic):
    """Gaussian with a hard upper cutoff at ``threshold`` (per-dim AND).

    ``buffer_multiplier`` over-samples the raw Gaussian before rejection
    sampling so that, after the cut, at least ``sample_num`` events remain.
    """

    mean: float = MISSING
    std: float = MISSING
    threshold: float = MISSING
    buffer_multiplier: float = MISSING

    def generate_data(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        raw = rng.normal(
            loc=self.mean, scale=self.std,
            size=(int(self.sample_num * self.buffer_multiplier), self.dim),
        )
        # Keep only events where every dimension is below threshold.
        mask = (raw < self.threshold).all(axis=1)
        filtered = raw[mask]
        if len(filtered) < self.sample_num:
            raise ValueError(
                f"HighCut threshold too strict: generated {len(filtered)} valid samples, "
                f"needed {self.sample_num}. Increase `buffer_multiplier`."
            )
        return filtered[:self.sample_num]


@dataclass
class MultiPeak(Synthetic):
    """Mixture of Gaussians: each peak is ``[mean, std, weight]`` (weights re-normalised)."""

    peaks: list = MISSING

    def generate_data(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        means = [p[0] for p in self.peaks]
        stds = [p[1] for p in self.peaks]
        weights = np.array([p[2] for p in self.peaks])
        weights = weights / weights.sum()
        samples_per_peak = rng.multinomial(self.sample_num, weights)

        parts = [
            rng.normal(loc=mu, scale=sigma, size=(count, self.dim))
            for count, mu, sigma in zip(samples_per_peak, means, stds) if count > 0
        ]
        data = np.vstack(parts)
        rng.shuffle(data)  # break per-peak ordering
        return data


@dataclass
class HighFrequency(Synthetic):
    """Base Gaussian plus ``num_noise_peaks`` sharp Gaussian noise peaks.

    ``noise_prob`` is the fraction of samples drawn from the noise peaks
    (each itself a narrow Gaussian of width ``noise_std``); the rest come
    from ``N(base_mean, base_std)``. Noise peak centres are drawn uniformly
    from ``noise_range``.
    """

    base_mean: float = MISSING
    base_std: float = MISSING

    noise_prob: float = MISSING
    num_noise_peaks: int = MISSING
    noise_std: float = MISSING
    noise_range: list = MISSING  # [min, max]

    def generate_data(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)

        n_base = int(self.sample_num * (1 - self.noise_prob))
        base_data = rng.normal(
            loc=self.base_mean, scale=self.base_std, size=(n_base, self.dim),
        )

        n_noise = self.sample_num - n_base
        peak_centers = rng.uniform(
            self.noise_range[0], self.noise_range[1], size=self.num_noise_peaks,
        )
        peak_choices = rng.choice(peak_centers, size=n_noise)
        noise_data = rng.normal(
            loc=peak_choices.reshape(-1, 1), scale=self.noise_std,
            size=(n_noise, self.dim),
        )

        data = np.vstack([base_data, noise_data])
        rng.shuffle(data)
        return data


@dataclass
class DeltaFunction(Synthetic):
    """Degenerate point-mass distribution: every sample equals ``center``."""

    center: float = MISSING

    def generate_data(self) -> np.ndarray:
        return np.full((self.sample_num, self.dim), self.center, dtype=np.float32)


@dataclass
class Uniform(Synthetic):
    """Uniform distribution on ``[low, high]`` (flat, no peak structure)."""

    low: float = MISSING
    high: float = MISSING

    def generate_data(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        return rng.uniform(self.low, self.high, size=(self.sample_num, self.dim))


@dataclass
class Exponential(Synthetic):
    """Shifted exponential distribution: ``loc + Exp(scale)`` (skewed, one-sided)."""

    scale: float = MISSING   # 1/lambda — mean of the un-shifted exponential.
    loc: float = MISSING     # additive shift.

    def generate_data(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        return self.loc + rng.exponential(self.scale, size=(self.sample_num, self.dim))
