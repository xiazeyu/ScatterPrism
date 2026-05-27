"""Detector effects applied to particle-level events.

Each detector implements :class:`BaseDetector` and operates on a pandas
``DataFrame`` of MC-POM events.  Detectors can be composed via
:class:`Compose`, instantiated from Hydra configs, and chained inside
:class:`~scatterprism.datasets.BaseDataset` to produce the ``detector_data``
half of paired training samples.
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import logging

from hepunits.units import MeV, GeV
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig
from particle.literals import photon, proton, pi_plus, pi_minus
import numpy as np
import pandas as pd
import vector

from scatterprism import kinematic
from scatterprism.schemas import EventNumpy

log = logging.getLogger(__name__)


@dataclass
class BaseDetector(metaclass=ABCMeta):
    """Abstract base class for detector effects on event DataFrames."""

    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the detector to *data* and return the modified DataFrame."""
        raise NotImplementedError


@dataclass
class Compose(BaseDetector):
    """Apply a sequence of :class:`BaseDetector` instances in order."""

    detectors: list[BaseDetector]

    def __post_init__(self):

        input_detectors = self.detectors
        self.detectors = []

        # Handle both dict-like and list-like inputs from OmegaConf
        if hasattr(input_detectors, 'items'):
            # It's a dict-like object
            items = input_detectors.values()
        elif hasattr(input_detectors, '__iter__'):
            items = input_detectors
        else:
            raise ValueError(
                f"Unexpected detectors type: {type(input_detectors)}")

        for i, det in enumerate(items):
            if isinstance(det, BaseDetector):
                self.detectors.append(det)
            elif isinstance(det, (dict, DictConfig)):
                if "_target_" in det:
                    self.detectors.append(instantiate(det))
                else:
                    raise ValueError(
                        f"Detector config at index {i} is missing '_target_'. Got: {det}")
            else:
                raise ValueError(
                    f"Unexpected detector type at index {i}: {type(det)}")

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        for detector in self.detectors:
            data = detector.apply(data)
        return data


@dataclass
class Identity(BaseDetector):
    """Detector that performs no operation."""

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


@dataclass
class CosThetaCut(BaseDetector):
    """Detector that applies a cut on the absolute value of cos(theta)."""

    threshold: float = MISSING

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        accepted = data['costh'].abs() <= self.threshold
        return data[accepted]


@dataclass
class ValueCut(BaseDetector):
    """Detector that applies a cut on a specified column."""

    column: str = MISSING
    min_value: float | None = MISSING
    max_value: float | None = MISSING

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        accepted = pd.Series(True, index=data.index)
        if self.min_value is not None:
            accepted &= data[self.column] >= self.min_value
        if self.max_value is not None:
            accepted &= data[self.column] <= self.max_value
        return data[accepted]


@dataclass
class MomentumSmearing(BaseDetector):
    """Gaussian smearing of the π± momentum components ``k1{1,2,3}``, ``k2{1,2,3}``.

    Each smeared component receives noise ``N(0, |p|² · sigma)`` (proportional
    to the squared momentum, mimicking ATLAS-style resolution growth). ``p2``
    is recomputed from momentum conservation so the event remains physical;
    the derived observables (``t``, ``mpipi``, ``costh``, ``phi``) are then
    refreshed from the new four-vectors.
    """

    sigma: float = MISSING
    random_seed: int | None = None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_seed)
        num_events = len(data)

        smeared_k11 = data['k11'] + rng.normal(
            loc=0.0, scale=(data['k11']**2) * self.sigma, size=num_events
        )
        smeared_k12 = data['k12'] + rng.normal(
            loc=0.0, scale=(data['k12']**2) * self.sigma, size=num_events
        )
        smeared_k13 = data['k13'] + rng.normal(
            loc=0.0, scale=(data['k13']**2) * self.sigma, size=num_events
        )

        smeared_k21 = data['k21'] + rng.normal(
            loc=0.0, scale=(data['k21']**2) * self.sigma, size=num_events
        )
        smeared_k22 = data['k22'] + rng.normal(
            loc=0.0, scale=(data['k22']**2) * self.sigma, size=num_events
        )
        smeared_k23 = data['k23'] + rng.normal(
            loc=0.0, scale=(data['k23']**2) * self.sigma, size=num_events
        )

        q = vector.array({
            "px": data["q1"], "py": data["q2"], "pz": data["q3"],
            "mass": np.full(num_events, (photon.mass * MeV) / GeV),
        })
        p1 = vector.array({
            "px": data["p11"], "py": data["p12"], "pz": data["p13"],
            "mass": np.full(num_events, (proton.mass * MeV) / GeV),
        })

        # k1 and k2 use the new, smeared momentum components
        k1 = vector.array({
            "px": smeared_k11, "py": smeared_k12, "pz": smeared_k13,
            "mass": np.full(num_events, (pi_plus.mass * MeV) / GeV),
        })

        k2 = vector.array({
            "px": smeared_k21, "py": smeared_k22, "pz": smeared_k23,
            "mass": np.full(num_events, (pi_minus.mass * MeV) / GeV),
        })

        p2 = q + p1 - k1 - k2

        event = EventNumpy(
            q=q, p1=p1, p2=p2, k1=k1, k2=k2
        )

        return data.assign(
            k10=k1.t,
            k11=smeared_k11,
            k12=smeared_k12,
            k13=smeared_k13,

            k20=k2.t,
            k21=smeared_k21,
            k22=smeared_k22,
            k23=smeared_k23,

            p20=p2.t,
            p21=p2.px,
            p22=p2.py,
            p23=p2.pz,

            t=kinematic.t(event),
            mpipi=kinematic.mpipi(event),
            costh=kinematic.cos_theta(event),
            phi=kinematic.phi(event),
        )


@dataclass
class GeneralSmearing(BaseDetector):
    """Per-column Gaussian smearing with noise scale ``|value| · sigma``.

    Applied independently to every numeric column. Unlike
    :class:`MomentumSmearing`, this does not preserve event-level
    conservation laws — it is a coarse, broadband corruption useful for
    sanity-checking unfolding pipelines.
    """

    sigma: float = MISSING
    random_seed: int | None = None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_seed)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        smeared_data = data.copy()
        for col in numeric_cols:
            noise = rng.normal(
                loc=0.0, scale=np.abs(data[col]) * self.sigma, size=len(data),
            )
            smeared_data[col] = data[col] + noise
        return smeared_data


@dataclass
class UniformPhi(BaseDetector):
    """Resample so the azimuthal angle ``phi`` becomes uniform in ``[0, 2π)``.

    Bins ``phi`` into ``num_bins`` equal-width bins and down-samples every
    bin to the minimum population, yielding a φ-flat acceptance profile.
    """

    num_bins: int = MISSING
    random_state: int | None = MISSING

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        bin_labels = range(self.num_bins)
        bin_edges = np.linspace(0, 2 * np.pi, self.num_bins + 1).tolist()
        data_binned = data.copy()
        data_binned['bin'] = pd.cut(
            x=data_binned['phi'],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True
        )

        data_binned.dropna(subset=['bin'], inplace=True)
        data_binned['bin'] = data_binned['bin'].astype(int)

        bin_counts = data_binned['bin'].value_counts()

        min_count = bin_counts.min()

        uniform_data = (
            data_binned.groupby('bin', group_keys=False)
            .apply(lambda x: x.sample(min_count, random_state=self.random_state), include_groups=False)
            .sort_index()
        )

        return uniform_data
