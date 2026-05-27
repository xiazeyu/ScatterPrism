"""Feature-space transformations applied between dataset and model.

Each transform implements :class:`BaseTransform` (``fit`` / ``transform`` /
``inverse_transform``) and can be composed via :class:`Compose`.  Transforms
are serialised into checkpoints so that prediction-time inverse mapping is
reproducible without re-fitting on the original training data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging

from hepunits.units import MeV, GeV
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig
from particle.literals import photon, proton, pi_plus, pi_minus
import numpy as np
import vector

from scatterprism import kinematic
from scatterprism.schemas import EventNumpy

log = logging.getLogger(__name__)

# arctanh(±1) diverges; clip φ/π just inside the unit interval so events with
# φ exactly at ±π map to finite values rather than ±inf.
_ARCTANH_PHI_CLIP = 1.0 - 1e-6


@dataclass
class BaseTransform:
    """Abstract base class for transformations."""

    input_columns: list[str] | None = None
    output_columns: list[str] | None = None

    dim_input: int | None = None
    dim_output: int | None = None

    @property
    def data_dim(self) -> int:
        """Get the dimensionality of the data after transformation."""
        if self.dim_output is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        return self.dim_output

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the transformation to the data."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data."""
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform the data."""
        raise NotImplementedError

    def serialize(self) -> dict:
        """Serialize transform to a dictionary for checkpoint storage."""
        # Default implementation for transforms without learned parameters
        return {'type': self.__class__.__name__}

    @staticmethod
    def _deserialize(state: dict) -> "BaseTransform":
        raise NotImplementedError

    @staticmethod
    def deserialize(state: dict) -> "BaseTransform":
        """Deserialize transform from checkpoint state dictionary."""
        if state is None:
            return Identity()

        transform_type = state.get('type')
        if transform_type is None:
            log.warning("Transform state missing 'type' field")
            return Identity()

        # Map type names to classes and their deserialize methods
        type_map: dict[str, type[BaseTransform]] = {
            'StandardScaler': StandardScaler,
            'LogTransformer': LogTransformer,
            'Compose': Compose,
            'Identity': Identity,
            'FourParticleRepresentation': FourParticleRepresentation,
            'ReduceRedundantv1': ReduceRedundantv1,
            'DLPPRepresentation': DLPPRepresentation,
        }

        cls = type_map.get(transform_type)
        if cls is None:
            log.warning(f"Unknown transform type: {transform_type}, skipping")
            return Identity()

        # Each class has its own deserialize implementation
        return cls._deserialize(state)


@dataclass
class Compose(BaseTransform):
    """Apply a sequence of :class:`BaseTransform` instances in order."""

    transforms: list[BaseTransform] = field(
        default_factory=lambda: [Identity()])

    def __post_init__(self):

        input_transformations = self.transforms
        self.transforms = []

        # Handle both dict-like and list-like inputs from OmegaConf
        if hasattr(input_transformations, 'items'):
            # It's a dict-like object
            items = input_transformations.values()
        elif hasattr(input_transformations, '__iter__'):
            items = input_transformations
        else:
            raise ValueError(
                f"Unexpected transformations type: {type(input_transformations)}")

        for i, transform in enumerate(items):
            if isinstance(transform, BaseTransform):
                self.transforms.append(transform)
            elif isinstance(transform, (dict, DictConfig)):
                if "_target_" in transform:
                    self.transforms.append(instantiate(transform))
                else:
                    raise ValueError(
                        f"Transform config at index {i} is missing '_target_'. Got: {transform}")
            else:
                raise ValueError(
                    f"Unexpected transform type at index {i}: {type(transform)}")

    def serialize(self) -> dict:
        """Serialize Compose transform with all nested transforms."""
        return {
            'type': 'Compose',
            'transforms': [t.serialize() for t in self.transforms],
        }

    @staticmethod
    def _deserialize(state: dict) -> "BaseTransform":
        """Deserialize Compose from state dictionary."""
        transforms = [BaseTransform.deserialize(t) for t in state.get('transforms', [])]
        if not transforms:
            return Identity()  # No nested transforms, return Identity
        compose = Compose.__new__(Compose)
        compose.transforms = transforms
        # Compute dim_input/dim_output from first/last transform
        compose.dim_input = transforms[0].dim_input if transforms[0].dim_input else None
        compose.dim_output = transforms[-1].dim_output if transforms[-1].dim_output else None
        return compose

    def fit(self, data: np.ndarray) -> None:
        self.dim_input = data.shape[1]
        last_column_names = None
        for transform in self.transforms:
            # Only check column compatibility if both are defined
            if transform.input_columns is not None and last_column_names is not None:
                if transform.input_columns != last_column_names:
                    raise ValueError(f"Input columns {transform.input_columns} do not match "
                                     f"the output columns of the previous transformation "
                                     f"{last_column_names}.")

            transform.fit(data)
            last_column_names = transform.output_columns
            data = transform.transform(data)

        self.dim_output = data.shape[1]

    def transform(self, data: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            data = transform.transform(data)
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        for transform in reversed(self.transforms):
            data = transform.inverse_transform(data)
        return data


@dataclass
class Identity(BaseTransform):
    """Transformation that performs no operation."""

    def fit(self, data: np.ndarray) -> None:
        self.dim_input = self.dim_output = data.shape[1]

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[1] == self.dim_input
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[1] == self.dim_output
        return data

    @staticmethod
    def _deserialize(state: dict) -> "BaseTransform":
        """Deserialize Identity from state dictionary."""
        return Identity()


@dataclass
class StandardScaler(BaseTransform):
    """Transformation that standardizes features by removing the mean and scaling to unit variance.

    The output is: scale * (data - mean) / std

    Args:
        mean: Pre-specified mean. If None, computed from data during fit.
        std: Pre-specified std. If None, computed from data during fit.
        scale: Multiplier applied after standardization. Default 1.0.
            E.g., scale=10 gives output with std ~10 (data roughly in [-30, 30]).
    """

    mean: np.ndarray | list[float] | None = None
    std: np.ndarray | list[float] | None = None
    scale: float = 1.0

    def __post_init__(self):
        # Convert lists to numpy arrays if provided
        if self.mean is not None and not isinstance(self.mean, np.ndarray):
            self.mean = np.array(self.mean)
        if self.std is not None and not isinstance(self.std, np.ndarray):
            self.std = np.array(self.std)

    def serialize(self) -> dict:
        """Serialize StandardScaler with mean, std, and scale parameters."""
        return {
            'type': 'StandardScaler',
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'scale': self.scale,
            'dim_input': self.dim_input,
            'dim_output': self.dim_output,
        }

    @staticmethod
    def _deserialize(state: dict) -> "BaseTransform":
        """Deserialize StandardScaler from state dictionary."""
        transform = StandardScaler(
            mean=np.atleast_1d(np.array(state['mean'])) if state.get('mean') is not None else None,
            std=np.atleast_1d(np.array(state['std'])) if state.get('std') is not None else None,
            scale=state.get('scale', 1.0),
        )
        # Prefer explicit dim from state (handles scalar mean/std correctly);
        # fall back to mean/std length only when they are per-column arrays
        transform.dim_input = state.get('dim_input')
        transform.dim_output = state.get('dim_output')
        if transform.dim_input is None and transform.mean is not None and len(transform.mean) > 1:
            transform.dim_input = len(transform.mean)
        if transform.dim_output is None and transform.mean is not None and len(transform.mean) > 1:
            transform.dim_output = len(transform.mean)
        return transform

    def fit(self, data: np.ndarray) -> None:
        self.dim_input = data.shape[1]
        if self.mean is None and self.std is None:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        elif self.mean is None or self.std is None:
            raise ValueError(
                "StandardScaler requires both `mean` and `std` to be set, "
                "or both to be None for fitting. Got "
                f"mean is None: {self.mean is None}, std is None: {self.std is None}."
            )
        else:
            log.warning("StandardScaler is already fitted; fit() call ignored.")
        self.dim_output = data.shape[1]

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        if self.dim_input is not None:
            assert data.shape[1] == self.dim_input

        numerator = data - self.mean
        denominator = self.std

        # We use np.divide to handle cases where std is zero, avoiding division by zero errors.
        # Where the standard deviation is 0, the scaled value will be 0.
        scaled = np.divide(numerator, denominator,
                           out=np.zeros_like(numerator, dtype=float),
                           where=denominator != 0)
        return scaled * self.scale

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        if self.dim_output is not None:
            assert data.shape[1] == self.dim_output
        return (data / self.scale) * self.std + self.mean


@dataclass
class LogTransformer(BaseTransform):
    """Element-wise log transformation for selected columns.

    Applies log(data + offset) to specified column indices. Useful for
    heavy-tailed features (e.g., momentum magnitudes) before StandardScaler.

    Args:
        columns: List of column indices to apply log transform to.
            If None, all columns are transformed.
        offset: Additive offset before taking log to avoid log(0). Default 1.0.
    """

    columns: list[int] | None = None
    offset: float = 1.0

    def fit(self, data: np.ndarray) -> None:
        self.dim_input = data.shape[1]
        self.dim_output = data.shape[1]
        if self.columns is None:
            self.columns = list(range(data.shape[1]))
        # Validate that data + offset > 0 for the selected columns; otherwise
        # transform() would silently emit NaN/-inf and downstream training
        # would break in non-obvious ways.
        min_vals = np.min(data[:, self.columns], axis=0)
        if np.any(min_vals + self.offset <= 0):
            raise ValueError(
                f"LogTransformer: some values in selected columns have "
                f"data + offset <= 0 (min values: {min_vals}). "
                f"Increase `offset` so the smallest value becomes strictly positive."
            )

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[1] == self.dim_input
        out = data.copy()
        out[:, self.columns] = np.log(data[:, self.columns] + self.offset)
        return out

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[1] == self.dim_output
        out = data.copy()
        out[:, self.columns] = np.exp(data[:, self.columns]) - self.offset
        return out

    def serialize(self) -> dict:
        """Serialize LogTransformer with columns, offset, and fit dims."""
        return {
            'type': 'LogTransformer',
            'columns': self.columns,
            'offset': self.offset,
            'dim_input': self.dim_input,
            'dim_output': self.dim_output,
        }

    @staticmethod
    def _deserialize(state: dict) -> "BaseTransform":
        """Deserialize LogTransformer from state dictionary."""
        transform = LogTransformer(
            columns=state.get('columns'),
            offset=state.get('offset', 1.0),
        )
        transform.dim_input = state.get('dim_input')
        transform.dim_output = state.get('dim_output')
        # Back-compat: older checkpoints did not store dim_input/dim_output.
        # Fall back to max(columns)+1, but only when nothing better is available.
        if transform.dim_input is None and transform.columns is not None:
            transform.dim_input = max(transform.columns) + 1
        if transform.dim_output is None:
            transform.dim_output = transform.dim_input
        return transform


@dataclass
class FourParticleRepresentation(BaseTransform):
    """Transformation to and from a four-particle representation."""
    input_columns: list[str] = field(default_factory=lambda: ['t', 'mpipi', 'costh', 'phi', 'q0', 'q1', 'q2', 'q3', 'p10', 'p11',
                                                              'p12', 'p13', 'k10', 'k11', 'k12', 'k13', 'k20', 'k21', 'k22', 'k23',
                                                              'p20', 'p21', 'p22', 'p23'])

    output_columns: list[str] = field(default_factory=lambda: [
        'q1', 'q2', 'q3',         # photon
        'p11', 'p12', 'p13',      # target_proton
        'k11', 'k12', 'k13',      # pi_plus
        'k21', 'k22', 'k23',      # pi_minus
    ])

    def fit(self, data: np.ndarray) -> None:
        self.dim_input = len(self.input_columns)
        self.dim_output = len(self.output_columns)

    def transform(self, data: np.ndarray) -> np.ndarray:

        assert data.shape[1] == self.dim_input
        out = data[:, [self.input_columns.index(
            col) for col in self.output_columns]]
        assert out.shape[1] == self.dim_output
        return out

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[1] == self.dim_output

        num_events = data.shape[0]

        # Photon (q)
        q = vector.array({
            "px": data[:, 0], "py": data[:, 1], "pz": data[:, 2],
            "mass": np.full(num_events, (photon.mass * MeV) / GeV),
        })

        # Target Proton (p1)
        p1 = vector.array({
            "px": data[:, 3], "py": data[:, 4], "pz": data[:, 5],
            "mass": np.full(num_events, (proton.mass * MeV) / GeV),
        })

        # Pi Plus (k1)
        k1 = vector.array({
            "px": data[:, 6], "py": data[:, 7], "pz": data[:, 8],
            "mass": np.full(num_events, (pi_plus.mass * MeV) / GeV),
        })

        # Pi Minus (k2)
        k2 = vector.array({
            "px": data[:, 9], "py": data[:, 10], "pz": data[:, 11],
            "mass": np.full(num_events, (pi_minus.mass * MeV) / GeV),
        })

        p2 = q + p1 - k1 - k2

        event = EventNumpy(
            q=q, p1=p1, p2=p2, k1=k1, k2=k2
        )

        out = np.stack([
            kinematic.t(event),
            kinematic.mpipi(event),
            kinematic.cos_theta(event),
            kinematic.phi(event),
            q.t, q.px, q.py, q.pz,
            p1.t, p1.px, p1.py, p1.pz,
            k1.t, k1.px, k1.py, k1.pz,
            k2.t, k2.px, k2.py, k2.pz,
            p2.t, p2.px, p2.py, p2.pz,
        ], axis=1)

        assert out.shape[1] == len(self.input_columns)

        return out

    @staticmethod
    def _deserialize(state: dict) -> "BaseTransform":
        """Deserialize FourParticleRepresentation from state dictionary."""
        transform = FourParticleRepresentation()
        transform.fit(np.zeros((1, 24)))  # Dummy fit to set dimensions
        return transform


@dataclass
class ReduceRedundantv1(BaseTransform):
    """Transformation to reduce redundant features in the four-particle representation."""
    input_columns: list[str] = field(default_factory=lambda: [
        'q1', 'q2', 'q3',         # photon
        'p11', 'p12', 'p13',      # target_proton
        'k11', 'k12', 'k13',      # pi_plus
        'k21', 'k22', 'k23',      # pi_minus
    ])

    output_columns: list[str] = field(default_factory=lambda: [
        'q1', 'q3',         # photon
        'p11', 'p13',      # target_proton
        'k11', 'k12', 'k13',      # pi_plus
        'k21', 'k22', 'k23',      # pi_minus
    ])

    def fit(self, data: np.ndarray) -> None:
        self.dim_input = len(self.input_columns)
        self.dim_output = len(self.output_columns)

    def transform(self, data: np.ndarray) -> np.ndarray:

        assert data.shape[1] == self.dim_input
        out = data[:, [self.input_columns.index(
            col) for col in self.output_columns]]
        assert out.shape[1] == self.dim_output
        return out

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[1] == len(self.output_columns)
        out = np.hstack([
            data[:, 0:1], np.zeros(
                (data.shape[0], 1)), data[:, 1:2],         # photon
            data[:, 2:3], np.zeros(
                (data.shape[0], 1)), data[:, 3:4],      # target_proton
            data[:, 4:7],      # pi_plus
            data[:, 7:10],      # pi_minus
        ])
        assert out.shape[1] == len(self.input_columns)
        return out

    @staticmethod
    def _deserialize(state: dict) -> "BaseTransform":
        """Deserialize ReduceRedundantv1 from state dictionary."""
        transform = ReduceRedundantv1()
        transform.fit(np.zeros((1, 12)))  # Dummy fit to set dimensions
        return transform


@dataclass
class DLPPRepresentation(BaseTransform):
    """
    Transformation to and from a reduced representation used in DLPP.
    This representation uses log(pt), eta, and a scaled phi for each particle.

    inspired by https://github.com/heidelberg-hepml/ml-tutorials/blob/main/tutorial-14-diffusion.ipynb
    """
    input_columns: list[str] = field(default_factory=lambda: ['t', 'mpipi', 'costh', 'phi', 'q0', 'q1', 'q2', 'q3', 'p10', 'p11',
                                                              'p12', 'p13', 'k10', 'k11', 'k12', 'k13', 'k20', 'k21', 'k22', 'k23',
                                                              'p20', 'p21', 'p22', 'p23'])

    output_columns: list[str] = field(default_factory=lambda: [
        "q_log_pt", "q_eta", "q_phi_scaled",
        "p1_log_pt", "p1_eta", "p1_phi_scaled",
        "k1_log_pt", "k1_eta", "k1_phi_scaled",
        "k2_log_pt", "k2_eta", "k2_phi_scaled",
    ])

    def fit(self, data: np.ndarray) -> None:
        self.dim_input = len(self.input_columns)
        self.dim_output = len(self.output_columns)

    def transform(self, data: np.ndarray) -> np.ndarray:

        assert data.shape[1] == self.dim_input

        num_events = data.shape[0]

        # Photon (q)
        q = vector.array({
            "px": data[:, 5], "py": data[:, 6], "pz": data[:, 7],
            "mass": np.full(num_events, (photon.mass * MeV) / GeV),
        })

        # Target Proton (p1)
        p1 = vector.array({
            "px": data[:, 9], "py": data[:, 10], "pz": data[:, 11],
            "mass": np.full(num_events, (proton.mass * MeV) / GeV),
        })

        # Pi Plus (k1)
        k1 = vector.array({
            "px": data[:, 13], "py": data[:, 14], "pz": data[:, 15],
            "mass": np.full(num_events, (pi_plus.mass * MeV) / GeV),
        })

        # Pi Minus (k2)
        k2 = vector.array({
            "px": data[:, 17], "py": data[:, 18], "pz": data[:, 19],
            "mass": np.full(num_events, (pi_minus.mass * MeV) / GeV),
        })

        def _phi_scaled(phi_arr: np.ndarray) -> np.ndarray:
            # Clip phi/pi just inside [-1, 1] so arctanh stays finite for events
            # with phi exactly at ±π (rare, but inevitable with O(1e7) events).
            return np.arctanh(np.clip(phi_arr / np.pi,
                                      -_ARCTANH_PHI_CLIP, _ARCTANH_PHI_CLIP))

        out = np.stack([
            np.log(q.pt),  q.eta,  _phi_scaled(q.phi),
            np.log(p1.pt), p1.eta, _phi_scaled(p1.phi),
            np.log(k1.pt), k1.eta, _phi_scaled(k1.phi),
            np.log(k2.pt), k2.eta, _phi_scaled(k2.phi),
        ], axis=1)

        assert out.shape[1] == self.dim_output
        return out

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[1] == len(self.output_columns)

        num_events = data.shape[0]

        # Reverse the transformations: log(pt) -> pt, arctanh(phi/pi) -> phi
        q = vector.array({
            "pt": np.exp(data[:, 0]),
            "eta": data[:, 1],
            "phi": np.pi * np.tanh(data[:, 2]),
            "mass": np.full(num_events, (photon.mass * MeV) / GeV),
        })

        p1 = vector.array({
            "pt": np.exp(data[:, 3]),
            "eta": data[:, 4],
            "phi": np.pi * np.tanh(data[:, 5]),
            "mass": np.full(num_events, (proton.mass * MeV) / GeV),
        })

        k1 = vector.array({
            "pt": np.exp(data[:, 6]),
            "eta": data[:, 7],
            "phi": np.pi * np.tanh(data[:, 8]),
            "mass": np.full(num_events, (pi_plus.mass * MeV) / GeV),
        })

        k2 = vector.array({
            "pt": np.exp(data[:, 9]),
            "eta": data[:, 10],
            "phi": np.pi * np.tanh(data[:, 11]),
            "mass": np.full(num_events, (pi_minus.mass * MeV) / GeV),
        })

        p2 = q + p1 - k1 - k2

        event = EventNumpy(
            q=q, p1=p1, p2=p2, k1=k1, k2=k2
        )

        out = np.stack([
            kinematic.t(event),
            kinematic.mpipi(event),
            kinematic.cos_theta(event),
            kinematic.phi(event),
            q.t, q.px, q.py, q.pz,
            p1.t, p1.px, p1.py, p1.pz,
            k1.t, k1.px, k1.py, k1.pz,
            k2.t, k2.px, k2.py, k2.pz,
            p2.t, p2.px, p2.py, p2.pz,
        ], axis=1)

        assert out.shape[1] == len(self.input_columns)
        return out

    @staticmethod
    def _deserialize(state: dict) -> "BaseTransform":
        """Deserialize DLPPRepresentation from state dictionary."""
        transform = DLPPRepresentation()
        transform.fit(np.zeros((1, 24)))  # Dummy fit to set dimensions
        return transform
