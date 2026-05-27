"""Kinematic observables for γ p → p π⁺ π⁻ events.

Each function takes a batched :class:`~scatterprism.schemas.EventNumpy` and
returns a per-event numpy array.  These are the standard physics-level
observables used throughout the package (training labels, detector output,
plotting axes).
"""

import numpy as np
import vector

from scatterprism.schemas import EventNumpy


def mpipi(event: EventNumpy) -> np.ndarray:
    """Di-pion invariant mass ``M(π⁺π⁻)``.

    The distribution peaks at the mass of any intermediate resonance that
    decays into the (π⁺, π⁻) pair.

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[N]`` (GeV).
    """
    return (event.k1 + event.k2).mass


def t(event: EventNumpy) -> np.ndarray:
    """Mandelstam variable ``t = (p1 − p2)²``.

    Related to the scattering angle and the four-momentum exchanged during
    the interaction.

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[N]`` (GeV²).
    """
    return (event.p1 - event.p2).mass2


def s(event: EventNumpy) -> np.ndarray:
    """Mandelstam variable ``s = (q + p1)²``.

    The square of the total centre-of-mass energy available in the
    collision; determines which final-state particles can be produced.

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[N]`` (GeV²).
    """
    return (event.q + event.p1).mass2


def s12(event: EventNumpy) -> np.ndarray:
    """Squared invariant mass of the (k1, k2) sub-system, ``(k1 + k2)²``.

    A Lorentz-invariant quantity representing the sub-energy of the di-pion
    system.

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[N]`` (GeV²).
    """
    return (event.k1 + event.k2).mass2


def cos_theta(event: EventNumpy) -> np.ndarray:
    """Cosine of the polar angle ``θ`` of particle k1.

    Describes the up-down direction of k1 relative to the lab z-axis.

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[N]`` in ``[-1, 1]``.
    """
    z_axis = vector.obj(x=0, y=0, z=1)
    return z_axis.dot(event.k1.to_3D()) / event.k1.mag


def phi(event: EventNumpy) -> np.ndarray:
    """Azimuthal angle ``φ`` of particle k1, mapped to ``[0, 2π)``.

    Describes the left-right (rotational) orientation of k1 around the
    lab z-axis.

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[N]`` in ``[0, 2π)``.
    """
    return np.arctan2(event.k1.y, event.k1.x) % (2 * np.pi)
