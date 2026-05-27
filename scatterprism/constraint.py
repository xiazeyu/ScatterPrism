"""Kinematic-constraint checks for γ p → p π⁺ π⁻ events.

Each function returns a per-event (or per-event-per-channel) residual array
that should be close to 0 for events satisfying the underlying physical
constraint.  Useful for validating generated samples or smeared detector data.
"""

from hepunits.units import MeV, GeV
from particle.literals import photon, proton, pi_plus, pi_minus
import numpy as np

from scatterprism.schemas import EventNumpy


def momentum_conservation(event: EventNumpy) -> np.ndarray:
    """Return per-event invariant mass of (q + p1) − (p2 + k1 + k2).

    For a momentum-conserving event this is identically zero.

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[N]`` with the residual mass per event.
    """
    return (event.q + event.p1).subtract(event.p2 + event.k1 + event.k2).mass


def energy_conservation(event: EventNumpy) -> np.ndarray:
    """Return per-event energy residual (E_q + E_p1) − (E_p2 + E_k1 + E_k2).

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[N]`` with the energy residual per event.
    """
    return (event.q.e + event.p1.e) - (event.p2.e + event.k1.e + event.k2.e)


def mass_conservation(event: EventNumpy) -> np.ndarray:
    """Return per-particle mass deviation from the PDG value.

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[5, N]``: rows are (q, p1, p2, k1, k2) and
        each entry is ``measured_mass − pdg_mass`` in GeV.
    """
    return np.array([
        event.q.mass - (photon.mass * MeV) / GeV,
        event.p1.mass - (proton.mass * MeV) / GeV,
        event.p2.mass - (proton.mass * MeV) / GeV,
        event.k1.mass - (pi_plus.mass * MeV) / GeV,
        event.k2.mass - (pi_minus.mass * MeV) / GeV,
    ])


def zero_momentum(event: EventNumpy) -> np.ndarray:
    """Return components that vanish in the standard MC-POM lab frame.

    Specifically: ``q.py``, ``p1.py``, ``p2.px``, ``p2.py``.

    Args:
        event: Batched event four-vectors.

    Returns:
        Numpy array of shape ``[4, N]`` with the four residual channels.
    """
    return np.array([event.q.py, event.p1.py, event.p2.px, event.p2.py])
