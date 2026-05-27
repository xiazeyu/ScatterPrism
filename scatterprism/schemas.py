"""Shared schemas and enums used across the ScatterPrism package."""

from enum import Enum
from dataclasses import dataclass
import logging
import vector

log = logging.getLogger(__name__)


class Mode(Enum):
    """Top-level execution modes selected via Hydra ``mode=...`` override."""

    TRAIN = "train"
    PREDICT = "predict"
    BATCH_PREDICT = "batch_predict"
    PLOT = "plot"
    TEST_FLOW = "test_flow"
    CHECKPOINT_EVOLUTION = "checkpoint_evolution"


@dataclass
class EventNumpy:
    """A batched γ p → p π⁺ π⁻ event represented as numpy-backed 4-vectors.

    Attributes:
        q:  Incident photon four-momentum.
        p1: Target-proton four-momentum.
        p2: Recoil-proton four-momentum.
        k1: Outgoing π⁺ four-momentum.
        k2: Outgoing π⁻ four-momentum.
    """

    q: vector.MomentumNumpy4D
    p1: vector.MomentumNumpy4D
    p2: vector.MomentumNumpy4D
    k1: vector.MomentumNumpy4D
    k2: vector.MomentumNumpy4D
