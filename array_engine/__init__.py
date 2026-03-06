"""
array_engine — phased-array beamforming core library.

Coordinate frames
-----------------
* **Array / body frame** (implemented):
  x = along-panel horizontal, y = along-panel vertical, z = boresight (outward normal).
* **Local ENU / ECEF** (planned): interfaces are stubbed in models but not yet wired.

Direction convention
--------------------
Internally every direction is a 3-element unit vector  u = [ux, uy, uz].
az/el are converted to u via :func:`steering.azel_to_unit_vector`.
"""

from array_engine.geometry import rectangular_lattice, triangular_lattice
from array_engine.models import (
    NullWeightsRequest,
    NullWeightsResponse,
    PatternCut,
    PatternRequest,
    PatternResponse,
    WeightsRequest,
    WeightsResponse,
)
from array_engine.nulling import compute_lcmv_weights
from array_engine.pattern import array_factor, pattern_cut
from array_engine.steering import (
    azel_to_unit_vector,
    quantize_phase,
    steering_phases,
    steering_weights,
)

__all__ = [
    "NullWeightsRequest",
    "NullWeightsResponse",
    "PatternCut",
    "PatternRequest",
    "PatternResponse",
    "WeightsRequest",
    "WeightsResponse",
    "array_factor",
    "azel_to_unit_vector",
    "compute_lcmv_weights",
    "pattern_cut",
    "quantize_phase",
    "rectangular_lattice",
    "steering_phases",
    "steering_weights",
    "triangular_lattice",
]
