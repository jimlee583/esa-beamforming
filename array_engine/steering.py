r"""
Steering-vector computation for planar phased arrays.

Conventions (READ THIS)
=======================

Coordinate frame (array / body)
-------------------------------
* x → panel horizontal
* y → panel vertical
* z → boresight (outward normal)

Direction representation
------------------------
Every direction is a **unit vector** ``u = [ux, uy, uz]``.

Azimuth / Elevation mapping
----------------------------
    az : angle in the x-z plane measured **from +z toward +x**.
         az = 0  → boresight (+z)
         az > 0  → beam steered toward +x

    el : angle measured **from the x-z plane toward +y**.
         el = 0  → in the x-z plane
         el > 0  → beam steered toward +y

Conversion:
    ux = sin(az) * cos(el)
    uy = sin(el)
    uz = cos(az) * cos(el)

At broadside (az=0, el=0):  u = [0, 0, 1]

Phase convention
----------------
    φ_n = −k · (r_n · u)        k = 2π / λ
    w_n = a_n · exp(j φ_n)

* ``r_n`` is the position vector of element n.
* ``u``  is the desired look-direction unit vector.
* ``a_n`` is the amplitude taper (1 for uniform).

**Broadside sanity check:**  u = [0,0,1] and all r_n have z=0 (planar array),
so r_n · u = 0 for every element ⇒ all phases = 0.  ✓
"""

from __future__ import annotations

import numpy as np


def azel_to_unit_vector(az_deg: float, el_deg: float) -> np.ndarray:
    """Convert azimuth/elevation (degrees) to a unit direction vector.

    See module docstring for sign conventions.
    """
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    ux = np.sin(az) * np.cos(el)
    uy = np.sin(el)
    uz = np.cos(az) * np.cos(el)
    return np.array([ux, uy, uz])


def steering_phases(
    positions: np.ndarray,
    freq_hz: float,
    az_deg: float = 0.0,
    el_deg: float = 0.0,
) -> np.ndarray:
    """Compute per-element steering phases (radians).

    Parameters
    ----------
    positions : (N, 3)
    freq_hz   : carrier frequency in Hz

    Returns
    -------
    phases : (N,) array in radians
    """
    c = 299_792_458.0
    lam = c / freq_hz
    k = 2.0 * np.pi / lam
    u = azel_to_unit_vector(az_deg, el_deg)
    # φ_n = -k * (r_n · u)
    phases: np.ndarray = -k * (positions @ u)
    return phases


def _amplitude_taper(n: int, taper: str) -> np.ndarray:
    """Return amplitude weights for *n* elements.

    Currently only ``"uniform"`` is implemented.  Taylor and Chebyshev are
    stubbed and will raise if selected.
    """
    if taper == "uniform":
        return np.ones(n)
    if taper in ("taylor", "chebyshev"):
        raise NotImplementedError(
            f"Taper '{taper}' is not yet implemented — contributions welcome!"
        )
    raise ValueError(f"Unknown taper type: {taper}")


def quantize_phase(
    phases_rad: np.ndarray,
    bits: int,
    mode: str = "round",
    phase_range: str = "0_2pi",
) -> np.ndarray:
    """Quantize continuous phases to an N-bit phase-shifter grid.

    Parameters
    ----------
    phases_rad : (N,) continuous phases in radians
    bits       : number of phase-shifter bits (e.g. 3 → 8 discrete levels)
    mode       : ``"round"`` (nearest), ``"floor"``, or ``"ceil"``
    phase_range: ``"0_2pi"`` outputs in [0, 2π), ``"neg_pi_pi"`` in [-π, π)

    Returns
    -------
    quantized : (N,) quantized phases in radians
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")

    n_levels = 2**bits
    step = 2.0 * np.pi / n_levels

    wrapped = np.asarray(phases_rad, dtype=np.float64) % (2.0 * np.pi)

    if mode == "round":
        indices = np.round(wrapped / step).astype(np.int64) % n_levels
    elif mode == "floor":
        indices = np.floor(wrapped / step).astype(np.int64) % n_levels
    elif mode == "ceil":
        indices = np.ceil(wrapped / step).astype(np.int64) % n_levels
    else:
        raise ValueError(f"Unknown quantization mode: {mode!r}")

    quantized: np.ndarray = indices * step

    if phase_range == "neg_pi_pi":
        quantized = (quantized + np.pi) % (2.0 * np.pi) - np.pi
    elif phase_range != "0_2pi":
        raise ValueError(f"Unknown phase_range: {phase_range!r}")

    return quantized


def steering_weights(
    positions: np.ndarray,
    freq_hz: float,
    az_deg: float = 0.0,
    el_deg: float = 0.0,
    taper: str = "uniform",
    phase_bits: int | None = None,
) -> np.ndarray:
    """Compute complex beamforming weights  w_n = a_n * exp(j φ_n).

    Parameters
    ----------
    phase_bits : if provided, quantize steering phases to an N-bit grid
                 before computing the exponential.

    Returns
    -------
    weights : (N,) complex array
    """
    phases = steering_phases(positions, freq_hz, az_deg, el_deg)
    if phase_bits is not None:
        phases = quantize_phase(phases, phase_bits)
    amplitudes = _amplitude_taper(len(positions), taper)
    return amplitudes * np.exp(1j * phases)
