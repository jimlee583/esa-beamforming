"""
Array-factor and pattern-cut computation.

The array factor for a set of weights w_n at element positions r_n,
evaluated in direction u(θ), is:

    AF(u) = Σ_n  w_n · exp(j k r_n · u(θ))

The total per-element phase is φ_steer + φ_scan = (-k r_n·u_steer) + (k r_n·u_scan),
which the formula above captures because w_n already encodes the steer phase.

Gain is ``20 log10(|AF| / |AF|_max)`` so the peak is always 0 dB.
"""

from __future__ import annotations

import numpy as np

from array_engine.steering import azel_to_unit_vector


def array_factor(
    positions: np.ndarray,
    weights: np.ndarray,
    freq_hz: float,
    az_deg: np.ndarray | float,
    el_deg: np.ndarray | float,
) -> np.ndarray:
    """Compute the complex array factor over a grid of directions.

    Parameters
    ----------
    positions : (N, 3) element positions in metres
    weights   : (N,)   complex beamforming weights
    freq_hz   : carrier frequency in Hz
    az_deg, el_deg : broadcastable arrays of azimuth/elevation in degrees.
        Typically one is a 1-D sweep and the other is scalar.

    Returns
    -------
    af : complex ndarray, same shape as the broadcast of az_deg & el_deg
    """
    c = 299_792_458.0
    k = 2.0 * np.pi * freq_hz / c

    az = np.atleast_1d(np.asarray(az_deg, dtype=np.float64))
    el = np.atleast_1d(np.asarray(el_deg, dtype=np.float64))
    # Broadcast az and el to the same shape before computing
    az_b, el_b = np.broadcast_arrays(az, el)
    out_shape = az_b.shape

    # Unit vectors for every scan angle — shape (M, 3)
    u_mat = azel_to_unit_vector(az_b.ravel(), el_b.ravel()).T  # (M, 3)

    # k * positions @ u^T  → (N, M)
    phase_matrix = k * (positions @ u_mat.T)  # (N, M)

    # AF = Σ_n w_n exp(j k r_n·u)
    af = (weights[:, None] * np.exp(1j * phase_matrix)).sum(axis=0)
    result: np.ndarray = af.reshape(out_shape)
    return result


def pattern_cut(
    positions: np.ndarray,
    weights: np.ndarray,
    freq_hz: float,
    *,
    sweep: str = "az",
    fixed_deg: float = 0.0,
    sweep_range_deg: float = 90.0,
    n_points: int = 361,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 1-D pattern cut.

    Parameters
    ----------
    sweep : ``"az"`` to sweep azimuth at fixed elevation, or ``"el"`` vice-versa.
    fixed_deg : the value of the non-swept coordinate.
    sweep_range_deg : half-span of the sweep (symmetric about 0).
    n_points : number of sample points.

    Returns
    -------
    angles_deg : (n_points,) sweep angles
    gain_db    : (n_points,) normalised gain in dB (peak = 0 dB)
    """
    angles = np.linspace(-sweep_range_deg, sweep_range_deg, n_points)

    if sweep == "az":
        af = array_factor(positions, weights, freq_hz, angles, fixed_deg)
    else:
        af = array_factor(positions, weights, freq_hz, fixed_deg, angles)

    mag = np.abs(af)
    peak = mag.max()
    if peak == 0:
        gain_db = np.full_like(mag, -100.0)
    else:
        with np.errstate(divide="ignore"):
            gain_db = 20.0 * np.log10(mag / peak)
        gain_db = np.clip(gain_db, -100.0, 0.0)

    return angles, gain_db
