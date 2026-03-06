"""Null-depth vs phase-bit analysis.

Computes how jammer null depth degrades as phase-shifter resolution is
reduced.  The continuous LCMV weights serve as the baseline; phases are
then quantized to successively coarser grids while preserving the
original per-element amplitudes.

Null depth convention (relative)
--------------------------------
    null_depth_db = 20 * log10(|w^T a(u_j)| / |w^T a(u_0)|)

where u_0 is the desired steer direction and u_j is a jammer direction.
A deeper (more negative) value means better jammer suppression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from array_engine.nulling import compute_lcmv_weights
from array_engine.pattern import array_factor
from array_engine.steering import quantize_phase


@dataclass
class BitSettingResult:
    """Analysis result for a single phase-bit setting."""

    label: str
    bits: int | None
    desired_gain_mag: float
    jammer_response_mag: list[float]
    null_depth_db: list[float]
    worst_null_depth_db: float


@dataclass
class NullDepthVsBitsResult:
    """Full analysis result across all phase-bit settings."""

    bit_settings_results: list[BitSettingResult]
    jammer_labels: list[str]
    continuous_worst_null_db: float
    best_quantized_worst_null_db: float | None
    best_quantized_label: str | None


def compute_null_depth_vs_phase_bits(
    positions: np.ndarray,
    freq_hz: float,
    steer_az_deg: float,
    steer_el_deg: float,
    jammer_azels: list[tuple[float, float]],
    bit_settings: list[int | None],
    diag_load: float = 1e-6,
) -> NullDepthVsBitsResult:
    """Evaluate null depth across multiple phase-quantisation settings.

    Parameters
    ----------
    positions : (N, 3) element positions in metres.
    freq_hz : carrier frequency in Hz.
    steer_az_deg, steer_el_deg : desired look direction (degrees).
    jammer_azels : list of (az_deg, el_deg) tuples to null.
    bit_settings : list of bit counts.  ``None`` means continuous
        (unquantised) and is used as the baseline.
    diag_load : diagonal loading for LCMV solver.

    Returns
    -------
    NullDepthVsBitsResult containing per-bit-setting null depths.
    """
    lcmv = compute_lcmv_weights(
        positions,
        freq_hz,
        steer_az_deg,
        steer_el_deg,
        jammer_azels,
        diag_load,
    )

    base_amplitudes = np.abs(lcmv.weights)
    base_phases = np.angle(lcmv.weights)

    jammer_labels = [f"az={az:.1f}\u00b0, el={el:.1f}\u00b0" for az, el in jammer_azels]

    results: list[BitSettingResult] = []
    continuous_worst: float = 0.0
    best_quant_worst: float | None = None
    best_quant_label: str | None = None

    for bits in bit_settings:
        if bits is None:
            w = lcmv.weights
            label = "continuous"
        else:
            q_phases = quantize_phase(base_phases, bits)
            w = base_amplitudes * np.exp(1j * q_phases)
            label = f"{bits}-bit"

        af_desired = array_factor(
            positions,
            w,
            freq_hz,
            float(steer_az_deg),
            float(steer_el_deg),
        )
        desired_mag = float(np.abs(af_desired).flat[0])

        jammer_mags: list[float] = []
        null_depths: list[float] = []
        for jaz, jel in jammer_azels:
            af_j = array_factor(positions, w, freq_hz, float(jaz), float(jel))
            j_mag = float(np.abs(af_j).flat[0])
            jammer_mags.append(j_mag)

            if desired_mag > 0:
                ratio = j_mag / desired_mag
                ratio = max(ratio, 1e-15)
                nd = 20.0 * np.log10(ratio)
            else:
                nd = 0.0
            null_depths.append(float(nd))

        worst_nd = float(max(null_depths))

        if bits is None:
            continuous_worst = worst_nd
        else:
            if best_quant_worst is None or worst_nd < best_quant_worst:
                best_quant_worst = worst_nd
                best_quant_label = label

        results.append(
            BitSettingResult(
                label=label,
                bits=bits,
                desired_gain_mag=desired_mag,
                jammer_response_mag=jammer_mags,
                null_depth_db=null_depths,
                worst_null_depth_db=worst_nd,
            )
        )

    return NullDepthVsBitsResult(
        bit_settings_results=results,
        jammer_labels=jammer_labels,
        continuous_worst_null_db=continuous_worst,
        best_quantized_worst_null_db=best_quant_worst,
        best_quantized_label=best_quant_label,
    )
