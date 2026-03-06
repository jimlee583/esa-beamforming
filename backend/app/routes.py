"""API route handlers."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from array_engine.analysis import compute_null_depth_vs_phase_bits
from array_engine.geometry import rectangular_lattice, triangular_lattice
from array_engine.models import (
    BitSettingResultModel,
    LatticeType,
    NullDepthSummary,
    NullDepthVsBitsRequest,
    NullDepthVsBitsResponse,
    NullWeightsRequest,
    NullWeightsResponse,
    PatternRequest,
    PatternResponse,
    WeightsRequest,
    WeightsResponse,
)
from array_engine.models import (
    PatternCut as PatternCutModel,
)
from array_engine.nulling import compute_lcmv_weights
from array_engine.pattern import pattern_cut
from array_engine.steering import quantize_phase, steering_phases, steering_weights

router = APIRouter()


def _build_array(req: WeightsRequest) -> tuple[np.ndarray, float]:
    """Build element positions from a request."""
    gen = rectangular_lattice if req.lattice == LatticeType.rectangular else triangular_lattice
    return gen(
        freq_hz=req.freq_hz,
        panel_size_m=req.panel_size_m,
        spacing_m=req.spacing_m,
        element_k_lambda=req.element_k_lambda,
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/weights", response_model=WeightsResponse)
def compute_weights(req: WeightsRequest) -> WeightsResponse:
    positions, d = _build_array(req)
    phases = steering_phases(positions, req.freq_hz, req.steer_az_deg, req.steer_el_deg)
    weights = steering_weights(
        positions,
        req.freq_hz,
        req.steer_az_deg,
        req.steer_el_deg,
        req.taper.value,
    )

    c = 299_792_458.0
    lam = c / req.freq_hz

    resp = WeightsResponse(
        n_elements=len(positions),
        positions=positions.tolist(),
        phases_rad=phases.tolist(),
        weights_re_im=[[float(w.real), float(w.imag)] for w in weights],
        spacing_m=d,
        spacing_lambda=d / lam,
    )

    if req.phase_bits is not None:
        q_phases = quantize_phase(phases, req.phase_bits)
        q_weights = steering_weights(
            positions,
            req.freq_hz,
            req.steer_az_deg,
            req.steer_el_deg,
            req.taper.value,
            phase_bits=req.phase_bits,
        )
        resp.phase_bits = req.phase_bits
        resp.quantized_phases_rad = q_phases.tolist()
        resp.quantized_weights_re_im = [[float(w.real), float(w.imag)] for w in q_weights]

    return resp


@router.post("/pattern", response_model=PatternResponse)
def compute_pattern(req: PatternRequest) -> PatternResponse:
    positions, d = _build_array(req)
    weights = steering_weights(
        positions,
        req.freq_hz,
        req.steer_az_deg,
        req.steer_el_deg,
        req.taper.value,
    )

    c = 299_792_458.0
    lam = c / req.freq_hz

    az_angles, az_gain = pattern_cut(
        positions,
        weights,
        req.freq_hz,
        sweep="az",
        fixed_deg=req.steer_el_deg,
        sweep_range_deg=req.az_range_deg,
        n_points=req.n_points,
    )
    el_angles, el_gain = pattern_cut(
        positions,
        weights,
        req.freq_hz,
        sweep="el",
        fixed_deg=req.steer_az_deg,
        sweep_range_deg=req.el_range_deg,
        n_points=req.n_points,
    )

    az_peak_idx = int(np.argmax(az_gain))
    el_peak_idx = int(np.argmax(el_gain))

    resp = PatternResponse(
        az_cut=PatternCutModel(
            angles_deg=az_angles.tolist(),
            gain_db=az_gain.tolist(),
            label=f"Az cut (el={req.steer_el_deg:.1f}°)",
        ),
        el_cut=PatternCutModel(
            angles_deg=el_angles.tolist(),
            gain_db=el_gain.tolist(),
            label=f"El cut (az={req.steer_az_deg:.1f}°)",
        ),
        peak_az_deg=float(az_angles[az_peak_idx]),
        peak_el_deg=float(el_angles[el_peak_idx]),
        peak_gain_db=0.0,
        n_elements=len(positions),
        spacing_lambda=d / lam,
    )

    if req.phase_bits is not None:
        q_weights = steering_weights(
            positions,
            req.freq_hz,
            req.steer_az_deg,
            req.steer_el_deg,
            req.taper.value,
            phase_bits=req.phase_bits,
        )
        q_az_angles, q_az_gain = pattern_cut(
            positions,
            q_weights,
            req.freq_hz,
            sweep="az",
            fixed_deg=req.steer_el_deg,
            sweep_range_deg=req.az_range_deg,
            n_points=req.n_points,
        )
        q_el_angles, q_el_gain = pattern_cut(
            positions,
            q_weights,
            req.freq_hz,
            sweep="el",
            fixed_deg=req.steer_az_deg,
            sweep_range_deg=req.el_range_deg,
            n_points=req.n_points,
        )
        resp.phase_bits = req.phase_bits
        resp.quantized_az_cut = PatternCutModel(
            angles_deg=q_az_angles.tolist(),
            gain_db=q_az_gain.tolist(),
            label=f"Az cut quantized ({req.phase_bits}-bit)",
        )
        resp.quantized_el_cut = PatternCutModel(
            angles_deg=q_el_angles.tolist(),
            gain_db=q_el_gain.tolist(),
            label=f"El cut quantized ({req.phase_bits}-bit)",
        )

    return resp


@router.post("/null_weights", response_model=NullWeightsResponse)
def compute_null_weights(req: NullWeightsRequest) -> NullWeightsResponse:
    positions, d = _build_array(req)

    jammer_tuples = [(j.az_deg, j.el_deg) for j in req.jammer_azels]
    result = compute_lcmv_weights(
        positions,
        req.freq_hz,
        req.steer_az_deg,
        req.steer_el_deg,
        jammer_tuples,
        diag_load=req.diag_load,
    )

    c = 299_792_458.0
    lam = c / req.freq_hz

    az_angles, az_gain = pattern_cut(
        positions,
        result.weights,
        req.freq_hz,
        sweep="az",
        fixed_deg=req.steer_el_deg,
        sweep_range_deg=90.0,
        n_points=601,
    )
    el_angles, el_gain = pattern_cut(
        positions,
        result.weights,
        req.freq_hz,
        sweep="el",
        fixed_deg=req.steer_az_deg,
        sweep_range_deg=90.0,
        n_points=601,
    )

    return NullWeightsResponse(
        n_elements=len(positions),
        positions=positions.tolist(),
        phases_rad=result.phases_rad.tolist(),
        weights_re_im=[[float(w.real), float(w.imag)] for w in result.weights],
        spacing_m=d,
        spacing_lambda=d / lam,
        constraint_residuals_re_im=[
            [float(r.real), float(r.imag)] for r in result.constraint_residuals
        ],
        az_cut=PatternCutModel(
            angles_deg=az_angles.tolist(),
            gain_db=az_gain.tolist(),
            label="Az cut (LCMV)",
        ),
        el_cut=PatternCutModel(
            angles_deg=el_angles.tolist(),
            gain_db=el_gain.tolist(),
            label="El cut (LCMV)",
        ),
    )


@router.post("/null_depth_vs_bits", response_model=NullDepthVsBitsResponse)
def null_depth_vs_bits(req: NullDepthVsBitsRequest) -> NullDepthVsBitsResponse:
    positions, _d = _build_array(req)

    jammer_tuples = [(j.az_deg, j.el_deg) for j in req.jammer_azels]

    bit_settings: list[int | None] = []
    if req.include_continuous:
        bit_settings.append(None)
    bit_settings.extend(req.bit_settings)

    analysis = compute_null_depth_vs_phase_bits(
        positions,
        req.freq_hz,
        req.steer_az_deg,
        req.steer_el_deg,
        jammer_tuples,
        bit_settings,
        diag_load=req.diag_load,
    )

    return NullDepthVsBitsResponse(
        results=[
            BitSettingResultModel(
                label=r.label,
                bits=r.bits,
                desired_gain_mag=r.desired_gain_mag,
                jammer_response_mag=r.jammer_response_mag,
                null_depth_db=r.null_depth_db,
                worst_null_depth_db=r.worst_null_depth_db,
            )
            for r in analysis.bit_settings_results
        ],
        jammer_labels=analysis.jammer_labels,
        summary=NullDepthSummary(
            continuous_worst_null_db=analysis.continuous_worst_null_db,
            best_quantized_worst_null_db=analysis.best_quantized_worst_null_db,
            best_quantized_label=analysis.best_quantized_label,
        ),
    )
