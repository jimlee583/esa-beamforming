"""API route handlers."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from array_engine.geometry import rectangular_lattice, triangular_lattice
from array_engine.models import (
    LatticeType,
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
from array_engine.steering import steering_phases, steering_weights

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

    return WeightsResponse(
        n_elements=len(positions),
        positions=positions.tolist(),
        phases_rad=phases.tolist(),
        weights_re_im=[[float(w.real), float(w.imag)] for w in weights],
        spacing_m=d,
        spacing_lambda=d / lam,
    )


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

    # Find peak direction from the az cut
    az_peak_idx = int(np.argmax(az_gain))
    el_peak_idx = int(np.argmax(el_gain))

    return PatternResponse(
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
