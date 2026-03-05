"""Pydantic models shared between array_engine and the FastAPI backend."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class LatticeType(StrEnum):
    rectangular = "rectangular"
    triangular = "triangular"


class TaperType(StrEnum):
    uniform = "uniform"
    taylor = "taylor"
    chebyshev = "chebyshev"


class CoordinateFrame(StrEnum):
    """Coordinate frame tag — only 'array' is implemented so far."""

    array = "array"
    enu = "enu"
    ecef = "ecef"


# ── Request models ──────────────────────────────────────────────────────────


class WeightsRequest(BaseModel):
    freq_hz: float = Field(..., gt=0, description="Carrier frequency in Hz")
    panel_size_m: float = Field(1.0, gt=0, description="Side length of the square panel in metres")
    lattice: LatticeType = LatticeType.rectangular
    spacing_m: float | None = Field(
        None,
        gt=0,
        description="Element spacing in metres (overrides element_k_lambda)",
    )
    element_k_lambda: float = Field(
        0.5,
        gt=0,
        le=1.0,
        description="Element spacing as a fraction of wavelength (default λ/2)",
    )
    steer_az_deg: float = Field(0.0, description="Steering azimuth in degrees")
    steer_el_deg: float = Field(0.0, description="Steering elevation in degrees")
    taper: TaperType = TaperType.uniform
    coordinate_frame: CoordinateFrame = CoordinateFrame.array


class PatternRequest(WeightsRequest):
    """Same inputs as WeightsRequest; pattern endpoints just compute more."""

    az_range_deg: float = Field(
        90.0, gt=0, le=180, description="Half-span of azimuth cut in degrees"
    )
    el_range_deg: float = Field(
        90.0, gt=0, le=180, description="Half-span of elevation cut in degrees"
    )
    n_points: int = Field(361, ge=10, le=3601, description="Points per cut")


class AzEl(BaseModel):
    az_deg: float
    el_deg: float


class NullWeightsRequest(WeightsRequest):
    """Same array parameters as WeightsRequest, plus jammer directions."""

    jammer_azels: list[AzEl] = Field(..., min_length=1, description="Jammer directions to null")
    diag_load: float = Field(1e-6, gt=0, description="Diagonal loading for LCMV")


# ── Response models ─────────────────────────────────────────────────────────


class WeightsResponse(BaseModel):
    n_elements: int
    positions: list[list[float]] = Field(
        ..., description="Element positions [[x,y,z], …] in metres"
    )
    phases_rad: list[float]
    weights_re_im: list[list[float]] = Field(..., description="Complex weights as [[re, im], …]")
    spacing_m: float
    spacing_lambda: float


class PatternCut(BaseModel):
    angles_deg: list[float]
    gain_db: list[float]
    label: str


class PatternResponse(BaseModel):
    az_cut: PatternCut
    el_cut: PatternCut
    peak_az_deg: float
    peak_el_deg: float
    peak_gain_db: float
    n_elements: int
    spacing_lambda: float


class NullWeightsResponse(BaseModel):
    n_elements: int
    positions: list[list[float]] = Field(
        ..., description="Element positions [[x,y,z], ...] in metres"
    )
    phases_rad: list[float]
    weights_re_im: list[list[float]] = Field(..., description="Complex weights as [[re, im], ...]")
    spacing_m: float
    spacing_lambda: float
    constraint_residuals_re_im: list[list[float]] = Field(
        ..., description="Constraint residuals as [[re, im], ...]"
    )
    az_cut: PatternCut | None = None
    el_cut: PatternCut | None = None
