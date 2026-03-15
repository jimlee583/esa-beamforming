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
        description="Element spacing as a fraction of wavelength (default 0.5; values >1 allowed)",
    )
    steer_az_deg: float = Field(0.0, description="Steering azimuth in degrees")
    steer_el_deg: float = Field(0.0, description="Steering elevation in degrees")
    taper: TaperType = TaperType.uniform
    coordinate_frame: CoordinateFrame = CoordinateFrame.array
    phase_bits: int | None = Field(
        None,
        ge=1,
        le=12,
        description="Phase-shifter resolution in bits (None = continuous)",
    )


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
    phase_bits: int | None = None
    quantized_phases_rad: list[float] | None = None
    quantized_weights_re_im: list[list[float]] | None = Field(
        None, description="Quantized complex weights as [[re, im], …]"
    )


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
    phase_bits: int | None = None
    quantized_az_cut: PatternCut | None = None
    quantized_el_cut: PatternCut | None = None


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


# ── Null depth vs phase bits ────────────────────────────────────────────────


class NullDepthVsBitsRequest(NullWeightsRequest):
    """Analyse null depth degradation across phase-bit settings."""

    bit_settings: list[int] = Field(
        default=[3, 4, 5, 6, 7],
        description="Phase-shifter bit counts to evaluate",
    )
    include_continuous: bool = Field(
        default=True,
        description="Include unquantised (continuous) baseline",
    )


class BitSettingResultModel(BaseModel):
    label: str
    bits: int | None = None
    desired_gain_mag: float
    jammer_response_mag: list[float]
    null_depth_db: list[float]
    worst_null_depth_db: float


class NullDepthSummary(BaseModel):
    continuous_worst_null_db: float
    best_quantized_worst_null_db: float | None = None
    best_quantized_label: str | None = None


class NullDepthVsBitsResponse(BaseModel):
    results: list[BitSettingResultModel]
    jammer_labels: list[str]
    summary: NullDepthSummary


# ── AOA geolocation ────────────────────────────────────────────────────────


class AOAGeolocationRequest(WeightsRequest):
    """Single-jammer AOA geolocation via synthetic beam scan.

    Inherits array parameters from ``WeightsRequest``.  The inherited
    ``steer_az_deg`` / ``steer_el_deg`` fields are unused — the jammer
    direction drives the scan instead.
    """

    jammer_az_deg: float = Field(..., description="True jammer azimuth in body frame (degrees)")
    jammer_el_deg: float = Field(..., description="True jammer elevation in body frame (degrees)")
    platform_lat_deg: float = Field(
        ..., ge=-90, le=90, description="Platform geodetic latitude (degrees)"
    )
    platform_lon_deg: float = Field(
        ..., ge=-180, le=180, description="Platform geodetic longitude (degrees)"
    )
    platform_alt_m: float = Field(
        ..., gt=0, description="Platform altitude above sea level (metres)"
    )
    scan_az_range_deg: float = Field(
        60.0, gt=0, le=180, description="Half-span of azimuth scan (degrees)"
    )
    scan_el_range_deg: float = Field(
        60.0, gt=0, le=180, description="Half-span of elevation scan (degrees)"
    )
    scan_n_az: int = Field(181, ge=10, le=721, description="Azimuth scan grid points")
    scan_n_el: int = Field(181, ge=10, le=721, description="Elevation scan grid points")


class AOAGeolocationResponse(BaseModel):
    """Result of single-jammer AOA geolocation."""

    estimated_az_deg: float
    estimated_el_deg: float
    los_body: list[float] = Field(..., description="LOS unit vector in body frame [x, y, z]")
    los_ecef: list[float] = Field(..., description="LOS unit vector in ECEF [x, y, z]")
    platform_ecef: list[float] = Field(
        ..., description="Platform position in ECEF [x, y, z] (metres)"
    )
    intersection_found: bool
    intersection_ecef: list[float] | None = Field(
        None, description="Ground intersection in ECEF [x, y, z] (metres)"
    )
    intersection_lat_deg: float | None = None
    intersection_lon_deg: float | None = None
    n_elements: int
    spacing_m: float
    spacing_lambda: float
    peak_power_db: float
    ambiguity_margin_db: float = Field(
        ..., description="dB margin between main peak and strongest sidelobe"
    )
    az_cut: PatternCut
    el_cut: PatternCut
