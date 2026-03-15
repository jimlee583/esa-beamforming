"""Single-jammer AOA geolocation via beam scanning and spherical-Earth ray intersection.

Pipeline
--------
1. Build a 2-D az/el scan grid.
2. Synthesise beam-scan power using the existing array-factor machinery
   (steer to the true jammer direction, evaluate |AF|² over the grid).
3. Pick the peak as the estimated angle-of-arrival.
4. Convert the body-frame LOS to ECEF using a nadir-pointing local-frame
   convention (body +x → East, +y → North, +z → nadir).
5. Intersect the ECEF ray with a spherical Earth to get the ground point.

Frame conventions
-----------------
Body frame
    Identical to the array/body frame defined in ``steering.py``.
Platform attitude (simplified first-pass, NED convention)
    Nadir-pointing: body +x → North, +y → East, +z → Down (nadir).
Earth model
    Sphere of radius 6 371 000 m (mean radius, sea level).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from array_engine.pattern import array_factor
from array_engine.steering import azel_to_unit_vector, steering_weights

EARTH_RADIUS_M = 6_371_000.0
_EXCLUSION_RADIUS_DEG = 5.0


@dataclass
class AOAGeolocationResult:
    """Full output of the single-jammer AOA geolocation pipeline."""

    estimated_az_deg: float
    estimated_el_deg: float
    los_body: np.ndarray
    los_ecef: np.ndarray
    platform_ecef: np.ndarray
    intersection_found: bool
    intersection_ecef: np.ndarray | None
    intersection_lat_deg: float | None
    intersection_lon_deg: float | None
    n_elements: int
    spacing_m: float
    spacing_lambda: float
    peak_power_db: float
    ambiguity_margin_db: float
    az_cut_angles_deg: np.ndarray
    az_cut_power_db: np.ndarray
    el_cut_angles_deg: np.ndarray
    el_cut_power_db: np.ndarray


# ── Geodesy helpers (spherical Earth) ──────────────────────────────────────


def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Geodetic (lat, lon, alt) to ECEF on a spherical Earth."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = EARTH_RADIUS_M + alt_m
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ])


def ecef_to_lla(pos: np.ndarray) -> tuple[float, float, float]:
    """ECEF to geodetic (lat_deg, lon_deg, alt_m) on a spherical Earth."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r = np.sqrt(x * x + y * y + z * z)
    lat_deg = float(np.degrees(np.arcsin(z / r)))
    lon_deg = float(np.degrees(np.arctan2(y, x)))
    alt_m = r - EARTH_RADIUS_M
    return lat_deg, lon_deg, alt_m


def body_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """3x3 rotation from body frame to ECEF (nadir-pointing, NED convention).

    Mapping: body +x → North, body +y → East, body +z → Down (nadir).
    This is a proper rotation (det = +1).
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e_north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat),
    ])
    e_east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    e_down = np.array([
        -np.cos(lat) * np.cos(lon),
        -np.cos(lat) * np.sin(lon),
        -np.sin(lat),
    ])
    return np.column_stack([e_north, e_east, e_down])


def ray_sphere_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float = EARTH_RADIUS_M,
) -> np.ndarray | None:
    """Nearest forward intersection of a ray with a sphere at the origin.

    Returns the 3-D intersection point, or ``None`` if the ray misses.
    """
    d = direction / np.linalg.norm(direction)
    od = float(np.dot(origin, d))
    oo = float(np.dot(origin, origin))
    disc = od * od - (oo - radius * radius)
    if disc < 0:
        return None

    sqrt_disc = float(np.sqrt(disc))
    t1 = -od - sqrt_disc
    t2 = -od + sqrt_disc

    t = t1 if t1 > 0 else t2
    if t <= 0:
        return None
    result: np.ndarray = origin + t * d
    return result


# ── 2-D beam scan ─────────────────────────────────────────────────────────


def scan_2d(
    positions: np.ndarray,
    weights: np.ndarray,
    freq_hz: float,
    az_edges: np.ndarray,
    el_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2-D beam-scan power map.

    Parameters
    ----------
    positions : (N, 3) element positions
    weights   : (N,) complex beamforming weights (steered to jammer direction)
    freq_hz   : carrier frequency
    az_edges  : 1-D azimuth scan angles (degrees)
    el_edges  : 1-D elevation scan angles (degrees)

    Returns
    -------
    AZ, EL   : meshgrid arrays (n_az, n_el) in degrees
    power_db  : normalised beam power in dB (peak = 0 dB), same shape
    """
    AZ, EL = np.meshgrid(az_edges, el_edges, indexing="ij")
    af = array_factor(positions, weights, freq_hz, AZ, EL)
    power = np.abs(af) ** 2
    peak = power.max()
    if peak == 0:
        power_db = np.full_like(power, -100.0)
    else:
        with np.errstate(divide="ignore"):
            power_db = 10.0 * np.log10(power / peak)
        power_db = np.clip(power_db, -100.0, 0.0)
    return AZ, EL, power_db


# ── Main pipeline ─────────────────────────────────────────────────────────


def aoa_geolocate(
    positions: np.ndarray,
    freq_hz: float,
    jammer_az_deg: float,
    jammer_el_deg: float,
    platform_lat_deg: float,
    platform_lon_deg: float,
    platform_alt_m: float,
    *,
    scan_az_range_deg: float = 60.0,
    scan_el_range_deg: float = 60.0,
    scan_n_az: int = 181,
    scan_n_el: int = 181,
    taper: str = "uniform",
    spacing_m: float = 0.0,
) -> AOAGeolocationResult:
    """Run the full single-jammer AOA geolocation pipeline."""
    c = 299_792_458.0
    lam = c / freq_hz

    # 1. Synthesise beam-scan power
    weights = steering_weights(positions, freq_hz, jammer_az_deg, jammer_el_deg, taper)

    az_edges = np.linspace(-scan_az_range_deg, scan_az_range_deg, scan_n_az)
    el_edges = np.linspace(-scan_el_range_deg, scan_el_range_deg, scan_n_el)
    AZ, EL, power_db = scan_2d(positions, weights, freq_hz, az_edges, el_edges)

    # 2. Peak picking
    peak_idx = np.unravel_index(int(np.argmax(power_db)), power_db.shape)
    est_az = float(AZ[peak_idx])
    est_el = float(EL[peak_idx])

    dist_sq = (AZ - est_az) ** 2 + (EL - est_el) ** 2
    mask = dist_sq > _EXCLUSION_RADIUS_DEG**2
    ambiguity_margin_db = float(-power_db[mask].max()) if mask.any() else float("inf")

    az_cut_power = power_db[:, peak_idx[1]]
    el_cut_power = power_db[peak_idx[0], :]

    # 3. Body-frame LOS
    los_body = azel_to_unit_vector(est_az, est_el)

    # 4. Frame conversion
    R_b2e = body_to_ecef_rotation(platform_lat_deg, platform_lon_deg)
    los_ecef_vec: np.ndarray = R_b2e @ los_body
    platform_ecef = lla_to_ecef(platform_lat_deg, platform_lon_deg, platform_alt_m)

    # 5. Earth intersection
    hit = ray_sphere_intersection(platform_ecef, los_ecef_vec)
    hit_lat: float | None = None
    hit_lon: float | None = None
    if hit is not None:
        hit_lat, hit_lon, _ = ecef_to_lla(hit)

    return AOAGeolocationResult(
        estimated_az_deg=est_az,
        estimated_el_deg=est_el,
        los_body=los_body,
        los_ecef=los_ecef_vec,
        platform_ecef=platform_ecef,
        intersection_found=hit is not None,
        intersection_ecef=hit,
        intersection_lat_deg=hit_lat,
        intersection_lon_deg=hit_lon,
        n_elements=len(positions),
        spacing_m=spacing_m,
        spacing_lambda=spacing_m / lam if spacing_m > 0 else 0.0,
        peak_power_db=0.0,
        ambiguity_margin_db=ambiguity_margin_db,
        az_cut_angles_deg=az_edges,
        az_cut_power_db=az_cut_power,
        el_cut_angles_deg=el_edges,
        el_cut_power_db=el_cut_power,
    )
