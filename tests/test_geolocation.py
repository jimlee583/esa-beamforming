"""Tests for AOA geolocation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from array_engine.geolocation import (
    EARTH_RADIUS_M,
    AOAGeolocationResult,
    aoa_geolocate,
    body_to_ecef_rotation,
    ecef_to_lla,
    lla_to_ecef,
    ray_sphere_intersection,
    scan_2d,
)
from array_engine.geometry import rectangular_lattice
from array_engine.steering import steering_weights

# ── Geodesy helpers ────────────────────────────────────────────────────────


class TestLlaToEcef:
    def test_equator_prime_meridian(self):
        pos = lla_to_ecef(0.0, 0.0, 0.0)
        np.testing.assert_allclose(pos, [EARTH_RADIUS_M, 0, 0], atol=1.0)

    def test_north_pole(self):
        pos = lla_to_ecef(90.0, 0.0, 0.0)
        np.testing.assert_allclose(pos, [0, 0, EARTH_RADIUS_M], atol=1.0)

    def test_south_pole(self):
        pos = lla_to_ecef(-90.0, 0.0, 0.0)
        np.testing.assert_allclose(pos, [0, 0, -EARTH_RADIUS_M], atol=1.0)

    def test_altitude_increases_radius(self):
        alt = 500_000.0
        pos = lla_to_ecef(0.0, 0.0, alt)
        assert abs(np.linalg.norm(pos) - (EARTH_RADIUS_M + alt)) < 1.0


class TestEcefToLla:
    def test_roundtrip(self):
        lat, lon, alt = 35.0, -120.0, 400_000.0
        pos = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(pos)
        assert abs(lat2 - lat) < 1e-10
        assert abs(lon2 - lon) < 1e-10
        assert abs(alt2 - alt) < 1.0

    def test_equator(self):
        pos = np.array([EARTH_RADIUS_M, 0.0, 0.0])
        lat, lon, alt = ecef_to_lla(pos)
        assert abs(lat) < 1e-10
        assert abs(lon) < 1e-10
        assert abs(alt) < 1.0

    def test_negative_longitude(self):
        lat, lon, alt = 10.0, -45.0, 0.0
        pos = lla_to_ecef(lat, lon, alt)
        lat2, lon2, _alt2 = ecef_to_lla(pos)
        assert abs(lat2 - lat) < 1e-10
        assert abs(lon2 - lon) < 1e-10


# ── Frame rotation ─────────────────────────────────────────────────────────


class TestBodyToEcefRotation:
    def test_is_proper_rotation(self):
        R = body_to_ecef_rotation(45.0, 90.0)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_boresight_points_nadir(self):
        """Body +z should map to the nadir direction (−e_up)."""
        lat, lon = 30.0, 60.0
        R = body_to_ecef_rotation(lat, lon)
        boresight_ecef = R @ np.array([0.0, 0.0, 1.0])
        e_up = lla_to_ecef(lat, lon, 0.0) / EARTH_RADIUS_M
        np.testing.assert_allclose(boresight_ecef, -e_up, atol=1e-12)

    def test_x_axis_is_north_at_origin(self):
        R = body_to_ecef_rotation(0.0, 0.0)
        north_ecef = R @ np.array([1.0, 0.0, 0.0])
        # At lat=0, lon=0: North is [0, 0, 1]
        np.testing.assert_allclose(north_ecef, [0, 0, 1], atol=1e-12)

    def test_y_axis_is_east_at_origin(self):
        R = body_to_ecef_rotation(0.0, 0.0)
        east_ecef = R @ np.array([0.0, 1.0, 0.0])
        # At lat=0, lon=0: East is [0, 1, 0]
        np.testing.assert_allclose(east_ecef, [0, 1, 0], atol=1e-12)


# ── Ray-sphere intersection ───────────────────────────────────────────────


class TestRaySphereIntersection:
    def test_hit_from_above(self):
        origin = np.array([0.0, 0.0, EARTH_RADIUS_M + 500e3])
        direction = np.array([0.0, 0.0, -1.0])
        hit = ray_sphere_intersection(origin, direction)
        assert hit is not None
        assert abs(np.linalg.norm(hit) - EARTH_RADIUS_M) < 1.0
        np.testing.assert_allclose(hit, [0, 0, EARTH_RADIUS_M], atol=1.0)

    def test_miss_pointing_away(self):
        origin = np.array([0.0, 0.0, EARTH_RADIUS_M + 500e3])
        direction = np.array([0.0, 0.0, 1.0])
        assert ray_sphere_intersection(origin, direction) is None

    def test_miss_perpendicular(self):
        origin = np.array([EARTH_RADIUS_M + 500e3, 0.0, 0.0])
        direction = np.array([0.0, 1.0, 0.0])
        assert ray_sphere_intersection(origin, direction) is None

    def test_returns_nearest_intersection(self):
        origin = np.array([0.0, 0.0, EARTH_RADIUS_M + 500e3])
        direction = np.array([0.0, 0.0, -1.0])
        hit = ray_sphere_intersection(origin, direction)
        assert hit is not None
        assert hit[2] > 0  # nearer intersection is on the +z side


# ── 2-D beam scan ─────────────────────────────────────────────────────────


class TestScan2d:
    def test_peak_at_jammer_direction(self):
        freq_hz = 10e9
        positions, _d = rectangular_lattice(freq_hz, panel_size_m=0.1)
        jammer_az, jammer_el = 15.0, 10.0
        weights = steering_weights(positions, freq_hz, jammer_az, jammer_el, "uniform")

        az_edges = np.linspace(-60, 60, 121)
        el_edges = np.linspace(-60, 60, 121)
        AZ, EL, power_db = scan_2d(positions, weights, freq_hz, az_edges, el_edges)

        peak_idx = np.unravel_index(int(np.argmax(power_db)), power_db.shape)
        peak_az = float(AZ[peak_idx])
        peak_el = float(EL[peak_idx])
        assert abs(peak_az - jammer_az) < 2.0
        assert abs(peak_el - jammer_el) < 2.0

    def test_peak_is_zero_db(self):
        freq_hz = 10e9
        positions, _d = rectangular_lattice(freq_hz, panel_size_m=0.1)
        weights = steering_weights(positions, freq_hz, 0.0, 0.0, "uniform")
        _, _, power_db = scan_2d(
            positions, weights, freq_hz,
            np.linspace(-30, 30, 61), np.linspace(-30, 30, 61),
        )
        assert abs(power_db.max()) < 1e-10

    def test_output_shapes(self):
        freq_hz = 10e9
        positions, _d = rectangular_lattice(freq_hz, panel_size_m=0.1)
        weights = steering_weights(positions, freq_hz, 0.0, 0.0, "uniform")
        n_az, n_el = 41, 31
        AZ, EL, power_db = scan_2d(
            positions, weights, freq_hz,
            np.linspace(-30, 30, n_az), np.linspace(-20, 20, n_el),
        )
        assert AZ.shape == (n_az, n_el)
        assert EL.shape == (n_az, n_el)
        assert power_db.shape == (n_az, n_el)


# ── Full pipeline ─────────────────────────────────────────────────────────


class TestAoaGeolocate:
    @pytest.fixture()
    def array_10ghz(self):
        freq_hz = 10e9
        positions, d = rectangular_lattice(freq_hz, panel_size_m=0.1)
        return positions, d, freq_hz

    def test_boresight_lands_below_platform(self, array_10ghz):
        positions, d, freq_hz = array_10ghz
        lat, lon, alt = 35.0, -120.0, 500_000.0
        result = aoa_geolocate(
            positions, freq_hz,
            jammer_az_deg=0.0, jammer_el_deg=0.0,
            platform_lat_deg=lat, platform_lon_deg=lon, platform_alt_m=alt,
            scan_n_az=61, scan_n_el=61,
            spacing_m=d,
        )
        assert result.intersection_found
        assert result.intersection_lat_deg is not None
        assert result.intersection_lon_deg is not None
        assert abs(result.intersection_lat_deg - lat) < 0.1
        assert abs(result.intersection_lon_deg - lon) < 0.1

    def test_estimated_aoa_matches_jammer(self, array_10ghz):
        positions, d, freq_hz = array_10ghz
        jammer_az, jammer_el = 20.0, 10.0
        result = aoa_geolocate(
            positions, freq_hz,
            jammer_az_deg=jammer_az, jammer_el_deg=jammer_el,
            platform_lat_deg=0.0, platform_lon_deg=0.0, platform_alt_m=500_000.0,
            scan_n_az=121, scan_n_el=121,
            spacing_m=d,
        )
        assert abs(result.estimated_az_deg - jammer_az) < 2.0
        assert abs(result.estimated_el_deg - jammer_el) < 2.0

    def test_off_boresight_intersection_displaced(self, array_10ghz):
        positions, d, freq_hz = array_10ghz
        lat, lon, alt = 0.0, 0.0, 500_000.0
        result = aoa_geolocate(
            positions, freq_hz,
            jammer_az_deg=20.0, jammer_el_deg=0.0,
            platform_lat_deg=lat, platform_lon_deg=lon, platform_alt_m=alt,
            scan_n_az=91, scan_n_el=91,
            spacing_m=d,
        )
        assert result.intersection_found
        assert result.intersection_lat_deg is not None
        assert result.intersection_lon_deg is not None
        dist = np.sqrt(
            (result.intersection_lat_deg - lat) ** 2
            + (result.intersection_lon_deg - lon) ** 2
        )
        assert dist > 0.1

    def test_result_fields(self, array_10ghz):
        positions, d, freq_hz = array_10ghz
        result = aoa_geolocate(
            positions, freq_hz,
            jammer_az_deg=0.0, jammer_el_deg=0.0,
            platform_lat_deg=0.0, platform_lon_deg=0.0, platform_alt_m=500_000.0,
            scan_n_az=31, scan_n_el=31,
            spacing_m=d,
        )
        assert isinstance(result, AOAGeolocationResult)
        assert result.los_body.shape == (3,)
        assert result.los_ecef.shape == (3,)
        assert result.platform_ecef.shape == (3,)
        assert result.peak_power_db == 0.0
        assert result.ambiguity_margin_db > 0
        assert len(result.az_cut_angles_deg) == 31
        assert len(result.el_cut_angles_deg) == 31

    def test_ambiguity_margin_positive(self, array_10ghz):
        positions, d, freq_hz = array_10ghz
        result = aoa_geolocate(
            positions, freq_hz,
            jammer_az_deg=10.0, jammer_el_deg=5.0,
            platform_lat_deg=40.0, platform_lon_deg=-75.0, platform_alt_m=300_000.0,
            scan_n_az=91, scan_n_el=91,
            spacing_m=d,
        )
        assert result.ambiguity_margin_db > 0


# ── API integration ────────────────────────────────────────────────────────


from backend.app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_client = TestClient(app)


class TestAoaGeolocateEndpoint:
    def test_response_shape(self):
        payload = {
            "freq_hz": 10e9,
            "panel_size_m": 0.1,
            "lattice": "rectangular",
            "element_k_lambda": 0.5,
            "taper": "uniform",
            "jammer_az_deg": 10.0,
            "jammer_el_deg": 5.0,
            "platform_lat_deg": 40.0,
            "platform_lon_deg": -75.0,
            "platform_alt_m": 500_000.0,
            "scan_n_az": 61,
            "scan_n_el": 61,
        }
        r = _client.post("/api/aoa_geolocate", json=payload)
        assert r.status_code == 200
        data = r.json()

        assert "estimated_az_deg" in data
        assert "estimated_el_deg" in data
        assert data["intersection_found"] is True
        assert data["intersection_lat_deg"] is not None
        assert data["intersection_lon_deg"] is not None
        assert len(data["los_body"]) == 3
        assert len(data["los_ecef"]) == 3
        assert len(data["platform_ecef"]) == 3
        assert data["n_elements"] > 0
        assert "az_cut" in data
        assert "el_cut" in data

    def test_boresight_consistency(self):
        payload = {
            "freq_hz": 10e9,
            "panel_size_m": 0.1,
            "lattice": "rectangular",
            "element_k_lambda": 0.5,
            "taper": "uniform",
            "jammer_az_deg": 0.0,
            "jammer_el_deg": 0.0,
            "platform_lat_deg": 45.0,
            "platform_lon_deg": 10.0,
            "platform_alt_m": 600_000.0,
            "scan_n_az": 41,
            "scan_n_el": 41,
        }
        r = _client.post("/api/aoa_geolocate", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert abs(data["estimated_az_deg"]) < 2.0
        assert abs(data["estimated_el_deg"]) < 2.0
        assert abs(data["intersection_lat_deg"] - 45.0) < 0.5
        assert abs(data["intersection_lon_deg"] - 10.0) < 0.5
