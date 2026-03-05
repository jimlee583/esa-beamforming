"""Tests for LCMV null-steering weight computation.

Verifies constraint satisfaction (w^T a(u) = desired) and null depth
improvement relative to uniform steering weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from array_engine.geometry import rectangular_lattice
from array_engine.nulling import _steering_vector, compute_lcmv_weights
from array_engine.pattern import pattern_cut
from array_engine.steering import steering_weights

FREQ = 10e9
C = 299_792_458.0
LAM = C / FREQ


class TestConstraintSatisfaction:
    """LCMV weights must satisfy  w^T a(u) = f  (matching array-factor convention)."""

    @pytest.fixture
    def array_pos(self) -> np.ndarray:
        pos, _d = rectangular_lattice(FREQ, panel_size_m=0.15)
        return pos

    def test_one_jammer(self, array_pos: np.ndarray) -> None:
        result = compute_lcmv_weights(
            array_pos,
            FREQ,
            steer_az_deg=0.0,
            steer_el_deg=0.0,
            jammer_azels=[(30.0, 0.0)],
        )

        a_steer = _steering_vector(array_pos, FREQ, 0.0, 0.0)
        a_jammer = _steering_vector(array_pos, FREQ, 30.0, 0.0)

        # w^T a(steer) should be ~1
        gain_steer = result.weights @ a_steer
        np.testing.assert_allclose(gain_steer, 1.0, atol=1e-6)

        # w^T a(jammer) should be ~0
        gain_jammer = result.weights @ a_jammer
        np.testing.assert_allclose(abs(gain_jammer), 0.0, atol=1e-6)

    def test_two_jammers(self, array_pos: np.ndarray) -> None:
        result = compute_lcmv_weights(
            array_pos,
            FREQ,
            steer_az_deg=0.0,
            steer_el_deg=0.0,
            jammer_azels=[(20.0, 0.0), (-25.0, 10.0)],
        )

        a_steer = _steering_vector(array_pos, FREQ, 0.0, 0.0)
        a_j1 = _steering_vector(array_pos, FREQ, 20.0, 0.0)
        a_j2 = _steering_vector(array_pos, FREQ, -25.0, 10.0)

        np.testing.assert_allclose(result.weights @ a_steer, 1.0, atol=1e-6)
        np.testing.assert_allclose(abs(result.weights @ a_j1), 0.0, atol=1e-6)
        np.testing.assert_allclose(abs(result.weights @ a_j2), 0.0, atol=1e-6)

    def test_constraint_residuals_small(self, array_pos: np.ndarray) -> None:
        result = compute_lcmv_weights(
            array_pos,
            FREQ,
            steer_az_deg=10.0,
            steer_el_deg=5.0,
            jammer_azels=[(40.0, 0.0)],
        )
        np.testing.assert_allclose(np.abs(result.constraint_residuals), 0.0, atol=1e-8)


class TestNullDepth:
    """Pattern null depth at jammer direction must be significantly deeper than uniform."""

    def test_null_depth_single_jammer(self) -> None:
        pos, _d = rectangular_lattice(FREQ, panel_size_m=0.15)
        jammer_az = 25.0

        w_uniform = steering_weights(pos, FREQ, az_deg=0.0, el_deg=0.0)
        angles_u, gain_u = pattern_cut(
            pos,
            w_uniform,
            FREQ,
            sweep="az",
            fixed_deg=0.0,
            sweep_range_deg=60.0,
            n_points=601,
        )

        result = compute_lcmv_weights(
            pos,
            FREQ,
            steer_az_deg=0.0,
            steer_el_deg=0.0,
            jammer_azels=[(jammer_az, 0.0)],
        )
        _angles_n, gain_n = pattern_cut(
            pos,
            result.weights,
            FREQ,
            sweep="az",
            fixed_deg=0.0,
            sweep_range_deg=60.0,
            n_points=601,
        )

        jammer_idx = int(np.argmin(np.abs(angles_u - jammer_az)))
        gain_at_jammer_uniform = gain_u[jammer_idx]
        gain_at_jammer_lcmv = gain_n[jammer_idx]

        improvement = gain_at_jammer_uniform - gain_at_jammer_lcmv
        assert improvement > 20.0, f"Null depth improvement {improvement:.1f} dB, expected > 20 dB"

    def test_null_depth_two_jammers(self) -> None:
        pos, _d = rectangular_lattice(FREQ, panel_size_m=0.15)
        jammer1_az, jammer2_az = 20.0, -30.0

        w_uniform = steering_weights(pos, FREQ, az_deg=0.0, el_deg=0.0)
        angles_u, gain_u = pattern_cut(
            pos,
            w_uniform,
            FREQ,
            sweep="az",
            fixed_deg=0.0,
            sweep_range_deg=60.0,
            n_points=601,
        )

        result = compute_lcmv_weights(
            pos,
            FREQ,
            steer_az_deg=0.0,
            steer_el_deg=0.0,
            jammer_azels=[(jammer1_az, 0.0), (jammer2_az, 0.0)],
        )
        _angles_n, gain_n = pattern_cut(
            pos,
            result.weights,
            FREQ,
            sweep="az",
            fixed_deg=0.0,
            sweep_range_deg=60.0,
            n_points=601,
        )

        for jaz in [jammer1_az, jammer2_az]:
            idx = int(np.argmin(np.abs(angles_u - jaz)))
            improvement = gain_u[idx] - gain_n[idx]
            assert improvement > 20.0, (
                f"Null at {jaz}: improvement {improvement:.1f} dB, expected > 20 dB"
            )


class TestNullingAPI:
    """Integration test for the /api/null_weights endpoint."""

    def test_null_weights_endpoint(self) -> None:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

        from backend.app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        r = client.post(
            "/api/null_weights",
            json={
                "freq_hz": 10e9,
                "panel_size_m": 0.15,
                "lattice": "rectangular",
                "element_k_lambda": 0.5,
                "steer_az_deg": 0.0,
                "steer_el_deg": 0.0,
                "taper": "uniform",
                "jammer_azels": [{"az_deg": 30.0, "el_deg": 0.0}],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["n_elements"] > 0
        assert len(data["weights_re_im"]) == data["n_elements"]
        assert len(data["constraint_residuals_re_im"]) == 2
        for ri in data["constraint_residuals_re_im"]:
            assert abs(ri[0]) < 1e-6 and abs(ri[1]) < 1e-6
        assert data["az_cut"] is not None
        assert data["el_cut"] is not None
