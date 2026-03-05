"""Golden-reference tests with hand-calculable tiny arrays.

These use arrays small enough (2×2 and 3×1) that expected phases can be
verified by hand, providing a solid regression anchor.
"""

from __future__ import annotations

import numpy as np
import pytest

from array_engine.steering import steering_phases, steering_weights

# ── Helpers ──────────────────────────────────────────────────────────────────

FREQ = 10e9
C = 299_792_458.0
LAM = C / FREQ
K = 2.0 * np.pi / LAM
D = LAM / 2.0  # half-wavelength spacing


class TestGolden2x2:
    """2×2 array at half-wave spacing, centred at origin.

    Elements at:
        (-d/2, -d/2, 0)   (-d/2, +d/2, 0)
        (+d/2, -d/2, 0)   (+d/2, +d/2, 0)
    """

    @pytest.fixture
    def pos(self) -> np.ndarray:
        h = D / 2.0
        return np.array(
            [
                [-h, -h, 0.0],
                [-h, +h, 0.0],
                [+h, -h, 0.0],
                [+h, +h, 0.0],
            ]
        )

    def test_broadside_phases(self, pos: np.ndarray):
        """At broadside (az=0,el=0), u=[0,0,1], all r·u=0 → phases=0."""
        phases = steering_phases(pos, FREQ, az_deg=0.0, el_deg=0.0)
        np.testing.assert_allclose(phases, [0.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_steer_pure_x(self, pos: np.ndarray):
        """Steer to az=90 (u=[1,0,0]).

        φ_n = -k * x_n
        x values: -d/2, -d/2, +d/2, +d/2
        Expected: +k*d/2, +k*d/2, -k*d/2, -k*d/2
        Since d = λ/2:  k*d/2 = (2π/λ)(λ/4) = π/2
        """
        phases = steering_phases(pos, FREQ, az_deg=90.0, el_deg=0.0)
        expected = np.array([np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2])
        np.testing.assert_allclose(phases, expected, atol=1e-10)

    def test_steer_pure_y(self, pos: np.ndarray):
        """Steer to el=90 (u=[0,1,0]).

        φ_n = -k * y_n
        y values: -d/2, +d/2, -d/2, +d/2
        Expected: +π/2, -π/2, +π/2, -π/2
        """
        phases = steering_phases(pos, FREQ, az_deg=0.0, el_deg=90.0)
        expected = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2])
        np.testing.assert_allclose(phases, expected, atol=1e-10)

    def test_weights_magnitude_uniform(self, pos: np.ndarray):
        """Uniform taper ⇒ all |w_n| = 1."""
        w = steering_weights(pos, FREQ, az_deg=30.0, el_deg=15.0)
        np.testing.assert_allclose(np.abs(w), 1.0, atol=1e-14)


class TestGolden3x1:
    """3-element linear array along x at half-wave spacing.

    Elements at x = {-d, 0, +d}, y=z=0.
    """

    @pytest.fixture
    def pos(self) -> np.ndarray:
        return np.array(
            [
                [-D, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [+D, 0.0, 0.0],
            ]
        )

    def test_broadside(self, pos: np.ndarray):
        phases = steering_phases(pos, FREQ, az_deg=0.0, el_deg=0.0)
        np.testing.assert_allclose(phases, [0, 0, 0], atol=1e-12)

    def test_steer_az30(self, pos: np.ndarray):
        """Steer to az=30°, el=0°.  u=[sin30, 0, cos30] = [0.5, 0, √3/2].

        φ_n = -k * x_n * sin(30°) = -k * x_n * 0.5
        x = {-d, 0, d}  →  φ = {+k*d*0.5, 0, -k*d*0.5}
        k*d = 2π/λ * λ/2 = π   →  φ = {+π/2, 0, -π/2}
        """
        phases = steering_phases(pos, FREQ, az_deg=30.0, el_deg=0.0)
        expected = np.array([np.pi / 2, 0.0, -np.pi / 2])
        np.testing.assert_allclose(phases, expected, atol=1e-10)

    def test_steer_az30_weights(self, pos: np.ndarray):
        """Verify complex weights match exp(jφ) for golden phases."""
        w = steering_weights(pos, FREQ, az_deg=30.0, el_deg=0.0)
        expected_phases = np.array([np.pi / 2, 0.0, -np.pi / 2])
        np.testing.assert_allclose(np.angle(w), expected_phases, atol=1e-10)
        np.testing.assert_allclose(np.abs(w), 1.0, atol=1e-14)
