"""Tests for steering vector conventions and correctness.

These tests guard against the most common sign / unit bugs in phased-array code.
"""

from __future__ import annotations

import numpy as np

from array_engine.geometry import rectangular_lattice
from array_engine.steering import azel_to_unit_vector, steering_phases, steering_weights

# ── az/el → unit-vector mapping ─────────────────────────────────────────────


class TestAzelToUnitVector:
    def test_broadside(self):
        """az=0, el=0 → boresight +z."""
        u = azel_to_unit_vector(0.0, 0.0)
        np.testing.assert_allclose(u, [0, 0, 1], atol=1e-15)

    def test_az90(self):
        """az=90, el=0 → +x direction."""
        u = azel_to_unit_vector(90.0, 0.0)
        np.testing.assert_allclose(u, [1, 0, 0], atol=1e-15)

    def test_el90(self):
        """az=0, el=90 → +y direction."""
        u = azel_to_unit_vector(0.0, 90.0)
        np.testing.assert_allclose(u, [0, 1, 0], atol=1e-15)

    def test_unit_norm(self):
        """Result is always a unit vector for arbitrary angles."""
        for az in [-45, 0, 30, 60, 90]:
            for el in [-30, 0, 15, 45, 90]:
                u = azel_to_unit_vector(az, el)
                assert abs(np.linalg.norm(u) - 1.0) < 1e-14


# ── Broadside phases ────────────────────────────────────────────────────────


class TestBroadsidePhases:
    """Broadside (az=0, el=0) must produce zero phases for a planar array."""

    def test_rectangular_broadside_phases_zero(self):
        freq = 10e9
        pos, _ = rectangular_lattice(freq, panel_size_m=0.1)
        phases = steering_phases(pos, freq, az_deg=0.0, el_deg=0.0)
        np.testing.assert_allclose(phases, 0.0, atol=1e-12)

    def test_broadside_weights_equal(self):
        freq = 10e9
        pos, _ = rectangular_lattice(freq, panel_size_m=0.1)
        w = steering_weights(pos, freq, az_deg=0.0, el_deg=0.0)
        np.testing.assert_allclose(w, 1.0 + 0j, atol=1e-12)


# ── Steering toward +x ──────────────────────────────────────────────────────


class TestSteerToX:
    """Steering toward +x should produce a linear phase ramp along x."""

    def test_phase_ramp_along_x(self):
        freq = 10e9
        c = 299_792_458.0
        lam = c / freq
        d = lam / 2.0

        # Simple 1-D array along x: 5 elements
        positions = np.column_stack(
            [
                np.arange(5) * d - 2 * d,
                np.zeros(5),
                np.zeros(5),
            ]
        )

        # Steer to az=90 (pure +x), el=0
        phases = steering_phases(positions, freq, az_deg=90.0, el_deg=0.0)

        # φ_n = -k * r_n · u;  u=[1,0,0],  r_n·u = x_n
        k = 2 * np.pi / lam
        expected = -k * positions[:, 0]
        np.testing.assert_allclose(phases, expected, atol=1e-10)

    def test_phase_differences_constant(self):
        """Adjacent elements should have a constant phase difference."""
        freq = 10e9
        c = 299_792_458.0
        lam = c / freq
        d = lam / 2.0

        positions = np.column_stack(
            [
                np.arange(5) * d,
                np.zeros(5),
                np.zeros(5),
            ]
        )
        phases = steering_phases(positions, freq, az_deg=30.0, el_deg=0.0)
        diffs = np.diff(phases)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-10)
