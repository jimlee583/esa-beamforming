"""Tests for array geometry generators."""

from __future__ import annotations

import numpy as np
import pytest

from array_engine.geometry import rectangular_lattice, triangular_lattice


class TestRectangularLattice:
    def test_centred(self):
        pos, _d = rectangular_lattice(10e9, panel_size_m=0.1)
        centroid = pos.mean(axis=0)
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-12)

    def test_z_zero(self):
        pos, _ = rectangular_lattice(10e9, panel_size_m=0.1)
        np.testing.assert_allclose(pos[:, 2], 0.0, atol=1e-15)

    def test_explicit_spacing(self):
        _pos, d = rectangular_lattice(10e9, panel_size_m=0.1, spacing_m=0.02)
        assert abs(d - 0.02) < 1e-15

    def test_element_count_reasonable(self):
        pos, d = rectangular_lattice(10e9, panel_size_m=1.0)
        n_per_side = int(1.0 / d) + 1
        assert len(pos) == pytest.approx(n_per_side**2, abs=n_per_side)


class TestTriangularLattice:
    def test_centred(self):
        pos, _d = triangular_lattice(10e9, panel_size_m=0.1)
        centroid = pos.mean(axis=0)
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-10)

    def test_z_zero(self):
        pos, _ = triangular_lattice(10e9, panel_size_m=0.1)
        np.testing.assert_allclose(pos[:, 2], 0.0, atol=1e-15)

    def test_more_elements_than_rect(self):
        """Triangular packing generally fits more elements in the same area."""
        pos_r, _ = rectangular_lattice(10e9, panel_size_m=0.5)
        pos_t, _ = triangular_lattice(10e9, panel_size_m=0.5)
        assert len(pos_t) >= len(pos_r) * 0.9
