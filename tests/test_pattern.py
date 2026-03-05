"""Tests for pattern computation — peak direction and basic shape."""

from __future__ import annotations

import numpy as np
import pytest

from array_engine.geometry import rectangular_lattice
from array_engine.pattern import pattern_cut
from array_engine.steering import steering_weights


class TestPatternPeak:
    """The pattern peak must occur near the requested steer direction."""

    @pytest.mark.parametrize("steer_az", [0.0, 15.0, -20.0, 30.0])
    def test_az_peak_near_steer(self, steer_az: float):
        freq = 10e9
        pos, _ = rectangular_lattice(freq, panel_size_m=0.15)
        w = steering_weights(pos, freq, az_deg=steer_az, el_deg=0.0)
        angles, gain = pattern_cut(
            pos,
            w,
            freq,
            sweep="az",
            fixed_deg=0.0,
            sweep_range_deg=60.0,
            n_points=601,
        )
        peak_idx = int(np.argmax(gain))
        assert abs(angles[peak_idx] - steer_az) < 1.5, (
            f"Peak at {angles[peak_idx]:.2f}° but steered to {steer_az}°"
        )

    def test_broadside_peak_at_zero(self):
        freq = 10e9
        pos, _ = rectangular_lattice(freq, panel_size_m=0.15)
        w = steering_weights(pos, freq, az_deg=0.0, el_deg=0.0)
        angles, gain = pattern_cut(
            pos,
            w,
            freq,
            sweep="az",
            fixed_deg=0.0,
            sweep_range_deg=60.0,
            n_points=601,
        )
        peak_idx = int(np.argmax(gain))
        assert abs(angles[peak_idx]) < 0.5
