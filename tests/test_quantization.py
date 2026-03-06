"""Tests for phase quantization correctness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from array_engine.geometry import rectangular_lattice
from array_engine.pattern import pattern_cut
from array_engine.steering import quantize_phase, steering_phases, steering_weights
from backend.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


class TestQuantizePhaseGrid:
    """All quantized values must lie on the allowed discrete grid."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 8])
    def test_output_on_grid_0_2pi(self, bits: int):
        rng = np.random.default_rng(42)
        phases = rng.uniform(-4 * np.pi, 4 * np.pi, size=200)
        q = quantize_phase(phases, bits, phase_range="0_2pi")

        n_levels = 2**bits
        step = 2 * np.pi / n_levels
        grid = np.arange(n_levels) * step

        for val in q:
            diffs = np.abs(grid - val)
            assert diffs.min() < 1e-12, f"{val} not on {bits}-bit grid"

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_output_on_grid_neg_pi_pi(self, bits: int):
        rng = np.random.default_rng(99)
        phases = rng.uniform(-3 * np.pi, 3 * np.pi, size=100)
        q = quantize_phase(phases, bits, phase_range="neg_pi_pi")

        n_levels = 2**bits
        step = 2 * np.pi / n_levels
        grid_0 = np.arange(n_levels) * step
        grid = (grid_0 + np.pi) % (2 * np.pi) - np.pi

        for val in q:
            diffs = np.abs(grid - val)
            assert diffs.min() < 1e-12, f"{val} not on {bits}-bit neg_pi_pi grid"


class TestQuantizePhaseMaxError:
    """Maximum quantization error must be <= step / 2."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6])
    def test_max_error_within_half_step(self, bits: int):
        rng = np.random.default_rng(7)
        phases = rng.uniform(0, 2 * np.pi, size=500)
        q = quantize_phase(phases, bits, phase_range="0_2pi")

        step = 2 * np.pi / (2**bits)
        wrapped = phases % (2 * np.pi)
        error = np.abs(wrapped - q)
        error = np.minimum(error, 2 * np.pi - error)
        assert np.all(error <= step / 2 + 1e-12)


class TestQuantizePhaseEdgeCases:
    def test_bits_1_gives_two_levels(self):
        phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        q = quantize_phase(phases, 1, phase_range="0_2pi")
        assert set(np.round(q, 10)) <= {0.0, round(np.pi, 10)}

    def test_broadside_zero_phases_unchanged(self):
        phases = np.zeros(10)
        for bits in [2, 3, 4, 6]:
            q = quantize_phase(phases, bits)
            np.testing.assert_allclose(q, 0.0, atol=1e-14)

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError, match="bits must be >= 1"):
            quantize_phase(np.array([0.0]), 0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown quantization mode"):
            quantize_phase(np.array([0.0]), 3, mode="bad")

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="Unknown phase_range"):
            quantize_phase(np.array([0.0]), 3, phase_range="bad")


class TestBeamPeakAfterQuantization:
    """Beam peak must remain near the steering direction for a small array."""

    @pytest.mark.parametrize("steer_az", [0.0, 15.0, 30.0])
    @pytest.mark.parametrize("bits", [3, 4, 5])
    def test_peak_near_steer(self, steer_az: float, bits: int):
        freq = 10e9
        pos, _ = rectangular_lattice(freq, panel_size_m=0.1)

        q_weights = steering_weights(
            pos, freq, az_deg=steer_az, el_deg=0.0, phase_bits=bits,
        )

        angles, gain_db = pattern_cut(
            pos, q_weights, freq, sweep="az", fixed_deg=0.0,
            sweep_range_deg=90.0, n_points=721,
        )

        peak_idx = int(np.argmax(gain_db))
        peak_az = float(angles[peak_idx])

        step_deg = 360.0 / (2**bits)
        tolerance = max(step_deg, 2.0)
        assert abs(peak_az - steer_az) < tolerance, (
            f"Peak at {peak_az}° too far from steer {steer_az}° "
            f"for {bits}-bit quantization"
        )


class TestQuantizationApi:
    """The /api/weights and /api/pattern endpoints handle phase_bits correctly."""

    PAYLOAD = {
        "freq_hz": 10e9,
        "panel_size_m": 0.1,
        "lattice": "rectangular",
        "element_k_lambda": 0.5,
        "steer_az_deg": 20.0,
        "steer_el_deg": 0.0,
        "taper": "uniform",
    }

    def test_weights_without_phase_bits(self):
        r = client.post("/api/weights", json=self.PAYLOAD)
        assert r.status_code == 200
        data = r.json()
        assert data["phase_bits"] is None
        assert data["quantized_phases_rad"] is None
        assert data["quantized_weights_re_im"] is None

    def test_weights_with_phase_bits(self):
        body = {**self.PAYLOAD, "phase_bits": 4}
        r = client.post("/api/weights", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["phase_bits"] == 4
        assert len(data["quantized_phases_rad"]) == data["n_elements"]
        assert len(data["quantized_weights_re_im"]) == data["n_elements"]
        for w in data["quantized_weights_re_im"]:
            assert isinstance(w, list) and len(w) == 2

    def test_pattern_without_phase_bits(self):
        body = {**self.PAYLOAD, "n_points": 101}
        r = client.post("/api/pattern", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["phase_bits"] is None
        assert data["quantized_az_cut"] is None
        assert data["quantized_el_cut"] is None

    def test_pattern_with_phase_bits(self):
        body = {**self.PAYLOAD, "n_points": 101, "phase_bits": 3}
        r = client.post("/api/pattern", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["phase_bits"] == 3
        assert len(data["quantized_az_cut"]["angles_deg"]) == 101
        assert len(data["quantized_el_cut"]["gain_db"]) == 101

    def test_continuous_fields_unchanged_with_phase_bits(self):
        body_no_q = {**self.PAYLOAD, "n_points": 101}
        body_q = {**self.PAYLOAD, "n_points": 101, "phase_bits": 4}

        r1 = client.post("/api/pattern", json=body_no_q)
        r2 = client.post("/api/pattern", json=body_q)
        d1, d2 = r1.json(), r2.json()

        assert d1["az_cut"]["gain_db"] == d2["az_cut"]["gain_db"]
        assert d1["el_cut"]["gain_db"] == d2["el_cut"]["gain_db"]
