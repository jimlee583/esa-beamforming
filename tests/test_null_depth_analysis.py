"""Tests for null-depth-vs-phase-bits analysis.

Validates that:
- Continuous LCMV weights produce much deeper nulls than coarse quantisation
- Increasing phase bits generally improves null depth
- Desired gain remains stable across bit settings
- API endpoint returns well-formed response
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from array_engine.analysis import compute_null_depth_vs_phase_bits
from array_engine.geometry import rectangular_lattice

FREQ = 10e9


class TestContinuousVsQuantised:
    """Continuous null depth must be significantly deeper than coarse quantised."""

    def test_continuous_deeper_than_3bit(self) -> None:
        pos, _ = rectangular_lattice(FREQ, panel_size_m=0.15)
        result = compute_null_depth_vs_phase_bits(
            pos,
            FREQ,
            0.0,
            0.0,
            jammer_azels=[(25.0, 0.0)],
            bit_settings=[None, 3],
        )
        continuous = result.bit_settings_results[0]
        three_bit = result.bit_settings_results[1]
        assert continuous.worst_null_depth_db < three_bit.worst_null_depth_db - 20, (
            f"Continuous ({continuous.worst_null_depth_db:.1f} dB) should be "
            f"much deeper than 3-bit ({three_bit.worst_null_depth_db:.1f} dB)"
        )

    def test_continuous_deeper_than_4bit(self) -> None:
        pos, _ = rectangular_lattice(FREQ, panel_size_m=0.15)
        result = compute_null_depth_vs_phase_bits(
            pos,
            FREQ,
            0.0,
            0.0,
            jammer_azels=[(25.0, 0.0), (-30.0, 10.0)],
            bit_settings=[None, 4],
        )
        continuous = result.bit_settings_results[0]
        four_bit = result.bit_settings_results[1]
        assert continuous.worst_null_depth_db < four_bit.worst_null_depth_db - 10


class TestMonotonicTrend:
    """Null depth should generally improve from 3-bit to 6-bit."""

    def test_6bit_better_than_3bit(self) -> None:
        pos, _ = rectangular_lattice(FREQ, panel_size_m=0.15)
        result = compute_null_depth_vs_phase_bits(
            pos,
            FREQ,
            0.0,
            0.0,
            jammer_azels=[(25.0, 0.0)],
            bit_settings=[3, 4, 5, 6],
        )
        three_bit = result.bit_settings_results[0]
        six_bit = result.bit_settings_results[3]
        assert six_bit.worst_null_depth_db < three_bit.worst_null_depth_db, (
            f"6-bit ({six_bit.worst_null_depth_db:.1f} dB) should be deeper "
            f"than 3-bit ({three_bit.worst_null_depth_db:.1f} dB)"
        )

    def test_overall_trend_improving(self) -> None:
        """Fit a line to worst null depth vs bits; slope should be negative."""
        pos, _ = rectangular_lattice(FREQ, panel_size_m=0.15)
        bits_list = [3, 4, 5, 6, 7]
        result = compute_null_depth_vs_phase_bits(
            pos,
            FREQ,
            0.0,
            0.0,
            jammer_azels=[(25.0, 0.0)],
            bit_settings=bits_list,
        )
        depths = [r.worst_null_depth_db for r in result.bit_settings_results]
        slope = np.polyfit(bits_list, depths, 1)[0]
        assert slope < 0, f"Trend should improve (negative slope), got {slope:.2f}"


class TestDesiredGainStability:
    """Desired-direction gain should stay near the LCMV target across bit settings."""

    def test_desired_gain_near_unity(self) -> None:
        pos, _ = rectangular_lattice(FREQ, panel_size_m=0.15)
        result = compute_null_depth_vs_phase_bits(
            pos,
            FREQ,
            0.0,
            0.0,
            jammer_azels=[(25.0, 0.0)],
            bit_settings=[None, 3, 4, 5, 6, 7],
        )
        for r in result.bit_settings_results:
            assert r.desired_gain_mag > 0.5, (
                f"{r.label}: desired gain {r.desired_gain_mag:.4f} too low"
            )

    def test_continuous_gain_close_to_one(self) -> None:
        pos, _ = rectangular_lattice(FREQ, panel_size_m=0.15)
        result = compute_null_depth_vs_phase_bits(
            pos,
            FREQ,
            0.0,
            0.0,
            jammer_azels=[(25.0, 0.0)],
            bit_settings=[None],
        )
        cont = result.bit_settings_results[0]
        np.testing.assert_allclose(cont.desired_gain_mag, 1.0, atol=1e-6)


class TestMultipleJammers:
    """Per-jammer null depths and labels must be correct."""

    def test_two_jammer_labels(self) -> None:
        pos, _ = rectangular_lattice(FREQ, panel_size_m=0.15)
        result = compute_null_depth_vs_phase_bits(
            pos,
            FREQ,
            0.0,
            0.0,
            jammer_azels=[(25.0, 0.0), (-30.0, 10.0)],
            bit_settings=[None, 5],
        )
        assert len(result.jammer_labels) == 2
        assert "25.0" in result.jammer_labels[0]
        assert "-30.0" in result.jammer_labels[1]
        for r in result.bit_settings_results:
            assert len(r.null_depth_db) == 2
            assert len(r.jammer_response_mag) == 2

    def test_worst_is_max_of_per_jammer(self) -> None:
        pos, _ = rectangular_lattice(FREQ, panel_size_m=0.15)
        result = compute_null_depth_vs_phase_bits(
            pos,
            FREQ,
            0.0,
            0.0,
            jammer_azels=[(20.0, 0.0), (-35.0, 5.0)],
            bit_settings=[None, 4, 6],
        )
        for r in result.bit_settings_results:
            assert r.worst_null_depth_db == pytest.approx(max(r.null_depth_db))


class TestSummary:
    """Result summary fields."""

    def test_summary_populated(self) -> None:
        pos, _ = rectangular_lattice(FREQ, panel_size_m=0.15)
        result = compute_null_depth_vs_phase_bits(
            pos,
            FREQ,
            0.0,
            0.0,
            jammer_azels=[(25.0, 0.0)],
            bit_settings=[None, 4, 6],
        )
        assert result.continuous_worst_null_db < -50
        assert result.best_quantized_worst_null_db is not None
        assert result.best_quantized_label is not None


class TestNullDepthAPI:
    """Integration test for /api/null_depth_vs_bits."""

    def test_response_shape(self) -> None:
        from backend.app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        r = client.post(
            "/api/null_depth_vs_bits",
            json={
                "freq_hz": 10e9,
                "panel_size_m": 0.15,
                "lattice": "rectangular",
                "element_k_lambda": 0.5,
                "steer_az_deg": 0.0,
                "steer_el_deg": 0.0,
                "taper": "uniform",
                "jammer_azels": [
                    {"az_deg": 25.0, "el_deg": 0.0},
                    {"az_deg": -30.0, "el_deg": 10.0},
                ],
                "bit_settings": [3, 4, 5, 6, 7],
                "include_continuous": True,
            },
        )
        assert r.status_code == 200
        data = r.json()

        assert len(data["results"]) == 6
        assert data["results"][0]["label"] == "continuous"
        assert data["results"][0]["bits"] is None
        assert data["results"][1]["label"] == "3-bit"
        assert data["results"][1]["bits"] == 3

        assert len(data["jammer_labels"]) == 2
        for result in data["results"]:
            assert len(result["null_depth_db"]) == 2
            assert len(result["jammer_response_mag"]) == 2
            assert isinstance(result["desired_gain_mag"], float)
            assert isinstance(result["worst_null_depth_db"], float)

        assert "summary" in data
        assert isinstance(data["summary"]["continuous_worst_null_db"], float)
        assert data["summary"]["best_quantized_worst_null_db"] is not None
        assert data["summary"]["best_quantized_label"] is not None

    def test_without_continuous(self) -> None:
        from backend.app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        r = client.post(
            "/api/null_depth_vs_bits",
            json={
                "freq_hz": 10e9,
                "panel_size_m": 0.15,
                "lattice": "rectangular",
                "element_k_lambda": 0.5,
                "steer_az_deg": 0.0,
                "steer_el_deg": 0.0,
                "taper": "uniform",
                "jammer_azels": [{"az_deg": 25.0, "el_deg": 0.0}],
                "bit_settings": [4, 5, 6],
                "include_continuous": False,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 3
        assert all(r["bits"] is not None for r in data["results"])
