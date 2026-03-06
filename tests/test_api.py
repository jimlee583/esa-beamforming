"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from backend.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


PAYLOAD = {
    "freq_hz": 10e9,
    "panel_size_m": 0.1,
    "lattice": "rectangular",
    "element_k_lambda": 0.5,
    "steer_az_deg": 0.0,
    "steer_el_deg": 0.0,
    "taper": "uniform",
}


class TestHealth:
    def test_health(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestWeightsEndpoint:
    def test_broadside(self):
        r = client.post("/api/weights", json=PAYLOAD)
        assert r.status_code == 200
        data = r.json()
        assert data["n_elements"] > 0
        assert len(data["phases_rad"]) == data["n_elements"]
        assert len(data["weights_re_im"]) == data["n_elements"]
        # broadside ⇒ all phases ≈ 0
        for p in data["phases_rad"]:
            assert abs(p) < 1e-10

    def test_weights_format(self):
        r = client.post("/api/weights", json=PAYLOAD)
        data = r.json()
        for w in data["weights_re_im"]:
            assert isinstance(w, list)
            assert len(w) == 2

    def test_steered(self):
        body = {**PAYLOAD, "steer_az_deg": 20.0}
        r = client.post("/api/weights", json=body)
        assert r.status_code == 200
        data = r.json()
        phases = data["phases_rad"]
        assert not all(abs(p) < 1e-10 for p in phases)

    def test_accepts_superlambda_spacing(self):
        body = {**PAYLOAD, "element_k_lambda": 1.2}
        r = client.post("/api/weights", json=body)
        assert r.status_code == 200
        data = r.json()
        assert abs(data["spacing_lambda"] - 1.2) < 1e-6

    def test_spacing_m_overrides_element_k_lambda(self):
        body = {**PAYLOAD, "element_k_lambda": 1.5, "spacing_m": 0.01}
        r = client.post("/api/weights", json=body)
        assert r.status_code == 200
        data = r.json()

        wavelength_m = 299_792_458.0 / body["freq_hz"]
        expected_spacing_lambda = body["spacing_m"] / wavelength_m
        assert abs(data["spacing_m"] - body["spacing_m"]) < 1e-12
        assert abs(data["spacing_lambda"] - expected_spacing_lambda) < 1e-6


class TestPatternEndpoint:
    def test_pattern(self):
        body = {**PAYLOAD, "n_points": 101}
        r = client.post("/api/pattern", json=body)
        assert r.status_code == 200
        data = r.json()
        assert len(data["az_cut"]["angles_deg"]) == 101
        assert len(data["el_cut"]["gain_db"]) == 101
        assert data["peak_gain_db"] == 0.0
