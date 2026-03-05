"""
Element-position generators for planar phased-array panels.

All positions are returned in the **array / body frame**:
    x — horizontal across the panel
    y — vertical across the panel
    z — boresight (outward normal); always 0 for a planar array

Positions are centred so the array centroid is at the origin.
"""

from __future__ import annotations

import numpy as np


def _wavelength(freq_hz: float) -> float:
    c = 299_792_458.0  # m/s
    return c / freq_hz


def _resolve_spacing(
    freq_hz: float,
    spacing_m: float | None,
    element_k_lambda: float,
) -> float:
    """Return element spacing in metres.

    If *spacing_m* is provided it wins; otherwise spacing = element_k_lambda * λ.
    """
    if spacing_m is not None:
        return spacing_m
    return element_k_lambda * _wavelength(freq_hz)


def rectangular_lattice(
    freq_hz: float,
    panel_size_m: float = 1.0,
    spacing_m: float | None = None,
    element_k_lambda: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Generate a rectangular-grid element layout.

    Returns
    -------
    positions : ndarray, shape (N, 3)
        Element centres in metres (array frame).
    d : float
        Actual element spacing used (metres).
    """
    d = _resolve_spacing(freq_hz, spacing_m, element_k_lambda)
    half = panel_size_m / 2.0
    xs = np.arange(-half, half + d * 0.01, d)
    ys = np.arange(-half, half + d * 0.01, d)
    xs -= xs.mean()
    ys -= ys.mean()
    gx, gy = np.meshgrid(xs, ys)
    positions = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    return positions, d


def triangular_lattice(
    freq_hz: float,
    panel_size_m: float = 1.0,
    spacing_m: float | None = None,
    element_k_lambda: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Generate a triangular (hex-packed) element layout.

    Rows are offset by d/2 in x on alternating y-rows.
    """
    d = _resolve_spacing(freq_hz, spacing_m, element_k_lambda)
    dy = d * np.sqrt(3) / 2.0
    half = panel_size_m / 2.0

    rows_y = np.arange(-half, half + dy * 0.01, dy)
    rows_y -= rows_y.mean()

    points: list[list[float]] = []
    for i, y in enumerate(rows_y):
        offset = d / 2.0 if i % 2 else 0.0
        xs = np.arange(-half + offset, half + d * 0.01, d)
        xs -= xs.mean()
        for x in xs:
            if abs(x) <= half + 1e-9 and abs(y) <= half + 1e-9:
                points.append([x, y, 0.0])

    positions = np.array(points)
    positions -= positions.mean(axis=0)
    return positions, d
