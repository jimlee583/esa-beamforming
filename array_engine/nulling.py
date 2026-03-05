r"""
LCMV (Linearly Constrained Minimum Variance) null-steering.

Given a desired look direction and one or more jammer directions, solve for
weights that maintain unity gain toward the look direction while placing
nulls toward each jammer.

Formulation
-----------
Our array factor convention (see ``pattern.py``) is:

    AF(u) = Σ_n  w_n · exp(j k r_n · u)  =  w^T a(u)

where a_n(u) = exp(j k r_n · u).

To place a null at direction u_j we need  w^T a(u_j) = 0, and to maintain
unity gain at the steer direction  w^T a(u_s) = 1.

Collecting these into a matrix equation  C^T w = f  where C has columns
a(u_i) and f = [1, 0, …]^T, the equivalent Hermitian form is:

    conj(C)^H w = f       (since f is real)

So the constraint matrix we hand to the standard LCMV solver is
conj(a(u)), i.e.  exp(−j k r_n · u).  With R = σ² I (diagonal loading):

    w = R^{-1} C* ( C*^H R^{-1} C* )^{-1} f

The resulting weights satisfy  w^T a(u_s) = 1  and  w^T a(u_j) = 0,
which means the nulls show up correctly in the array-factor pattern.

Phase convention
----------------
Same as ``steering.py``:  φ_n = −k (r_n · u),  k = 2π / λ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from array_engine.steering import azel_to_unit_vector

_C0 = 299_792_458.0


def _steering_vector(
    positions: np.ndarray, freq_hz: float, az_deg: float, el_deg: float
) -> np.ndarray:
    """Conventional steering vector a(u) with shape (N,).

    a_n = exp(j k r_n · u)
    """
    lam = _C0 / freq_hz
    k = 2.0 * np.pi / lam
    u = azel_to_unit_vector(az_deg, el_deg)
    sv: np.ndarray = np.exp(1j * k * (positions @ u))
    return sv


@dataclass
class LCMVResult:
    """Result container for LCMV weight computation."""

    weights: np.ndarray
    phases_rad: np.ndarray
    constraint_residuals: np.ndarray


def compute_lcmv_weights(
    positions: np.ndarray,
    freq_hz: float,
    steer_az_deg: float,
    steer_el_deg: float,
    jammer_azels: list[tuple[float, float]],
    diag_load: float = 1e-6,
) -> LCMVResult:
    """Compute LCMV beamforming weights with null constraints.

    Parameters
    ----------
    positions : (N, 3)
        Element positions in metres (array frame).
    freq_hz : float
        Carrier frequency in Hz.
    steer_az_deg, steer_el_deg : float
        Desired look direction in degrees.
    jammer_azels : list of (az_deg, el_deg) tuples
        Directions to null.
    diag_load : float
        Diagonal loading factor for numerical stability.

    Returns
    -------
    LCMVResult with weights (N,), phases_rad (N,), and constraint_residuals.
    """
    n_elements = len(positions)
    n_constraints = 1 + len(jammer_azels)

    # Build constraint matrix using conj(a(u)) so that the LCMV solution
    # satisfies  w^T a(u) = f_i  (matching our array-factor convention).
    constraint_matrix = np.zeros((n_elements, n_constraints), dtype=complex)
    constraint_matrix[:, 0] = np.conj(
        _steering_vector(positions, freq_hz, steer_az_deg, steer_el_deg)
    )
    for i, (jaz, jel) in enumerate(jammer_azels):
        constraint_matrix[:, i + 1] = np.conj(_steering_vector(positions, freq_hz, jaz, jel))

    # Desired response: unity toward steer, zero toward jammers
    f = np.zeros(n_constraints, dtype=complex)
    f[0] = 1.0

    # Covariance with diagonal loading: R = diag_load * I
    r_inv = np.eye(n_elements, dtype=complex) / diag_load

    # LCMV: w = R^{-1} C* (C*^H R^{-1} C*)^{-1} f
    r_inv_c = r_inv @ constraint_matrix
    gram = constraint_matrix.conj().T @ r_inv_c
    gram_inv = np.linalg.inv(gram)
    weights: np.ndarray = r_inv_c @ gram_inv @ f

    phases = np.angle(weights)

    # Verify constraints: w^T a(u_i) should equal f_i
    # a(u_i) = conj(constraint_matrix[:, i]), so w^T a = w^T conj(C) = (C^H w)^*
    a_matrix = np.conj(constraint_matrix)
    residuals: np.ndarray = a_matrix.T @ weights - f

    return LCMVResult(
        weights=weights,
        phases_rad=phases,
        constraint_residuals=residuals,
    )
