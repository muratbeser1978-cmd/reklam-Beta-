"""
Best Response Function.

Implements BR(p, γ) and its derivative for equilibrium analysis.
"""
import numpy as np
from src.distributions.beta_diff import F_Z, f_Z


def tau(p: float, gamma: float) -> float:
    """
    Compute threshold τ(p, γ).

    Args:
        p: Market share
        gamma: Individual taste weight

    Returns:
        τ(p) = [(1-γ)/γ] · (1-2p)  [Eq 32]
    """
    if gamma == 0:
        # Limit case: pure conformity, no threshold
        return 0.0
    return ((1 - gamma) / gamma) * (1 - 2 * p)


def best_response(p, gamma: float):
    """
    Compute Best Response BR(p, γ).

    Args:
        p: Market share (scalar or array)
        gamma: Individual taste weight

    Returns:
        BR(p, γ) value(s)

    Equation:
        Eq 33: BR(p, γ) = 1 - F_Z(τ(p))
        where τ(p) = [(1-γ)/γ] · (1-2p)  [Eq 32]

    Properties:
        - BR(0.5, γ) = 0.5 for all γ (symmetric equilibrium exists)
        - BR is monotonic increasing in p
        - 0 ≤ BR(p, γ) ≤ 1
    """
    if np.isscalar(p):
        tau_val = tau(p, gamma)
        br_val = 1.0 - F_Z(tau_val)
        # Clip to [0, 1] to handle numerical precision
        return np.clip(br_val, 0.0, 1.0)
    else:
        # Vectorized
        p_array = np.asarray(p)
        tau_vals = tau(p_array, gamma)
        br_vals = 1.0 - F_Z(tau_vals)
        # Clip to [0, 1] to handle numerical precision
        return np.clip(br_vals, 0.0, 1.0)


def br_derivative(p: float, gamma: float) -> float:
    """
    Compute BR derivative BR'(p).

    Args:
        p: Market share
        gamma: Individual taste weight

    Returns:
        dBR/dp at (p, γ)

    Equation:
        Eq 34: BR'(p) = f_Z(τ(p)) · 2(1-γ)/γ

    Critical value:
        Eq 35: BR'(1/2) = f_Z(0) · 2(1-γ)/γ
    """
    if gamma == 0:
        return 0.0

    tau_val = tau(p, gamma)
    f_Z_val = f_Z(tau_val)
    return f_Z_val * 2 * (1 - gamma) / gamma
