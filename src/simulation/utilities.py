"""
Utility Functions.

Implements Q function for consumer utility computation.
"""
import numpy as np


def Q(theta: np.ndarray, p: float, gamma: float) -> np.ndarray:
    """
    Compute utility Q(θ, p, γ).

    Args:
        theta: Preference values (scalar, array, or (N,2) array)
        p: Market share
        gamma: Individual taste weight

    Returns:
        Utility values in [0, 1]

    Equation:
        Eq 22: Q(θ, p, γ) = γ·θ + (1-γ)·p

    Properties:
        - 0 ≤ Q ≤ 1 (clipped)
        - γ=1 → Q=θ (pure individual preference)
        - γ=0 → Q=p (pure conformity)
    """
    utility = gamma * theta + (1 - gamma) * p
    # Clip to [0, 1] to handle numerical precision
    return np.clip(utility, 0.0, 1.0)
