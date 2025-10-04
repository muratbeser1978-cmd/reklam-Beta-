"""
Potential Function for Equilibrium Stability Analysis.

Implements the potential function V(p, γ) and its derivatives.
"""
import numpy as np
from scipy.integrate import quad
from src.distributions.beta_diff import F_Z


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
        return 0.0
    return ((1 - gamma) / gamma) * (1 - 2 * p)


def potential(p: float, gamma: float) -> float:
    """
    Compute potential function V(p, γ).

    Args:
        p: Market share
        gamma: Individual taste weight

    Returns:
        V(p, γ): Potential value

    Equation:
        V(p, γ) = ∫₀ᵖ [x - BR(x, γ)] dx

        Note: This is the negative of Eq 36 in the original documentation.
        The sign is chosen such that:
        - dV/dp = p - BR(p)
        - d²V/dp² = 1 - BR'(p)
        - Stable equilibria are local minima (d²V/dp² > 0)

        Where BR(x, γ) = 1 - F_Z(τ(x))

    Properties:
        - Equilibria occur where dV/dp = 0 → p = BR(p)
        - Stable equilibria are local minima (d²V/dp² > 0 → BR'(p) < 1)
        - V is symmetric around p = 0.5
        - V(0.5, γ) is an extremum for all γ
    """
    # V(p) = ∫₀ᵖ [x - BR(x)] dx
    # Compute via numerical integration

    def integrand(x):
        tau_x = tau(x, gamma)
        BR_x = 1.0 - F_Z(tau_x)
        return x - BR_x

    result, _ = quad(integrand, 0, p, epsabs=1e-10, epsrel=1e-10)
    return result


def potential_derivative(p: float, gamma: float) -> float:
    """
    Compute potential derivative dV/dp.

    Args:
        p: Market share
        gamma: Individual taste weight

    Returns:
        dV/dp at (p, γ)

    Equation:
        dV/dp = p - BR(p, γ)

        Note: This is consistent with V(p) = ∫[x - BR(x)]dx

    Properties:
        - dV/dp = 0 at equilibria (p = BR(p))
        - Sign of dV/dp indicates gradient direction
    """
    tau_p = tau(p, gamma)
    BR_p = 1.0 - F_Z(tau_p)
    return p - BR_p


def potential_second_derivative(p: float, gamma: float, h: float = 1e-6) -> float:
    """
    Compute second derivative d²V/dp².

    Args:
        p: Market share
        gamma: Individual taste weight
        h: Step size for numerical derivative

    Returns:
        d²V/dp² at (p, γ)

    Equation:
        Eq 40: d²V/dp² = 1 - BR'(p, γ)

        Derivation:
            V(p) = -∫[BR(s) - s]ds
            dV/dp = -(BR(p) - p)
            d²V/dp² = -(BR'(p) - 1) = 1 - BR'(p)

    Stability criterion:
        - d²V/dp² > 0 → stable equilibrium (local minimum) → BR'(p) < 1
        - d²V/dp² < 0 → unstable equilibrium (local maximum) → BR'(p) > 1
        - d²V/dp² = 0 → neutral stability → BR'(p) = 1
    """
    from src.analysis.best_response import br_derivative

    BR_prime = br_derivative(p, gamma)
    return 1.0 - BR_prime


def is_stable_equilibrium(p: float, gamma: float) -> bool:
    """
    Check if p is a stable equilibrium for given γ.

    Args:
        p: Market share (candidate equilibrium)
        gamma: Individual taste weight

    Returns:
        True if stable (d²V/dp² > 0), False otherwise

    Criteria:
        1. Must be equilibrium: |dV/dp| ≈ 0
        2. Must be stable: d²V/dp² > 0
    """
    # Check if equilibrium
    dV = potential_derivative(p, gamma)
    if abs(dV) > 1e-3:
        return False  # Not an equilibrium

    # Check stability
    d2V = potential_second_derivative(p, gamma)
    return d2V > 0


def compute_potential_landscape(
    gamma: float, p_min: float = 0.0, p_max: float = 1.0, n_points: int = 100
) -> tuple:
    """
    Compute potential landscape V(p) for visualization.

    Args:
        gamma: Individual taste weight
        p_min: Minimum p value
        p_max: Maximum p value
        n_points: Number of points

    Returns:
        (p_values, V_values): Arrays for plotting

    Use Case:
        Visualize potential landscape to understand equilibrium stability
        and basins of attraction.
    """
    p_values = np.linspace(p_min, p_max, n_points)
    V_values = np.array([potential(p, gamma) for p in p_values])

    return p_values, V_values
