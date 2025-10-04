"""
Unit tests for Potential function.

Verifies V(p, γ) properties and equilibrium stability analysis.
"""
import pytest
import numpy as np
from src.analysis.potential import (
    potential,
    potential_derivative,
    potential_second_derivative,
    is_stable_equilibrium,
    compute_potential_landscape,
)
from src.distributions.constants import GAMMA_STAR


class TestPotentialFunction:
    """Test potential function V(p, γ) and derivatives."""

    def test_symmetric_point_is_extremum(self):
        """V(0.5, γ) should be extremum (dV/dp = 0) for all γ."""
        gamma_values = [0.3, 0.5, 0.7, 0.9]

        for gamma in gamma_values:
            dV = potential_derivative(0.5, gamma)
            assert abs(dV) < 0.1, (
                f"dV/dp at p=0.5 should be ≈0 for all γ. "
                f"Got dV/dp = {dV:.4f} for γ={gamma}"
            )

    def test_symmetry_around_half(self):
        """V(p, γ) should be symmetric around p = 0.5."""
        gamma = 0.7
        p_below = 0.3
        p_above = 0.7

        V_below = potential(p_below, gamma)
        V_above = potential(p_above, gamma)

        # Should be approximately equal (symmetric)
        assert abs(V_below - V_above) < 0.2, (
            f"Potential should be symmetric around 0.5. "
            f"V(0.3) = {V_below:.4f}, V(0.7) = {V_above:.4f}"
        )

    def test_stable_equilibrium_above_gamma_star(self):
        """For γ > γ*, p=0.5 should be stable (d²V/dp² > 0)."""
        gamma = 0.80  # > γ*

        d2V = potential_second_derivative(0.5, gamma)

        assert d2V > 0, (
            f"For γ > γ*, symmetric equilibrium should be stable. "
            f"Got d²V/dp² = {d2V:.4f} (should be > 0)"
        )

    def test_unstable_equilibrium_below_gamma_star(self):
        """For γ < γ*, p=0.5 should be unstable (d²V/dp² < 0)."""
        gamma = 0.60  # < γ*

        d2V = potential_second_derivative(0.5, gamma)

        assert d2V < 0, (
            f"For γ < γ*, symmetric equilibrium should be unstable. "
            f"Got d²V/dp² = {d2V:.4f} (should be < 0)"
        )

    def test_stability_check(self):
        """is_stable_equilibrium should correctly identify stability."""
        # Stable case: γ > γ*
        assert is_stable_equilibrium(0.5, gamma=0.80) is True, "0.5 should be stable for γ > γ*"

        # Unstable case: γ < γ*
        assert (
            is_stable_equilibrium(0.5, gamma=0.60) is False
        ), "0.5 should be unstable for γ < γ*"

    def test_potential_landscape_shape(self):
        """Potential landscape should have correct shape."""
        gamma = 0.80  # > γ* (single minimum)

        p_values, V_values = compute_potential_landscape(gamma, n_points=50)

        # Should have 50 points
        assert len(p_values) == 50
        assert len(V_values) == 50

        # Find minimum
        min_idx = np.argmin(V_values)
        p_min = p_values[min_idx]

        # Minimum should be near 0.5 for γ > γ*
        assert abs(p_min - 0.5) < 0.15, (
            f"For γ > γ*, potential minimum should be near 0.5. " f"Got minimum at p = {p_min:.4f}"
        )

    def test_derivative_consistency(self):
        """dV/dp computed analytically should match numerical derivative of V."""
        gamma = 0.7
        p = 0.6
        h = 1e-6

        # Analytical derivative
        dV_analytical = potential_derivative(p, gamma)

        # Numerical derivative
        V_plus = potential(p + h, gamma)
        V_minus = potential(p - h, gamma)
        dV_numerical = (V_plus - V_minus) / (2 * h)

        # Should match closely
        assert abs(dV_analytical - dV_numerical) < 0.1, (
            f"Analytical and numerical derivatives should match. "
            f"Analytical: {dV_analytical:.4f}, Numerical: {dV_numerical:.4f}"
        )

    def test_equilibrium_at_zero_derivative(self):
        """Equilibria should occur where dV/dp = 0."""
        # We know p=0.5 is always an equilibrium
        gamma = 0.75

        dV = potential_derivative(0.5, gamma)

        assert abs(dV) < 0.05, (
            f"Derivative should be zero at equilibrium. " f"Got dV/dp = {dV:.4f} at p=0.5"
        )
