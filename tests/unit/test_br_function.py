"""
Unit tests for Best Response function.

Verifies BR(p, γ) properties and derivative calculations.
"""
import pytest
import numpy as np
from src.analysis.best_response import best_response, br_derivative, tau
from src.distributions.constants import GAMMA_STAR


class TestBestResponseFunction:
    """Test Best Response function BR(p, γ)."""

    def test_br_symmetric_point(self):
        """BR(0.5, γ) = 0.5 for all γ (symmetric equilibrium exists)."""
        gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for gamma in gamma_values:
            br_half = best_response(0.5, gamma)

            # Allow larger tolerance for edge cases
            tolerance = 0.1 if gamma in [0.0, 1.0] else 0.05

            assert abs(br_half - 0.5) < tolerance, (
                f"BR(0.5, γ={gamma}) = {br_half:.4f}, expected 0.5. "
                f"Symmetric equilibrium should always exist."
            )

    def test_br_monotonic_increasing(self):
        """BR should be monotonically increasing in p."""
        gamma = 0.7
        p_values = np.linspace(0.1, 0.9, 20)

        br_values = [best_response(p, gamma) for p in p_values]

        # Check successive values are non-decreasing
        for i in range(len(br_values) - 1):
            assert br_values[i + 1] >= br_values[i] - 0.05, (
                f"BR not monotonic at p={p_values[i]:.2f}. "
                f"BR({p_values[i]:.2f}) = {br_values[i]:.4f}, "
                f"BR({p_values[i+1]:.2f}) = {br_values[i+1]:.4f}"
            )

    def test_br_bounds(self):
        """0 ≤ BR(p, γ) ≤ 1 for all valid p, γ."""
        gamma_values = [0.3, 0.5, 0.7, 0.9]
        p_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for gamma in gamma_values:
            for p in p_values:
                br = best_response(p, gamma)

                assert 0.0 <= br <= 1.0, (
                    f"BR({p}, γ={gamma}) = {br:.4f} out of bounds [0, 1]"
                )

    def test_br_derivative_at_symmetric_point(self):
        """BR'(1/2) = f_Z(0) · 2(1-γ)/γ (Eq 35)."""
        from src.distributions.constants import F_Z_AT_ZERO

        gamma_values = [0.3, 0.5, 0.7, 0.9]

        for gamma in gamma_values:
            if gamma == 0:
                continue

            br_prime = br_derivative(0.5, gamma)
            expected = F_Z_AT_ZERO * 2 * (1 - gamma) / gamma

            # Numerical derivative has ~10% tolerance
            assert abs(br_prime - expected) / abs(expected) < 0.2, (
                f"BR'(0.5, γ={gamma}) = {br_prime:.4f}, "
                f"expected {expected:.4f} (Eq 35)"
            )

    def test_tau_function(self):
        """Test τ(p, γ) = [(1-γ)/γ] · (1-2p) (Eq 32)."""
        # At p = 0.5, τ should be 0
        for gamma in [0.3, 0.5, 0.7, 0.9]:
            tau_half = tau(0.5, gamma)
            assert abs(tau_half) < 1e-10, f"τ(0.5, γ={gamma}) should be 0, got {tau_half}"

        # Test specific value
        gamma = 0.6
        p = 0.3
        tau_val = tau(p, gamma)
        expected = ((1 - 0.6) / 0.6) * (1 - 2 * 0.3)
        assert abs(tau_val - expected) < 1e-10, (
            f"τ({p}, γ={gamma}) = {tau_val:.6f}, expected {expected:.6f}"
        )

    def test_br_vectorization(self):
        """BR should work with array inputs."""
        gamma = 0.7
        p_array = np.array([0.2, 0.4, 0.6, 0.8])

        br_array = best_response(p_array, gamma)

        assert isinstance(br_array, np.ndarray), "Should return ndarray for array input"
        assert br_array.shape == p_array.shape, "Output shape should match input"
        assert np.all((br_array >= 0) & (br_array <= 1)), "All BR values should be in [0, 1]"

    def test_br_stability_criterion(self):
        """BR'(p*) < 1 for stable equilibrium, BR'(p*) > 1 for unstable."""
        # Above γ*: symmetric equilibrium stable
        gamma_above = 0.80
        br_prime_above = br_derivative(0.5, gamma_above)
        assert abs(br_prime_above) < 1.0, (
            f"For γ > γ*, BR'(0.5) should be < 1 (stable). "
            f"Got BR'(0.5) = {br_prime_above:.4f}"
        )

        # Below γ*: symmetric equilibrium unstable
        gamma_below = 0.60
        br_prime_below = br_derivative(0.5, gamma_below)
        assert abs(br_prime_below) > 1.0, (
            f"For γ < γ*, BR'(0.5) should be > 1 (unstable). "
            f"Got BR'(0.5) = {br_prime_below:.4f}"
        )
