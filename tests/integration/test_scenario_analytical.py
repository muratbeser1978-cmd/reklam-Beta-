"""
Integration Test: Analytical Constants (Scenario 5 from quickstart.md).

Verifies exact numerical constants match specification.
"""
import pytest
import numpy as np
from src.distributions.constants import F_Z_AT_ZERO, GAMMA_STAR, HETEROGENEITY
from src.distributions.beta_diff import f_Z, F_Z
from src.analysis.best_response import best_response


class TestAnalyticalConstants:
    """Test analytical constants and properties."""

    def test_f_Z_at_zero(self):
        """
        Scenario 5a: f_Z(0) = 6/5 = 1.2.

        Equations Tested:
            - Eq 5: f_Z(0) = 6/5 (Beta(2,2) difference PDF at zero)
        """
        computed_f_Z_0 = f_Z(0.0)
        expected_f_Z_0 = 6.0 / 5.0

        # Numerical derivative has ~1e-2 tolerance
        assert abs(computed_f_Z_0 - expected_f_Z_0) < 0.05, (
            f"FAIL: f_Z(0) = {computed_f_Z_0:.4f}, expected {expected_f_Z_0:.4f}. "
            f"Note: Numerical derivative tolerance is ~0.05"
        )

        # Also check constant
        assert abs(F_Z_AT_ZERO - 1.2) < 1e-10, "Constant F_Z_AT_ZERO != 1.2"

    def test_gamma_star(self):
        """
        Scenario 5b: γ* = 12/17.

        Equations Tested:
            - Eq 8: γ* = 12/17 ≈ 0.7058823529411765 (critical bifurcation threshold)
        """
        expected_gamma_star = 12.0 / 17.0

        assert abs(GAMMA_STAR - expected_gamma_star) < 1e-15, (
            f"FAIL: γ* = {GAMMA_STAR:.16f}, " f"expected {expected_gamma_star:.16f}"
        )

    def test_heterogeneity(self):
        """
        Scenario 5c: H = 23/60.

        Equations Tested:
            - Eq 6: H = 23/60 ≈ 0.383333... (heterogeneity parameter)
        """
        expected_H = 23.0 / 60.0

        assert abs(HETEROGENEITY - expected_H) < 1e-10, (
            f"FAIL: H = {HETEROGENEITY:.10f}, " f"expected {expected_H:.10f}"
        )

    def test_BR_symmetric_point(self):
        """
        Scenario 5d: BR(0.5, γ) = 0.5 for all γ.

        Equations Tested:
            - Eq 33: BR(p, γ) = 1 - F_Z(τ(p))
            - Eq 41: Symmetric equilibrium always exists
        """
        gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for gamma in gamma_values:
            br_half = best_response(0.5, gamma)

            # Allow larger tolerance for edge cases (gamma near 0 or 1)
            tolerance = 0.1 if gamma in [0.0, 1.0] else 0.05

            assert abs(br_half - 0.5) < tolerance, (
                f"FAIL: BR(0.5, γ={gamma}) = {br_half:.4f}, expected 0.5. "
                f"Symmetric equilibrium should always exist."
            )

    def test_F_Z_properties(self):
        """Additional F_Z properties."""
        # F_Z(-1) ≈ 0
        F_Z_min = F_Z(-1.0)
        assert F_Z_min < 0.01, f"F_Z(-1) = {F_Z_min}, should be near 0"

        # F_Z(1) ≈ 1
        F_Z_max = F_Z(1.0)
        assert F_Z_max > 0.99, f"F_Z(1) = {F_Z_max}, should be near 1"

        # F_Z(0) ≈ 0.5 (symmetry)
        F_Z_zero = F_Z(0.0)
        assert abs(F_Z_zero - 0.5) < 0.01, f"F_Z(0) = {F_Z_zero}, should be near 0.5"

        # F_Z is monotonic
        z_values = np.linspace(-1, 1, 50)
        F_values = F_Z(z_values)
        diffs = np.diff(F_values)

        assert np.all(diffs >= -1e-10), (
            "F_Z is not monotonically increasing. " f"Found negative differences: {diffs[diffs < 0]}"
        )
