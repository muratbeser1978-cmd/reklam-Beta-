"""
Unit tests for analytical constants.

Verifies that mathematical constants match exact values from specification.
"""
import pytest
from src.distributions.constants import F_Z_AT_ZERO, GAMMA_STAR, HETEROGENEITY


class TestAnalyticalConstants:
    """Test exact values of analytical constants."""

    def test_f_z_at_zero(self):
        """Test F_Z_AT_ZERO == 6/5 = 1.2 (Eq 5)."""
        expected = 6.0 / 5.0
        assert F_Z_AT_ZERO == expected, f"F_Z_AT_ZERO = {F_Z_AT_ZERO}, expected {expected}"
        assert F_Z_AT_ZERO == 1.2, "F_Z_AT_ZERO should equal 1.2"

    def test_gamma_star(self):
        """Test GAMMA_STAR == 12/17 ≈ 0.7058823529411765 (Eq 8)."""
        expected = 12.0 / 17.0
        assert (
            abs(GAMMA_STAR - expected) < 1e-15
        ), f"GAMMA_STAR = {GAMMA_STAR}, expected {expected}"
        # Check approximate value for documentation
        assert abs(GAMMA_STAR - 0.7058823529411765) < 1e-15

    def test_heterogeneity(self):
        """Test HETEROGENEITY == 23/60 ≈ 0.383333... (Eq 6)."""
        expected = 23.0 / 60.0
        assert (
            abs(HETEROGENEITY - expected) < 1e-10
        ), f"HETEROGENEITY = {HETEROGENEITY}, expected {expected}"
        # Check approximate value
        assert abs(HETEROGENEITY - 0.38333333) < 1e-7
