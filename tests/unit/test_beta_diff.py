"""
Unit tests for Beta distribution difference CDF and PDF.

Verifies F_Z and f_Z computations against analytical results and properties.
"""
import pytest
import numpy as np
from src.distributions.beta_diff import F_Z, f_Z
from src.distributions.constants import F_Z_AT_ZERO


class TestBetaDifference:
    """Test Beta distribution difference functions."""

    def test_f_Z_at_zero(self):
        """Test f_Z(0) == 6/5 = 1.2 (Eq 5, analytical result)."""
        computed = f_Z(0.0)
        expected = 6.0 / 5.0
        assert abs(computed - expected) < 1e-2, (
            f"f_Z(0) = {computed}, expected {expected}. "
            "Note: Numerical derivative has ~1e-2 tolerance"
        )
        # Verify against constant
        assert abs(computed - F_Z_AT_ZERO) < 1e-2

    def test_F_Z_bounds(self):
        """Test F_Z(-1) ≈ 0 and F_Z(1) ≈ 1 (CDF bounds)."""
        F_Z_min = F_Z(-1.0)
        F_Z_max = F_Z(1.0)

        assert F_Z_min < 0.01, f"F_Z(-1) = {F_Z_min}, should be near 0"
        assert F_Z_max > 0.99, f"F_Z(1) = {F_Z_max}, should be near 1"

    def test_F_Z_symmetry(self):
        """Test F_Z(0) ≈ 0.5 (symmetry of Beta(2,2))."""
        F_Z_zero = F_Z(0.0)
        assert abs(F_Z_zero - 0.5) < 0.01, (
            f"F_Z(0) = {F_Z_zero}, expected ≈ 0.5. "
            "Beta(2,2) difference is symmetric"
        )

    def test_F_Z_monotonic(self):
        """Property test: F_Z is monotonically increasing."""
        z_values = np.linspace(-1, 1, 50)
        F_values = F_Z(z_values)

        # Check all successive differences are non-negative
        diffs = np.diff(F_values)
        assert np.all(diffs >= -1e-10), (
            "F_Z is not monotonically increasing. "
            f"Found negative differences: {diffs[diffs < 0]}"
        )

    def test_F_Z_range(self):
        """Test F_Z values are in [0, 1] for all z in [-1, 1]."""
        z_values = np.linspace(-1, 1, 100)
        F_values = F_Z(z_values)

        assert np.all(F_values >= 0), f"F_Z has negative values: min = {F_values.min()}"
        assert np.all(F_values <= 1), f"F_Z exceeds 1: max = {F_values.max()}"

    def test_F_Z_vectorization(self):
        """Test F_Z works with array inputs."""
        z_array = np.array([-0.5, 0.0, 0.5])
        F_array = F_Z(z_array)

        assert F_array.shape == z_array.shape, "F_Z output shape doesn't match input"
        assert isinstance(F_array, np.ndarray), "F_Z should return ndarray for array input"

    def test_f_Z_positive(self):
        """Test f_Z(z) ≥ 0 for all z (PDF is non-negative)."""
        z_values = np.linspace(-0.9, 0.9, 20)
        for z in z_values:
            f_z_val = f_Z(z)
            assert f_z_val >= -1e-6, (
                f"f_Z({z}) = {f_z_val} is negative. "
                "PDF must be non-negative (allowing small numerical error)"
            )
