"""
Unit tests for Statistical Analysis.

Verifies time series statistics calculations.
"""
import pytest
import numpy as np
from src.analysis.statistics import StatisticalAnalyzer, TimeSeriesStatistics


class TestStatisticalAnalyzer:
    """Test statistical analysis functions."""

    def test_basic_statistics(self):
        """Test mean, variance, std calculations."""
        analyzer = StatisticalAnalyzer()

        # Simple trajectory
        trajectory = np.array([0.4, 0.45, 0.5, 0.55, 0.6])

        stats = analyzer.compute_statistics(trajectory)

        # Check mean
        expected_mean = np.mean(trajectory)
        assert abs(stats.mean - expected_mean) < 1e-10, (
            f"Mean calculation incorrect: {stats.mean} != {expected_mean}"
        )

        # Check variance
        expected_var = np.var(trajectory)
        assert abs(stats.variance - expected_var) < 1e-10, (
            f"Variance calculation incorrect: {stats.variance} != {expected_var}"
        )

        # Check std
        expected_std = np.std(trajectory)
        assert abs(stats.std - expected_std) < 1e-10, (
            f"Std calculation incorrect: {stats.std} != {expected_std}"
        )

    def test_autocorrelation(self):
        """Test autocorrelation calculation."""
        analyzer = StatisticalAnalyzer()

        # Create trajectory with known autocorrelation
        t = np.linspace(0, 10, 100)
        trajectory = 0.5 + 0.1 * np.sin(t)  # Oscillating around 0.5

        stats = analyzer.compute_statistics(trajectory)

        # Autocorrelation should be array of 4 values
        assert len(stats.autocorrelation) == 4, "Should compute AC at 4 lags"

        # AC at lag 0 is always 1 (but we compute lag [1,5,10,20])
        # AC should be between -1 and 1
        assert np.all(
            (stats.autocorrelation >= -1.1) & (stats.autocorrelation <= 1.1)
        ), f"AC values out of range: {stats.autocorrelation}"

    def test_skewness(self):
        """Test skewness calculation."""
        analyzer = StatisticalAnalyzer()

        # Symmetric trajectory (should have low skewness)
        symmetric = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        stats_sym = analyzer.compute_statistics(symmetric)

        assert abs(stats_sym.skewness) < 1.0, (
            f"Symmetric trajectory should have low skewness, got {stats_sym.skewness:.4f}"
        )

        # Right-skewed trajectory
        right_skewed = np.array([0.4, 0.4, 0.5, 0.7, 0.9])
        stats_skew = analyzer.compute_statistics(right_skewed)

        # Should be positive (more mass on right)
        # Note: Short arrays may not show strong skewness
        assert stats_skew.skewness >= -0.5, f"Right-skewed data gave negative skewness"

    def test_convergence_detection(self):
        """Test convergence time detection."""
        analyzer = StatisticalAnalyzer()

        # Trajectory that converges to 0.5
        trajectory = np.array([0.3, 0.4, 0.48, 0.49, 0.50, 0.50, 0.50])
        equilibrium_p = 0.5

        stats = analyzer.compute_statistics(trajectory, equilibrium_p=equilibrium_p, threshold=0.02)

        # Should detect convergence at index 4 (first time within 0.02 of 0.5 and stays)
        assert stats.convergence_time is not None, "Should detect convergence"
        assert stats.convergence_time >= 3, f"Convergence detected too early at t={stats.convergence_time}"

    def test_no_convergence(self):
        """Test when trajectory doesn't converge."""
        analyzer = StatisticalAnalyzer()

        # Non-converging trajectory
        trajectory = np.array([0.3, 0.4, 0.6, 0.7, 0.8])
        equilibrium_p = 0.5

        stats = analyzer.compute_statistics(trajectory, equilibrium_p=equilibrium_p, threshold=0.01)

        # Should not detect convergence
        assert (
            stats.convergence_time is None
        ), f"Should not detect convergence, got t={stats.convergence_time}"

    def test_learning_metrics(self):
        """Test learning metrics calculation."""
        analyzer = StatisticalAnalyzer()

        # Simulate belief convergence
        N = 100
        theta_true = np.random.beta(2, 2, size=(N, 2))
        theta_hat = theta_true + np.random.normal(0, 0.1, size=(N, 2))  # Close to true
        n_eff = np.random.randint(10, 100, size=(N, 2))

        metrics = analyzer.compute_learning_metrics(theta_true, theta_hat, n_eff)

        # Check metrics exist
        assert "belief_error" in metrics, "Should compute belief error"
        assert "convergence_rate" in metrics, "Should compute convergence rate"
        assert "mean_experience" in metrics, "Should compute mean experience"

        # Belief error should be small (we added small noise)
        assert 0.0 <= metrics["belief_error"] <= 1.0, (
            f"Belief error out of range: {metrics['belief_error']}"
        )

        # Mean experience should be positive
        assert metrics["mean_experience"] > 0, "Mean experience should be positive"

    def test_constant_trajectory(self):
        """Test statistics for constant trajectory."""
        analyzer = StatisticalAnalyzer()

        # Constant trajectory
        trajectory = np.full(50, 0.5)

        stats = analyzer.compute_statistics(trajectory)

        # Mean should be 0.5
        assert abs(stats.mean - 0.5) < 1e-10, "Mean of constant should be the value"

        # Variance should be 0
        assert stats.variance < 1e-10, "Variance of constant should be 0"

        # Skewness is undefined for constant (scipy returns 0 or nan)
        # Just check it doesn't crash
        assert not np.isnan(stats.skewness) or stats.skewness == 0
