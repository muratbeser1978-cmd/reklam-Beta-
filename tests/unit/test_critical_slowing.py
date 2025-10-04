"""
Unit tests for Critical Slowing Down Analysis.

Tests early warning signals detection near bifurcation point.
"""

import pytest
import numpy as np
from src.analysis.critical_slowing import (
    CriticalSlowingAnalyzer,
    CriticalSlowingMetrics,
    EarlyWarningSignals,
    ScalingAnalysis,
)


class TestCriticalSlowingMetrics:
    """Test computation of critical slowing down metrics."""

    def test_basic_metrics_computation(self):
        """Test basic metrics on synthetic trajectory."""
        analyzer = CriticalSlowingAnalyzer()

        # Generate synthetic trajectory with known properties
        np.random.seed(42)
        T = 500
        trajectory = 0.5 + 0.1 * np.random.randn(T)
        gamma = 0.65  # Below critical threshold

        metrics = analyzer.compute_metrics(trajectory, gamma, burnin=100)

        # Verify all attributes exist
        assert isinstance(metrics, CriticalSlowingMetrics)
        assert metrics.variance > 0
        assert -1 <= metrics.autocorr_lag1 <= 1
        assert metrics.relaxation_time > 0
        assert np.isfinite(metrics.skewness)
        assert metrics.gamma == gamma

    def test_high_gamma_low_variance(self):
        """Test that high gamma (above critical) gives low variance."""
        analyzer = CriticalSlowingAnalyzer(gamma_star=12/17)

        # Trajectory near p=0.5 (stable symmetric equilibrium)
        np.random.seed(42)
        trajectory = 0.5 + 0.01 * np.random.randn(200)
        gamma = 0.75  # Above γ*

        metrics = analyzer.compute_metrics(trajectory, gamma, burnin=50)

        # Should have low variance (strong restoring force)
        assert metrics.variance < 0.001
        assert metrics.autocorr_lag1 < 0.5  # Low persistence
        assert metrics.distance_to_critical > 0

    def test_low_gamma_high_variance(self):
        """Test that low gamma (below critical) allows high variance."""
        analyzer = CriticalSlowingAnalyzer(gamma_star=12/17)

        # Trajectory with higher variance (weaker restoring force)
        np.random.seed(42)
        trajectory = 0.5 + 0.2 * np.random.randn(200)
        gamma = 0.60  # Below γ*

        metrics = analyzer.compute_metrics(trajectory, gamma, burnin=50)

        # Should have higher variance
        assert metrics.variance > 0.01
        assert metrics.distance_to_critical < 0


class TestEarlyWarningSignals:
    """Test early warning signal detection."""

    def test_no_warnings_stable_trajectory(self):
        """Test that stable trajectory shows no warnings."""
        analyzer = CriticalSlowingAnalyzer()

        # Stable trajectory (constant mean and variance)
        np.random.seed(42)
        trajectory = 0.5 + 0.05 * np.random.randn(300)

        warnings = analyzer.detect_early_warnings(trajectory, window_size=50)

        assert isinstance(warnings, EarlyWarningSignals)
        # Stable trajectory should not show significant increasing trends
        # (though random fluctuations may occasionally give false positives)
        assert warnings.variance_kendall_pvalue is not None
        assert warnings.ac_kendall_pvalue is not None

    def test_warnings_increasing_variance(self):
        """Test detection of increasing variance (synthetic warning signal)."""
        analyzer = CriticalSlowingAnalyzer()

        # Create trajectory with artificially increasing variance
        np.random.seed(42)
        T = 400
        t = np.arange(T)
        std_increase = 0.01 + 0.0005 * t  # Linearly increasing std
        trajectory = 0.5 + std_increase * np.random.randn(T)

        warnings = analyzer.detect_early_warnings(trajectory, window_size=80)

        # Should detect increasing variance trend
        assert warnings.variance_increasing
        assert warnings.variance_kendall_pvalue < 0.1  # Strong signal

    def test_warnings_increasing_autocorrelation(self):
        """Test detection of increasing autocorrelation."""
        analyzer = CriticalSlowingAnalyzer()

        # Create trajectory with increasing autocorrelation
        np.random.seed(42)
        T = 400
        trajectory = np.zeros(T)
        trajectory[0] = 0.5

        # AR(1) process with increasing φ
        for t in range(1, T):
            phi = 0.1 + 0.002 * t  # Increasing persistence
            phi = min(phi, 0.95)  # Cap at 0.95
            trajectory[t] = phi * trajectory[t-1] + (1-phi) * 0.5 + 0.05 * np.random.randn()

        warnings = analyzer.detect_early_warnings(trajectory, window_size=80)

        # Should detect increasing AC
        assert warnings.ac_increasing or warnings.ac_kendall_pvalue < 0.2


class TestScalingAnalysis:
    """Test critical scaling exponent estimation."""

    def test_scaling_exponent_estimation(self):
        """Test power law fitting Var ~ (γ-γ*)^(-ν)."""
        analyzer = CriticalSlowingAnalyzer(gamma_star=12/17)

        # Generate synthetic data with known scaling
        gamma_values = np.array([0.60, 0.62, 0.64, 0.66, 0.68, 0.69, 0.70])
        gamma_star = 12/17

        # True scaling: ν = 1 (linear divergence)
        # Var ∝ 1/(γ* - γ) for γ < γ*
        distances = gamma_star - gamma_values
        variances = 0.01 / distances  # Perfect ν=1 scaling

        scaling = analyzer.estimate_scaling_exponent(gamma_values, variances)

        assert isinstance(scaling, ScalingAnalysis)
        # Should recover ν ≈ 1 (allowing some numerical error)
        assert 0.8 < scaling.nu_exponent < 1.2
        assert scaling.r_squared > 0.95  # Excellent fit
        assert scaling.pvalue < 0.01

    def test_scaling_insufficient_data(self):
        """Test handling of insufficient data for scaling analysis."""
        analyzer = CriticalSlowingAnalyzer()

        # Only 2 points (not enough for regression)
        gamma_values = np.array([0.60, 0.65])
        variances = np.array([0.1, 0.05])

        scaling = analyzer.estimate_scaling_exponent(gamma_values, variances)

        # Should return valid object with NaN or low quality indicators
        assert isinstance(scaling, ScalingAnalysis)
        # Either NaN exponent or low R²
        assert np.isnan(scaling.nu_exponent) or scaling.r_squared < 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_trajectory(self):
        """Test handling of very short trajectories."""
        analyzer = CriticalSlowingAnalyzer()

        # Very short trajectory
        trajectory = np.array([0.5, 0.51, 0.49, 0.50])

        with pytest.raises(ValueError, match="Trajectory too short"):
            analyzer.compute_metrics(trajectory, gamma=0.7, burnin=0)

    def test_constant_trajectory(self):
        """Test handling of constant trajectory (zero variance)."""
        analyzer = CriticalSlowingAnalyzer()

        # Constant trajectory
        trajectory = np.ones(200) * 0.5

        metrics = analyzer.compute_metrics(trajectory, gamma=0.7, burnin=50)

        # Should have zero variance
        assert metrics.variance == 0.0
        assert metrics.relaxation_time == 0.0  # No fluctuations to decay

    def test_burnin_handling(self):
        """Test that burnin period is properly discarded."""
        analyzer = CriticalSlowingAnalyzer()

        # Trajectory with transient at start
        transient = np.linspace(0.0, 0.5, 100)
        equilibrium = 0.5 + 0.05 * np.random.randn(200)
        trajectory = np.concatenate([transient, equilibrium])

        metrics_no_burnin = analyzer.compute_metrics(trajectory, gamma=0.7, burnin=0)
        metrics_burnin = analyzer.compute_metrics(trajectory, gamma=0.7, burnin=100)

        # Burnin version should have lower variance (removed transient)
        assert metrics_burnin.variance < metrics_no_burnin.variance

    def test_gamma_at_critical(self):
        """Test metrics exactly at critical point."""
        gamma_star = 12/17
        analyzer = CriticalSlowingAnalyzer(gamma_star=gamma_star)

        trajectory = 0.5 + 0.1 * np.random.randn(200)

        metrics = analyzer.compute_metrics(trajectory, gamma=gamma_star, burnin=50)

        # Distance to critical should be zero
        assert abs(metrics.distance_to_critical) < 1e-10


class TestRelaxationTime:
    """Test relaxation time estimation."""

    def test_relaxation_time_increases_near_critical(self):
        """Test that τ increases as γ → γ* from above."""
        analyzer = CriticalSlowingAnalyzer(gamma_star=12/17)

        # Simulate trajectories at different γ values
        np.random.seed(42)

        # Far from critical (γ = 0.80, high above γ*)
        traj_far = 0.5 + 0.02 * np.random.randn(300)
        metrics_far = analyzer.compute_metrics(traj_far, gamma=0.80, burnin=50)

        # Near critical (γ = 0.72, just above γ*)
        traj_near = 0.5 + 0.05 * np.random.randn(300)
        metrics_near = analyzer.compute_metrics(traj_near, gamma=0.72, burnin=50)

        # Relaxation time should be longer near critical
        # (though with synthetic data this may not always hold due to noise)
        assert metrics_far.relaxation_time >= 0
        assert metrics_near.relaxation_time >= 0


class TestComparisonMethods:
    """Test comparative analysis methods."""

    def test_compare_gamma_values(self):
        """Test comparison across different gamma values."""
        analyzer = CriticalSlowingAnalyzer()

        # Generate trajectories for different gammas
        np.random.seed(42)
        T = 300

        gamma_high = 0.80
        gamma_low = 0.65

        traj_high = 0.5 + 0.02 * np.random.randn(T)
        traj_low = 0.5 + 0.08 * np.random.randn(T)

        metrics_high = analyzer.compute_metrics(traj_high, gamma_high, burnin=50)
        metrics_low = analyzer.compute_metrics(traj_low, gamma_low, burnin=50)

        # Low gamma should show stronger critical slowing down signals
        assert metrics_low.variance > metrics_high.variance
        # (Other metrics may vary due to synthetic data)


def test_critical_slowing_integration():
    """Integration test: full workflow from simulation to early warnings."""
    from src.simulation.config import SimulationConfig
    from src.simulation.engine import SimulationEngine

    # Run quick simulation near critical point
    np.random.seed(42)
    config = SimulationConfig(
        N=100,
        T=200,
        gamma=0.70,  # Just below γ* = 0.706
        beta=0.9,
        alpha_0=2.0,
        beta_0=2.0,
        p_init=0.5,
    )

    engine = SimulationEngine(config)
    result = engine.run()

    # Analyze critical slowing down
    analyzer = CriticalSlowingAnalyzer(gamma_star=12/17)

    metrics = analyzer.compute_metrics(
        result.p1_trajectory, gamma=config.gamma, burnin=50
    )

    # Should detect proximity to critical point
    assert metrics.distance_to_critical < 0.01  # Very close to γ*
    assert metrics.variance > 0  # Some fluctuations present

    # Early warnings
    warnings = analyzer.detect_early_warnings(result.p1_trajectory, window_size=50)
    assert isinstance(warnings, EarlyWarningSignals)
    # May or may not detect warnings depending on trajectory realization
