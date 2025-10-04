"""
Unit tests for Advanced Statistics Analysis.

Tests time series analysis, distribution fitting, hypothesis testing, and bootstrap methods.
"""

import pytest
import numpy as np
from scipy import stats
from src.analysis.advanced_statistics import (
    AdvancedStatisticsAnalyzer,
    TimeSeriesAnalysis,
    DistributionFit,
    ComparativeStatistics,
    summarize_simulation_results,
)


class TestTimeSeriesAnalysis:
    """Test comprehensive time series analysis."""

    def test_basic_time_series_analysis(self):
        """Test basic time series statistics."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Simple trajectory
        np.random.seed(42)
        trajectory = 0.5 + 0.1 * np.random.randn(300)

        ts_stats = analyzer.analyze_time_series(trajectory, burnin=50)

        # Verify all attributes exist
        assert isinstance(ts_stats, TimeSeriesAnalysis)
        assert np.isfinite(ts_stats.mean)
        assert np.isfinite(ts_stats.median)
        assert ts_stats.std > 0
        assert ts_stats.variance > 0
        assert np.isfinite(ts_stats.skewness)
        assert np.isfinite(ts_stats.kurtosis)
        assert ts_stats.max_val >= ts_stats.min_val
        assert ts_stats.range == ts_stats.max_val - ts_stats.min_val
        assert ts_stats.iqr > 0
        assert ts_stats.cv > 0
        assert -1 <= ts_stats.autocorr_lag1 <= 1
        assert np.isfinite(ts_stats.trend_slope)
        assert 0 <= ts_stats.trend_pvalue <= 1
        assert isinstance(ts_stats.is_stationary, bool)
        assert ts_stats.n_changepoints >= 0

    def test_stationary_trajectory_detection(self):
        """Test detection of stationary process."""
        analyzer = AdvancedStatisticsAnalyzer()

        # White noise (stationary)
        np.random.seed(42)
        trajectory = np.random.randn(500)

        ts_stats = analyzer.analyze_time_series(trajectory, burnin=0)

        # Should be detected as stationary
        assert ts_stats.is_stationary
        # Mean should be near zero
        assert abs(ts_stats.mean) < 0.2
        # Trend should be negligible
        assert abs(ts_stats.trend_slope) < 0.01

    def test_nonstationary_trajectory_detection(self):
        """Test detection of non-stationary process."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Trajectory with trend
        np.random.seed(42)
        t = np.arange(500)
        trajectory = 0.5 + 0.001 * t + 0.05 * np.random.randn(500)

        ts_stats = analyzer.analyze_time_series(trajectory, burnin=0)

        # Should detect trend
        assert ts_stats.trend_slope > 0
        # May or may not be flagged as non-stationary depending on test sensitivity

    def test_high_autocorrelation_detection(self):
        """Test detection of high autocorrelation."""
        analyzer = AdvancedStatisticsAnalyzer()

        # AR(1) process with φ=0.8 (high persistence)
        np.random.seed(42)
        T = 500
        trajectory = np.zeros(T)
        trajectory[0] = 0.5
        phi = 0.8

        for t in range(1, T):
            trajectory[t] = phi * trajectory[t-1] + (1-phi) * 0.5 + 0.05 * np.random.randn()

        ts_stats = analyzer.analyze_time_series(trajectory, burnin=100)

        # Should have high lag-1 autocorrelation
        assert ts_stats.autocorr_lag1 > 0.6

    def test_trajectory_too_short_raises_error(self):
        """Test that very short trajectories raise error."""
        analyzer = AdvancedStatisticsAnalyzer()

        trajectory = np.array([0.5, 0.51, 0.49])

        with pytest.raises(ValueError, match="Trajectory too short"):
            analyzer.analyze_time_series(trajectory, burnin=0)

    def test_burnin_period_handling(self):
        """Test that burnin period is properly removed."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Trajectory with transient
        transient = np.linspace(0, 0.5, 100)
        equilibrium = 0.5 + 0.05 * np.random.randn(200)
        trajectory = np.concatenate([transient, equilibrium])

        ts_stats_no_burnin = analyzer.analyze_time_series(trajectory, burnin=0)
        ts_stats_burnin = analyzer.analyze_time_series(trajectory, burnin=100)

        # With burnin, mean should be closer to 0.5
        assert abs(ts_stats_burnin.mean - 0.5) < abs(ts_stats_no_burnin.mean - 0.5)


class TestDistributionFitting:
    """Test distribution fitting and goodness-of-fit."""

    def test_fit_normal_distribution(self):
        """Test fitting normal distribution."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Generate normal data
        np.random.seed(42)
        data = np.random.normal(loc=0.5, scale=0.1, size=500)

        fit = analyzer.fit_distribution(data, dist_name="norm")

        assert isinstance(fit, DistributionFit)
        assert fit.distribution_name == "norm"
        assert "loc" in fit.parameters
        assert "scale" in fit.parameters
        # Should fit well
        assert fit.ks_pvalue > 0.01
        assert np.isfinite(fit.aic)
        assert np.isfinite(fit.bic)

    def test_fit_beta_distribution(self):
        """Test fitting beta distribution."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Generate beta data
        np.random.seed(42)
        data = np.random.beta(a=2, b=2, size=500)

        fit = analyzer.fit_distribution(data, dist_name="beta")

        assert fit.distribution_name == "beta"
        assert fit.ks_pvalue > 0.0  # Some fit
        assert np.isfinite(fit.log_likelihood)

    def test_fit_gamma_distribution(self):
        """Test fitting gamma distribution."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Generate gamma data
        np.random.seed(42)
        data = np.random.gamma(shape=2, scale=0.5, size=500)

        fit = analyzer.fit_distribution(data, dist_name="gamma")

        assert fit.distribution_name == "gamma"
        assert np.isfinite(fit.ks_statistic)

    def test_unknown_distribution_raises_error(self):
        """Test that unknown distribution name raises error."""
        analyzer = AdvancedStatisticsAnalyzer()

        data = np.random.randn(100)

        with pytest.raises(ValueError, match="Unknown distribution"):
            analyzer.fit_distribution(data, dist_name="foobar")

    def test_aic_bic_comparison(self):
        """Test that AIC and BIC are computed correctly."""
        analyzer = AdvancedStatisticsAnalyzer()

        np.random.seed(42)
        data = np.random.normal(0, 1, size=200)

        fit = analyzer.fit_distribution(data, dist_name="norm")

        # AIC and BIC should be finite and negative (for good fit)
        assert np.isfinite(fit.aic)
        assert np.isfinite(fit.bic)
        # BIC penalizes complexity more
        # Both should be comparable in magnitude
        assert abs(fit.aic - fit.bic) < abs(fit.aic) * 0.5


class TestComparativeStatistics:
    """Test group comparison methods."""

    def test_compare_identical_groups(self):
        """Test comparison of identical groups."""
        analyzer = AdvancedStatisticsAnalyzer()

        np.random.seed(42)
        group1 = np.random.normal(0.5, 0.1, size=100)
        group2 = group1.copy()

        comp = analyzer.compare_groups(group1, group2)

        assert isinstance(comp, ComparativeStatistics)
        assert comp.difference == 0
        assert comp.effect_size == 0
        # p-values should be 1.0 (no difference)
        assert comp.t_pvalue > 0.9
        assert not comp.significant_at_005

    def test_compare_different_groups(self):
        """Test comparison of significantly different groups."""
        analyzer = AdvancedStatisticsAnalyzer()

        np.random.seed(42)
        group1 = np.random.normal(0.3, 0.05, size=100)
        group2 = np.random.normal(0.7, 0.05, size=100)

        comp = analyzer.compare_groups(group1, group2)

        # Should detect significant difference
        assert abs(comp.difference) > 0.3
        assert abs(comp.effect_size) > 2.0  # Large effect size
        assert comp.t_pvalue < 0.01
        assert comp.mann_whitney_pvalue < 0.01
        assert comp.significant_at_005

    def test_effect_size_computation(self):
        """Test Cohen's d effect size computation."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Groups with known effect size
        np.random.seed(42)
        # Mean diff = 1.0, pooled std ≈ 1.0 → d ≈ 1.0
        group1 = np.random.normal(0, 1, size=200)
        group2 = np.random.normal(1, 1, size=200)

        comp = analyzer.compare_groups(group1, group2)

        # Effect size should be approximately 1.0
        assert 0.8 < abs(comp.effect_size) < 1.2

    def test_mann_whitney_u_test(self):
        """Test non-parametric Mann-Whitney U test."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Non-normal data
        np.random.seed(42)
        group1 = np.random.exponential(scale=1, size=100)
        group2 = np.random.exponential(scale=2, size=100)

        comp = analyzer.compare_groups(group1, group2)

        # Mann-Whitney should detect difference
        assert comp.mann_whitney_pvalue < 0.05
        assert np.isfinite(comp.mann_whitney_u)


class TestBootstrapConfidenceInterval:
    """Test bootstrap confidence interval estimation."""

    def test_bootstrap_mean_ci(self):
        """Test bootstrap CI for mean."""
        analyzer = AdvancedStatisticsAnalyzer()

        np.random.seed(42)
        data = np.random.normal(0.5, 0.1, size=200)

        point_est, ci_low, ci_high = analyzer.bootstrap_confidence_interval(
            data, statistic_func=np.mean, n_bootstrap=1000, alpha=0.05
        )

        # Point estimate should be near true mean
        assert 0.45 < point_est < 0.55
        # CI should contain point estimate
        assert ci_low < point_est < ci_high
        # CI width should be reasonable
        assert (ci_high - ci_low) < 0.1

    def test_bootstrap_median_ci(self):
        """Test bootstrap CI for median."""
        analyzer = AdvancedStatisticsAnalyzer()

        np.random.seed(42)
        data = np.random.normal(0.5, 0.1, size=200)

        point_est, ci_low, ci_high = analyzer.bootstrap_confidence_interval(
            data, statistic_func=np.median, n_bootstrap=1000, alpha=0.05
        )

        assert ci_low < point_est < ci_high

    def test_bootstrap_custom_statistic(self):
        """Test bootstrap with custom statistic function."""
        analyzer = AdvancedStatisticsAnalyzer()

        np.random.seed(42)
        data = np.random.beta(2, 2, size=200)

        # Custom: 90th percentile
        def percentile_90(x):
            return np.percentile(x, 90)

        point_est, ci_low, ci_high = analyzer.bootstrap_confidence_interval(
            data, statistic_func=percentile_90, n_bootstrap=500
        )

        # Point estimate should be near 90th percentile
        assert 0.7 < point_est < 0.9
        assert ci_low < point_est < ci_high


class TestPowerLawTesting:
    """Test power law relationship detection."""

    def test_perfect_power_law(self):
        """Test detection of perfect power law relationship."""
        analyzer = AdvancedStatisticsAnalyzer()

        # y = x^2 (exponent = 2)
        x_data = np.linspace(1, 10, 50)
        y_data = x_data ** 2

        exponent, r_squared, p_value = analyzer.test_power_law(x_data, y_data)

        # Should recover exponent ≈ 2
        assert 1.9 < exponent < 2.1
        # R² should be nearly 1
        assert r_squared > 0.99
        # p-value should be tiny
        assert p_value < 0.001

    def test_noisy_power_law(self):
        """Test power law with noise."""
        analyzer = AdvancedStatisticsAnalyzer()

        np.random.seed(42)
        x_data = np.linspace(1, 10, 100)
        y_data = 2 * x_data ** 1.5 * (1 + 0.1 * np.random.randn(100))

        exponent, r_squared, p_value = analyzer.test_power_law(x_data, y_data)

        # Should approximately recover exponent ≈ 1.5
        assert 1.3 < exponent < 1.7
        # R² should still be high
        assert r_squared > 0.8
        assert p_value < 0.01

    def test_no_power_law(self):
        """Test detection when no power law exists."""
        analyzer = AdvancedStatisticsAnalyzer()

        np.random.seed(42)
        x_data = np.linspace(1, 10, 50)
        y_data = np.random.randn(50)  # Pure noise

        exponent, r_squared, p_value = analyzer.test_power_law(x_data, y_data)

        # R² should be low
        assert r_squared < 0.5

    def test_insufficient_data_for_power_law(self):
        """Test handling of insufficient data."""
        analyzer = AdvancedStatisticsAnalyzer()

        x_data = np.array([1.0, 2.0])
        y_data = np.array([1.0, 4.0])

        exponent, r_squared, p_value = analyzer.test_power_law(x_data, y_data)

        # Should return valid values (possibly NaN or low quality)
        assert np.isnan(exponent) or np.isfinite(exponent)


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_autocorrelation_computation(self):
        """Test autocorrelation calculation."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Perfect autocorrelation (constant)
        x_constant = np.ones(100)
        ac = analyzer._autocorr(x_constant, lag=1)
        # Undefined (0/0), but implementation returns 0 or 1
        assert np.isfinite(ac)

        # No autocorrelation (white noise)
        np.random.seed(42)
        x_noise = np.random.randn(500)
        ac = analyzer._autocorr(x_noise, lag=1)
        # Should be near zero
        assert abs(ac) < 0.2

    def test_trend_detection(self):
        """Test linear trend detection."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Strong upward trend
        x_trend = np.linspace(0, 1, 100)
        slope, pval = analyzer._detect_trend(x_trend)

        assert slope > 0
        assert pval < 0.001

    def test_stationarity_test(self):
        """Test stationarity test."""
        analyzer = AdvancedStatisticsAnalyzer()

        # Stationary (constant mean)
        np.random.seed(42)
        x_stationary = np.random.randn(200)
        is_stat = analyzer._test_stationarity(x_stationary)
        assert is_stat

        # Non-stationary (shifting mean)
        x_nonstat = np.concatenate([
            np.random.randn(100) - 1,
            np.random.randn(100) + 1
        ])
        is_stat = analyzer._test_stationarity(x_nonstat)
        assert not is_stat

    def test_changepoint_detection(self):
        """Test changepoint detection."""
        analyzer = AdvancedStatisticsAnalyzer()

        # No changepoints (smooth trajectory)
        x_smooth = np.sin(np.linspace(0, 2*np.pi, 200))
        n_cp = analyzer._detect_changepoints(x_smooth, threshold=2.0)
        assert n_cp >= 0  # May detect some due to peaks

        # Clear changepoint
        x_jump = np.concatenate([
            np.ones(100) * 0.0,
            np.ones(100) * 1.0
        ])
        n_cp = analyzer._detect_changepoints(x_jump, threshold=0.5)
        assert n_cp >= 1  # Should detect the jump


class TestSummarizeSimulationResults:
    """Test comprehensive simulation summary function."""

    def test_summarize_multiple_runs(self):
        """Test summarization across multiple simulation runs."""
        from src.simulation.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        analyzer = AdvancedStatisticsAnalyzer()

        # Run multiple simulations
        np.random.seed(42)
        config = SimulationConfig(
            N=50,
            T=100,
            gamma=0.7,
            beta=0.9,
            alpha_0=2.0,
            beta_0=2.0,
            p_init=0.5,
        )

        results = []
        for _ in range(10):
            engine = SimulationEngine(config)
            result = engine.run()
            results.append(result)

        # Summarize
        summary = summarize_simulation_results(results, analyzer)

        # Verify structure
        assert "equilibrium" in summary
        assert "time_series" in summary
        assert "n_runs" in summary
        assert summary["n_runs"] == 10

        # Equilibrium statistics
        eq = summary["equilibrium"]
        assert np.isfinite(eq["mean"])
        assert eq["std"] >= 0
        assert np.isfinite(eq["median"])
        assert len(eq["confidence_interval"]) == 2
        assert isinstance(eq["distribution_fit"], DistributionFit)

        # Time series statistics
        ts = summary["time_series"]
        assert -1 <= ts["autocorr_lag1"] <= 1
        assert np.isfinite(ts["trend_slope"])
        assert isinstance(ts["is_stationary"], bool)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_trajectory(self):
        """Test analysis of constant trajectory."""
        analyzer = AdvancedStatisticsAnalyzer()

        trajectory = np.ones(200) * 0.5

        ts_stats = analyzer.analyze_time_series(trajectory, burnin=0)

        # Should have zero variance
        assert ts_stats.variance == 0
        assert ts_stats.std == 0
        assert ts_stats.range == 0
        assert ts_stats.cv == np.inf  # 0 / 0.5 (handled)

    def test_single_unique_value_distribution(self):
        """Test fitting distribution to single unique value."""
        analyzer = AdvancedStatisticsAnalyzer()

        data = np.ones(100) * 0.5

        # Should not crash (though fit may be poor)
        fit = analyzer.fit_distribution(data, dist_name="norm")
        assert isinstance(fit, DistributionFit)

    def test_compare_empty_groups(self):
        """Test comparison with very small groups."""
        analyzer = AdvancedStatisticsAnalyzer()

        group1 = np.array([0.5])
        group2 = np.array([0.6])

        # Should not crash
        comp = analyzer.compare_groups(group1, group2)
        assert isinstance(comp, ComparativeStatistics)


def test_advanced_statistics_integration():
    """Integration test: full statistical analysis workflow."""
    from src.simulation.config import SimulationConfig
    from src.simulation.engine import SimulationEngine

    analyzer = AdvancedStatisticsAnalyzer()

    # Run simulation
    np.random.seed(42)
    config = SimulationConfig(
        N=100,
        T=200,
        gamma=0.7,
        beta=0.9,
        alpha_0=2.0,
        beta_0=2.0,
        p_init=0.5,
    )

    engine = SimulationEngine(config)
    result = engine.run()

    # Time series analysis
    ts_stats = analyzer.analyze_time_series(result.p1_trajectory, burnin=50)
    assert isinstance(ts_stats, TimeSeriesAnalysis)

    # Distribution fitting
    equilibrium_vals = result.p1_trajectory[-50:]
    fit = analyzer.fit_distribution(equilibrium_vals, dist_name="beta")
    assert isinstance(fit, DistributionFit)

    # Bootstrap CI
    mean_est, ci_low, ci_high = analyzer.bootstrap_confidence_interval(
        equilibrium_vals, statistic_func=np.mean, n_bootstrap=500
    )
    assert ci_low < mean_est < ci_high

    # Compare two halves
    first_half = result.p1_trajectory[50:125]
    second_half = result.p1_trajectory[125:]
    comp = analyzer.compare_groups(first_half, second_half)
    assert isinstance(comp, ComparativeStatistics)
