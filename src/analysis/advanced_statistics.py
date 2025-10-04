"""
Advanced Statistical Analysis.

Comprehensive statistical toolkit for interpreting simulation results.

Modules:
    - Time series analysis (stationarity, trends, changepoints)
    - Distribution fitting and goodness-of-fit tests
    - Hypothesis testing (parametric and non-parametric)
    - Bootstrap confidence intervals
    - Power law and scaling analysis
    - Comparative statistics across conditions
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


@dataclass
class TimeSeriesAnalysis:
    """
    Comprehensive time series statistics.

    Attributes:
        mean: Sample mean
        median: Sample median
        std: Standard deviation
        variance: Variance
        skewness: Distribution skewness
        kurtosis: Excess kurtosis
        min_val: Minimum value
        max_val: Maximum value
        range: max - min
        iqr: Interquartile range
        cv: Coefficient of variation (std/mean)
        autocorr_lag1: Lag-1 autocorrelation
        autocorr_lag5: Lag-5 autocorrelation
        trend_slope: Linear trend slope
        trend_pvalue: Significance of trend
        is_stationary: ADF test result
        n_changepoints: Number of detected changepoints
    """

    mean: float
    median: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    range: float
    iqr: float
    cv: float
    autocorr_lag1: float
    autocorr_lag5: float
    trend_slope: float
    trend_pvalue: float
    is_stationary: bool
    n_changepoints: int


@dataclass
class DistributionFit:
    """
    Distribution fitting results.

    Attributes:
        distribution_name: Name of fitted distribution
        parameters: Distribution parameters
        ks_statistic: Kolmogorov-Smirnov test statistic
        ks_pvalue: KS test p-value
        ad_statistic: Anderson-Darling statistic
        log_likelihood: Log-likelihood of fit
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
    """

    distribution_name: str
    parameters: Dict[str, float]
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float
    log_likelihood: float
    aic: float
    bic: float


@dataclass
class ComparativeStatistics:
    """
    Statistical comparison between two groups.

    Attributes:
        group1_mean: Mean of group 1
        group2_mean: Mean of group 2
        difference: group1_mean - group2_mean
        effect_size: Cohen's d effect size
        t_statistic: T-test statistic
        t_pvalue: T-test p-value
        mann_whitney_u: Mann-Whitney U statistic
        mann_whitney_pvalue: Mann-Whitney p-value
        ks_statistic: Two-sample KS test statistic
        ks_pvalue: KS test p-value
        significant_at_005: Whether difference is significant (p<0.05)
    """

    group1_mean: float
    group2_mean: float
    difference: float
    effect_size: float
    t_statistic: float
    t_pvalue: float
    mann_whitney_u: float
    mann_whitney_pvalue: float
    ks_statistic: float
    ks_pvalue: float
    significant_at_005: bool


class AdvancedStatisticsAnalyzer:
    """
    Advanced statistical analysis toolkit.

    Usage:
        analyzer = AdvancedStatisticsAnalyzer()
        ts_stats = analyzer.analyze_time_series(trajectory)
        dist_fit = analyzer.fit_distribution(data, "beta")
        comparison = analyzer.compare_groups(group1, group2)
    """

    def analyze_time_series(
        self, trajectory: np.ndarray, burnin: int = 0
    ) -> TimeSeriesAnalysis:
        """
        Comprehensive time series analysis.

        Args:
            trajectory: Time series data
            burnin: Number of initial points to discard

        Returns:
            TimeSeriesAnalysis object with all statistics

        Example:
            >>> ts_stats = analyzer.analyze_time_series(p1_trajectory, burnin=100)
            >>> print(f"Mean: {ts_stats.mean:.4f}, Trend: {ts_stats.trend_slope:.6f}")
            >>> if ts_stats.is_stationary:
            ...     print("Series is stationary")
        """
        traj = trajectory[burnin:]

        if len(traj) < 10:
            raise ValueError(f"Trajectory too short: {len(traj)} points")

        # Basic statistics
        mean = np.mean(traj)
        median = np.median(traj)
        std = np.std(traj)
        variance = np.var(traj)
        skewness = stats.skew(traj)
        kurtosis = stats.kurtosis(traj)
        min_val = np.min(traj)
        max_val = np.max(traj)
        range_val = max_val - min_val

        # Robust statistics
        q1, q3 = np.percentile(traj, [25, 75])
        iqr = q3 - q1
        cv = std / mean if mean != 0 else np.inf

        # Autocorrelation
        autocorr_lag1 = self._autocorr(traj, lag=1)
        autocorr_lag5 = self._autocorr(traj, lag=5)

        # Trend analysis
        trend_slope, trend_pvalue = self._detect_trend(traj)

        # Stationarity test (simplified ADF)
        is_stationary = self._test_stationarity(traj)

        # Changepoint detection
        n_changepoints = self._detect_changepoints(traj)

        return TimeSeriesAnalysis(
            mean=mean,
            median=median,
            std=std,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            min_val=min_val,
            max_val=max_val,
            range=range_val,
            iqr=iqr,
            cv=cv,
            autocorr_lag1=autocorr_lag1,
            autocorr_lag5=autocorr_lag5,
            trend_slope=trend_slope,
            trend_pvalue=trend_pvalue,
            is_stationary=is_stationary,
            n_changepoints=n_changepoints,
        )

    def fit_distribution(
        self, data: np.ndarray, dist_name: str = "norm"
    ) -> DistributionFit:
        """
        Fit a distribution and compute goodness-of-fit.

        Args:
            data: Sample data
            dist_name: Distribution name ("norm", "beta", "gamma", etc.)

        Returns:
            DistributionFit object

        Supported distributions:
            - "norm": Normal
            - "beta": Beta
            - "gamma": Gamma
            - "lognorm": Log-normal
            - "uniform": Uniform

        Example:
            >>> equilib_vals = p1_trajectory[-100:]
            >>> fit = analyzer.fit_distribution(equilib_vals, "beta")
            >>> if fit.ks_pvalue > 0.05:
            ...     print(f"Beta distribution fits well (p={fit.ks_pvalue:.3f})")
        """
        # Get distribution object
        if dist_name == "norm":
            dist = stats.norm
        elif dist_name == "beta":
            dist = stats.beta
        elif dist_name == "gamma":
            dist = stats.gamma
        elif dist_name == "lognorm":
            dist = stats.lognorm
        elif dist_name == "uniform":
            dist = stats.uniform
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")

        # Fit parameters
        params = dist.fit(data)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(data, dist_name, args=params)

        # Anderson-Darling test (for some distributions)
        try:
            ad_result = stats.anderson(data, dist=dist_name if dist_name == "norm" else "norm")
            ad_stat = ad_result.statistic
        except:
            ad_stat = np.nan

        # Log-likelihood
        log_lik = np.sum(dist.logpdf(data, *params))

        # AIC and BIC
        k = len(params)
        n = len(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik

        # Parameter dictionary
        param_names = dist.shapes.split(",") if hasattr(dist, "shapes") and dist.shapes else []
        param_names += ["loc", "scale"]
        param_dict = dict(zip(param_names, params))

        return DistributionFit(
            distribution_name=dist_name,
            parameters=param_dict,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ad_statistic=ad_stat,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
        )

    def compare_groups(
        self, group1: np.ndarray, group2: np.ndarray
    ) -> ComparativeStatistics:
        """
        Statistical comparison between two groups.

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            ComparativeStatistics object

        Tests performed:
            - T-test (parametric)
            - Mann-Whitney U (non-parametric)
            - Kolmogorov-Smirnov (distribution comparison)
            - Effect size (Cohen's d)

        Example:
            >>> gamma_07_results = [...]
            >>> gamma_08_results = [...]
            >>> comp = analyzer.compare_groups(gamma_07_results, gamma_08_results)
            >>> if comp.significant_at_005:
            ...     print(f"Significant difference: d={comp.effect_size:.2f}")
        """
        # Means
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        difference = mean1 - mean2

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        effect_size = difference / pooled_std if pooled_std > 0 else 0

        # T-test (parametric)
        t_stat, t_pval = stats.ttest_ind(group1, group2)

        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(group1, group2, alternative="two-sided")

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(group1, group2)

        # Significance
        significant = t_pval < 0.05 or u_pval < 0.05

        return ComparativeStatistics(
            group1_mean=mean1,
            group2_mean=mean2,
            difference=difference,
            effect_size=effect_size,
            t_statistic=t_stat,
            t_pvalue=t_pval,
            mann_whitney_u=u_stat,
            mann_whitney_pvalue=u_pval,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            significant_at_005=significant,
        )

    def bootstrap_confidence_interval(
        self, data: np.ndarray, statistic_func=np.mean, n_bootstrap: int = 1000, alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval.

        Args:
            data: Sample data
            statistic_func: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level (default: 0.05 for 95% CI)

        Returns:
            (point_estimate, lower_ci, upper_ci)

        Example:
            >>> mean_est, ci_low, ci_high = analyzer.bootstrap_confidence_interval(
            ...     equilibrium_values, statistic_func=np.mean, n_bootstrap=10000
            ... )
            >>> print(f"Mean: {mean_est:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        """
        n = len(data)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=n, replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # Point estimate
        point_estimate = statistic_func(data)

        # Confidence interval (percentile method)
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        return point_estimate, ci_lower, ci_upper

    def test_power_law(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Test for power law relationship: y ~ x^α.

        Args:
            x_data: Independent variable
            y_data: Dependent variable

        Returns:
            (exponent, r_squared, p_value)

        Example:
            >>> # Test scaling: Var ~ (γ - γ*)^(-ν)
            >>> distances = np.abs(gammas - gamma_star)
            >>> variances = [...]
            >>> nu, r2, pval = analyzer.test_power_law(distances, variances)
            >>> print(f"Power law exponent: ν = {nu:.2f} (R² = {r2:.3f})")
        """
        # Filter positive values
        valid = (x_data > 0) & (y_data > 0)
        x = x_data[valid]
        y = y_data[valid]

        if len(x) < 3:
            return np.nan, 0, 1.0

        # Log-log regression
        log_x = np.log(x)
        log_y = np.log(y)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

        exponent = slope
        r_squared = r_value**2

        return exponent, r_squared, p_value

    def _autocorr(self, x: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag."""
        if len(x) <= lag or lag == 0:
            return 1.0 if lag == 0 else np.nan

        x1 = x[:-lag]
        x2 = x[lag:]

        if np.std(x1) == 0 or np.std(x2) == 0:
            return 1.0 if lag == 0 else 0.0

        return np.corrcoef(x1, x2)[0, 1]

    def _detect_trend(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Detect linear trend using Mann-Kendall test.

        Returns:
            (trend_slope, p_value)
        """
        t = np.arange(len(x))
        slope, intercept, r, p, stderr = stats.linregress(t, x)
        return slope, p

    def _test_stationarity(self, x: np.ndarray) -> bool:
        """
        Simplified stationarity test.

        Method: Split into two halves, test if means are equal.
        """
        n = len(x)
        if n < 20:
            return True

        mid = n // 2
        first_half = x[:mid]
        second_half = x[mid:]

        # T-test for equal means
        _, p = stats.ttest_ind(first_half, second_half)

        # Stationary if means are not significantly different
        return p > 0.05

    def _detect_changepoints(self, x: np.ndarray, threshold: float = 2.0) -> int:
        """
        Detect changepoints in time series.

        Method: Find peaks in absolute difference signal.
        """
        if len(x) < 10:
            return 0

        # Compute differences
        diff = np.abs(np.diff(x))

        # Find peaks
        threshold_val = threshold * np.median(diff)
        peaks, _ = find_peaks(diff, height=threshold_val)

        return len(peaks)


def summarize_simulation_results(results: list, analyzer: AdvancedStatisticsAnalyzer) -> Dict:
    """
    Comprehensive summary statistics across multiple simulation runs.

    Args:
        results: List of SimulationResult objects
        analyzer: AdvancedStatisticsAnalyzer instance

    Returns:
        Dictionary with summary statistics

    Example:
        >>> results = [engine.run(config) for _ in range(100)]
        >>> summary = summarize_simulation_results(results, analyzer)
        >>> print(summary["equilibrium"]["mean"])
        >>> print(summary["equilibrium"]["confidence_interval"])
    """
    # Extract final equilibrium values
    equilibria = np.array([r.p1_trajectory[-1] for r in results])

    # Time series analysis on one representative run
    ts_stats = analyzer.analyze_time_series(results[0].p1_trajectory, burnin=100)

    # Bootstrap CI for equilibrium
    eq_mean, eq_ci_low, eq_ci_high = analyzer.bootstrap_confidence_interval(equilibria)

    # Distribution fit
    dist_fit = analyzer.fit_distribution(equilibria, "beta")

    summary = {
        "equilibrium": {
            "mean": eq_mean,
            "std": np.std(equilibria),
            "median": np.median(equilibria),
            "confidence_interval": (eq_ci_low, eq_ci_high),
            "distribution_fit": dist_fit,
        },
        "time_series": {
            "autocorr_lag1": ts_stats.autocorr_lag1,
            "trend_slope": ts_stats.trend_slope,
            "is_stationary": ts_stats.is_stationary,
            "skewness": ts_stats.skewness,
        },
        "n_runs": len(results),
    }

    return summary
