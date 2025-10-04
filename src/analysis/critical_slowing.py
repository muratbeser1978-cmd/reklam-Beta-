"""
Critical Slowing Down Analysis.

Analyzes critical phenomena near bifurcation point γ*.
Early warning signals for regime transitions.

Theoretical Background:
    Near γ*, system exhibits critical slowing down:
    - Relaxation time diverges: τ ~ (γ - γ*)^(-ν)
    - Variance increases: Var[p] ~ (γ - γ*)^(-1)
    - Autocorrelation approaches 1: AC ~ 1 - (γ - γ*)

    These are "early warning signals" for approaching bifurcation.

References:
    - Eq 57-58: Variance and autocorrelation scaling
    - Scheffer et al. (2009): Early warning signals
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy import stats
from scipy.optimize import curve_fit


@dataclass
class CriticalSlowingMetrics:
    """
    Critical slowing down indicators.

    Attributes:
        variance: Time series variance
        autocorr_lag1: Autocorrelation at lag 1
        autocorr_lag5: Autocorrelation at lag 5
        skewness: Distribution skewness
        kurtosis: Distribution kurtosis
        relaxation_time: Estimated relaxation time
        detrended_variance: Variance after detrending
        gamma: Parameter value
        distance_to_critical: |γ - γ*|
    """

    variance: float
    autocorr_lag1: float
    autocorr_lag5: float
    skewness: float
    kurtosis: float
    relaxation_time: Optional[float]
    detrended_variance: float
    gamma: float
    distance_to_critical: float


@dataclass
class EarlyWarningSignals:
    """
    Early warning signals for approaching bifurcation.

    Attributes:
        kendall_tau_variance: Kendall's τ for increasing variance trend
        kendall_tau_ac: Kendall's τ for increasing autocorrelation trend
        p_value_variance: Statistical significance for variance trend
        p_value_ac: Statistical significance for AC trend
        warning_level: 0=none, 1=weak, 2=moderate, 3=strong
    """

    kendall_tau_variance: float
    kendall_tau_ac: float
    p_value_variance: float
    p_value_ac: float
    warning_level: int

    @property
    def variance_kendall_pvalue(self) -> float:
        """Alias for backward compatibility."""
        return self.p_value_variance

    @property
    def ac_kendall_pvalue(self) -> float:
        """Alias for backward compatibility."""
        return self.p_value_ac

    @property
    def variance_increasing(self) -> bool:
        """Whether variance shows significant increasing trend."""
        return self.kendall_tau_variance > 0 and self.p_value_variance < 0.05

    @property
    def ac_increasing(self) -> bool:
        """Whether autocorrelation shows significant increasing trend."""
        return self.kendall_tau_ac > 0 and self.p_value_ac < 0.05


@dataclass
class ScalingAnalysis:
    """
    Power law scaling analysis results.

    Attributes:
        nu_exponent: Scaling exponent ν in Var ~ |γ-γ*|^(-ν)
        r_squared: R² goodness-of-fit
        pvalue: Statistical significance
        fitted_variances: Fitted variance values
        gamma_values: Gamma values used
    """

    nu_exponent: float
    r_squared: float
    pvalue: float
    fitted_variances: np.ndarray
    gamma_values: np.ndarray


class CriticalSlowingAnalyzer:
    """
    Analyze critical slowing down phenomena.

    Usage:
        analyzer = CriticalSlowingAnalyzer()
        metrics = analyzer.compute_metrics(p_trajectory, gamma)
        scaling = analyzer.estimate_scaling_exponent(gamma_values, variances)
    """

    def __init__(self, gamma_star: float = 12.0 / 17.0):
        """
        Initialize analyzer.

        Args:
            gamma_star: Critical bifurcation threshold
        """
        self.gamma_star = gamma_star

    def compute_metrics(
        self, trajectory: np.ndarray, gamma: float, burnin: int = 100
    ) -> CriticalSlowingMetrics:
        """
        Compute all critical slowing metrics.

        Args:
            trajectory: Market share time series p(t)
            gamma: Individual taste weight parameter
            burnin: Number of initial timesteps to discard

        Returns:
            CriticalSlowingMetrics object

        Example:
            >>> trajectory = result.p1_trajectory
            >>> metrics = analyzer.compute_metrics(trajectory, gamma=0.71)
            >>> print(f"Variance: {metrics.variance:.4f}")
            >>> print(f"AC(1): {metrics.autocorr_lag1:.4f}")
        """
        # Remove burnin period
        traj = trajectory[burnin:]

        if len(traj) < 10:
            raise ValueError(f"Trajectory too short after burnin: {len(traj)} steps")

        # Basic statistics
        variance = np.var(traj)
        skewness = stats.skew(traj)
        kurtosis = stats.kurtosis(traj)

        # Autocorrelation
        autocorr_lag1 = self._compute_autocorrelation(traj, lag=1)
        autocorr_lag5 = self._compute_autocorrelation(traj, lag=5)

        # Relaxation time (from autocorrelation decay)
        relaxation_time = self._estimate_relaxation_time(traj)

        # Detrended variance (remove linear trend first)
        detrended_variance = self._detrended_variance(traj)

        # Distance to critical point
        distance_to_critical = abs(gamma - self.gamma_star)

        return CriticalSlowingMetrics(
            variance=variance,
            autocorr_lag1=autocorr_lag1,
            autocorr_lag5=autocorr_lag5,
            skewness=skewness,
            kurtosis=kurtosis,
            relaxation_time=relaxation_time,
            detrended_variance=detrended_variance,
            gamma=gamma,
            distance_to_critical=distance_to_critical,
        )

    def detect_early_warnings(
        self, trajectory: np.ndarray, window_size: int = 50, step: int = 10
    ) -> EarlyWarningSignals:
        """
        Detect early warning signals using rolling window analysis.

        Method:
            1. Divide trajectory into rolling windows
            2. Compute variance and AC for each window
            3. Test for increasing trend (Kendall's τ)
            4. Assess warning level based on significance

        Args:
            trajectory: Market share time series
            window_size: Size of rolling window
            step: Step size between windows

        Returns:
            EarlyWarningSignals object

        Interpretation:
            - Positive Kendall τ → Increasing trend
            - p < 0.05 → Statistically significant
            - Warning level 3 → Strong evidence of approaching transition

        Example:
            >>> warnings = analyzer.detect_early_warnings(trajectory)
            >>> if warnings.warning_level >= 2:
            >>>     print("WARNING: Approaching bifurcation!")
        """
        if len(trajectory) < window_size * 2:
            raise ValueError("Trajectory too short for rolling window analysis")

        # Compute rolling statistics
        variances = []
        autocorrs = []
        window_centers = []

        for start in range(0, len(trajectory) - window_size, step):
            window = trajectory[start : start + window_size]
            window_centers.append(start + window_size // 2)
            variances.append(np.var(window))
            autocorrs.append(self._compute_autocorrelation(window, lag=1))

        variances = np.array(variances)
        autocorrs = np.array(autocorrs)

        # Test for increasing trend (Kendall's tau)
        tau_var, p_var = stats.kendalltau(window_centers, variances)
        tau_ac, p_ac = stats.kendalltau(window_centers, autocorrs)

        # Determine warning level
        warning_level = 0
        if p_var < 0.05 and tau_var > 0:
            warning_level += 1
        if p_ac < 0.05 and tau_ac > 0:
            warning_level += 1
        if p_var < 0.01 and p_ac < 0.01 and tau_var > 0.3 and tau_ac > 0.3:
            warning_level = 3

        return EarlyWarningSignals(
            kendall_tau_variance=tau_var,
            kendall_tau_ac=tau_ac,
            p_value_variance=p_var,
            p_value_ac=p_ac,
            warning_level=warning_level,
        )

    def estimate_scaling_exponent(
        self, gamma_values: np.ndarray, variances: np.ndarray, above_critical: bool = True
    ) -> ScalingAnalysis:
        """
        Estimate scaling exponent ν in Var ~ (γ - γ*)^(-ν).

        Theory:
            Eq 57: Var[pₜ] ~ (γ - γ*)^(-1)  →  ν = 1 (mean field)

        Args:
            gamma_values: Array of γ values
            variances: Corresponding variance values
            above_critical: If True, use γ > γ*; else γ < γ*

        Returns:
            ScalingAnalysis object

        Example:
            >>> gammas = np.linspace(0.71, 0.80, 10)
            >>> vars = [compute_variance(gamma) for gamma in gammas]
            >>> scaling = analyzer.estimate_scaling_exponent(gammas, vars)
            >>> print(f"Scaling exponent: ν = {scaling.nu_exponent:.2f} (theory: 1.0)")
        """
        # Filter to appropriate regime
        if above_critical:
            mask = gamma_values > self.gamma_star
        else:
            mask = gamma_values < self.gamma_star

        gamma_filtered = gamma_values[mask]
        var_filtered = variances[mask]

        if len(gamma_filtered) < 3:
            return ScalingAnalysis(
                nu_exponent=np.nan,
                r_squared=0.0,
                pvalue=1.0,
                fitted_variances=np.array([]),
                gamma_values=gamma_filtered,
            )

        # Distance from critical point
        distance = np.abs(gamma_filtered - self.gamma_star)

        # Log-log fit: log(Var) = -ν·log(distance) + const
        # Filter out very small distances to avoid numerical issues
        valid = distance > 1e-4
        if valid.sum() < 3:
            return ScalingAnalysis(
                nu_exponent=np.nan,
                r_squared=0.0,
                pvalue=1.0,
                fitted_variances=np.array([]),
                gamma_values=gamma_filtered,
            )

        log_dist = np.log(distance[valid])
        log_var = np.log(var_filtered[valid])

        # Linear regression in log-log space
        from scipy import stats as scipy_stats
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_dist, log_var)
        exponent = -slope  # ν = -slope
        r_squared = r_value ** 2

        # Generate fitted curve
        fitted_variances = np.exp(intercept) * distance**slope

        return ScalingAnalysis(
            nu_exponent=exponent,
            r_squared=r_squared,
            pvalue=p_value,
            fitted_variances=fitted_variances,
            gamma_values=gamma_filtered,
        )

    def compute_susceptibility(
        self, trajectories: list, gamma: float, perturbation_size: float = 0.01
    ) -> float:
        """
        Compute susceptibility χ = ∂⟨p⟩/∂h.

        Method:
            Run simulations with small perturbation, measure response.
            χ ~ (γ - γ*)^(-1) near critical point.

        Args:
            trajectories: List of p trajectories with different perturbations
            gamma: Parameter value
            perturbation_size: Size of applied perturbation

        Returns:
            Susceptibility value

        Note:
            Requires running multiple simulations with perturbed initial conditions.
        """
        if len(trajectories) < 2:
            raise ValueError("Need at least 2 trajectories for susceptibility")

        # Compute equilibrium values (last 10% of trajectory)
        equilibria = []
        for traj in trajectories:
            eq_window = traj[-len(traj) // 10 :]
            equilibria.append(np.mean(eq_window))

        # Susceptibility = response / perturbation
        mean_response = np.std(equilibria)
        susceptibility = mean_response / perturbation_size

        return susceptibility

    def _compute_autocorrelation(self, trajectory: np.ndarray, lag: int) -> float:
        """
        Compute autocorrelation at given lag.

        Formula:
            AC(k) = Corr[pₜ, pₜ₊ₖ]
        """
        if len(trajectory) <= lag:
            return np.nan

        x = trajectory[:-lag] if lag > 0 else trajectory
        y = trajectory[lag:] if lag > 0 else trajectory

        if len(x) == 0 or len(y) == 0:
            return np.nan

        # Pearson correlation
        if np.std(x) == 0 or np.std(y) == 0:
            return 1.0 if lag == 0 else 0.0

        return np.corrcoef(x, y)[0, 1]

    def _estimate_relaxation_time(self, trajectory: np.ndarray, max_lag: int = 50) -> Optional[float]:
        """
        Estimate relaxation time from autocorrelation decay.

        Method:
            Fit AC(k) = exp(-k/τ) and extract τ.

        Returns:
            Relaxation time τ (in timesteps), or None if fit fails
        """
        if len(trajectory) < max_lag * 2:
            return None

        # Compute autocorrelation function
        lags = np.arange(1, min(max_lag, len(trajectory) // 2))
        ac_values = [self._compute_autocorrelation(trajectory, lag) for lag in lags]
        ac_values = np.array(ac_values)

        # Remove NaNs
        valid = ~np.isnan(ac_values) & (ac_values > 0)
        if valid.sum() < 5:
            return None

        lags_valid = lags[valid]
        ac_valid = ac_values[valid]

        # Exponential fit: AC(k) = exp(-k/τ)
        try:

            def exp_decay(k, tau):
                return np.exp(-k / tau)

            (tau,), _ = curve_fit(exp_decay, lags_valid, ac_valid, p0=[10], maxfev=1000)
            return float(tau) if tau > 0 else None
        except:
            return None

    def _detrended_variance(self, trajectory: np.ndarray) -> float:
        """
        Compute variance after removing linear trend.

        Purpose:
            Separate long-term drift from fluctuations.
        """
        t = np.arange(len(trajectory))
        # Linear detrending
        slope, intercept = np.polyfit(t, trajectory, 1)
        trend = slope * t + intercept
        detrended = trajectory - trend
        return np.var(detrended)
