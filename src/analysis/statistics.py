"""
Statistical Analysis of Time Series.

Implements statistical measures for market share trajectories.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy import stats


@dataclass
class TimeSeriesStatistics:
    """
    Statistical measures of market share trajectory.

    Equation References:
        - variance: Eq 57 (Var[pₜ])
        - autocorrelation: Eq 58 (AC[pₜ] at various lags)
        - skewness: Eq 59 (Skew[pₜ])
        - mean, std: basic statistics
    """

    mean: float
    variance: float  # Eq 57
    std: float
    autocorrelation: np.ndarray  # AC at lags [1, 5, 10, 20], Eq 58
    skewness: float  # Eq 59
    convergence_time: Optional[int]  # Timestep when equilibrium reached


class StatisticalAnalyzer:
    """Compute time series statistics."""

    def compute_statistics(
        self, trajectory: np.ndarray, equilibrium_p: Optional[float] = None, threshold: float = 0.01
    ) -> TimeSeriesStatistics:
        """
        Analyze market share trajectory.

        Args:
            trajectory: p₁(t) values over time
            equilibrium_p: Known equilibrium value (for convergence detection)
            threshold: Convergence threshold

        Returns:
            Statistical summary

        Equation Coverage:
            - Mean, variance (Eq 57), std: basic statistics
            - Autocorrelation (Eq 58): AC[pₜ] at lags [1, 5, 10, 20]
            - Skewness (Eq 59): distribution asymmetry
            - Convergence time: first t where |p(t) - p*| < threshold
        """
        # Basic statistics
        mean = np.mean(trajectory)
        variance = np.var(trajectory)  # Eq 57
        std = np.std(trajectory)

        # Autocorrelation at specified lags (Eq 58)
        lags = [1, 5, 10, 20]
        autocorr = np.array([self._autocorrelation(trajectory, lag) for lag in lags])

        # Skewness (Eq 59)
        skewness = stats.skew(trajectory)

        # Convergence time
        convergence_time = None
        if equilibrium_p is not None:
            deviations = np.abs(trajectory - equilibrium_p)
            converged_indices = np.where(deviations < threshold)[0]
            if len(converged_indices) > 0:
                # First time it converges and stays converged
                for idx in converged_indices:
                    if np.all(deviations[idx:] < threshold):
                        convergence_time = int(idx)
                        break

        return TimeSeriesStatistics(
            mean=mean,
            variance=variance,
            std=std,
            autocorrelation=autocorr,
            skewness=skewness,
            convergence_time=convergence_time,
        )

    def _autocorrelation(self, x: np.ndarray, lag: int) -> float:
        """
        Compute autocorrelation at given lag.

        Args:
            x: Time series
            lag: Lag value

        Returns:
            Autocorrelation coefficient at lag

        Equation:
            Eq 58: AC(lag) = Corr[x(t), x(t-lag)]
        """
        if lag >= len(x):
            return np.nan

        # Remove mean
        x_centered = x - np.mean(x)

        # Compute autocorrelation
        numerator = np.sum(x_centered[lag:] * x_centered[:-lag])
        denominator = np.sum(x_centered**2)

        if denominator == 0:
            return np.nan

        return numerator / denominator

    def compute_learning_metrics(
        self, theta_true: np.ndarray, theta_hat: np.ndarray, n_eff: np.ndarray
    ) -> dict:
        """
        Analyze learning accuracy.

        Args:
            theta_true: True preferences θᵢ,ₐ
            theta_hat: Posterior means θ̂ᵢ,ₐ
            n_eff: Effective experience nᵢ,ₐ^eff

        Returns:
            {"belief_error": E[|θ̂-θ|], "convergence_rate": O(1/√n)}

        Equation References:
            - Eq 54: Learning rate O(1/√n_{eff})
            - Eq 30: θ̂ᵢ,ₐ = αᵢ,ₐ / (αᵢ,ₐ + βᵢ,ₐ)
            - Eq 29: n_{eff} = α + β - (α₀ + β₀)
        """
        # Belief error (Eq 54)
        belief_error = np.mean(np.abs(theta_hat - theta_true))

        # Convergence rate analysis
        # Error should scale as O(1/√n_eff)
        n_eff_mean = np.mean(n_eff[n_eff > 0])
        if n_eff_mean > 0:
            expected_rate = 1.0 / np.sqrt(n_eff_mean)
        else:
            expected_rate = np.nan

        return {
            "belief_error": float(belief_error),
            "convergence_rate": float(expected_rate),
            "mean_experience": float(n_eff_mean) if n_eff_mean > 0 else 0.0,
        }
