"""
Bifurcation Analysis.

Analyzes bifurcation phenomena and critical slowing down near γ*.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from src.analysis.equilibrium import EquilibriumFinder, EquilibriumType
from src.distributions.constants import GAMMA_STAR


@dataclass
class BifurcationData:
    """
    Equilibrium positions across γ parameter range.

    Equation References:
        - gamma_grid: γ values (Eq 3)
        - p_symmetric: Symmetric equilibrium (always 0.5) (Eq 41)
        - p_asymmetric_high: Upper asymmetric equilibrium (NaN if γ>γ*) (Eq 46)
        - p_asymmetric_low: Lower asymmetric equilibrium (NaN if γ>γ*) (Eq 47)
        - stable_mask_symmetric: True if symmetric equilibrium stable (Eq 42)
        - stable_mask_asymmetric: True if asymmetric equilibria stable (Eq 43)
    """

    gamma_grid: np.ndarray  # (M,) γ values
    p_symmetric: np.ndarray  # (M,) always 0.5
    p_asymmetric_high: np.ndarray  # (M,) upper equilibrium or NaN
    p_asymmetric_low: np.ndarray  # (M,) lower equilibrium or NaN
    stable_mask_symmetric: np.ndarray  # (M,) stability of symmetric
    stable_mask_asymmetric: np.ndarray  # (M,) stability of asymmetric


class BifurcationAnalyzer:
    """Analyze bifurcation phenomena."""

    def __init__(self):
        self.finder = EquilibriumFinder()

    def compute_bifurcation_diagram(
        self, gamma_min: float = 0.5, gamma_max: float = 0.9, resolution: int = 100
    ) -> BifurcationData:
        """
        Generate bifurcation diagram data.

        Args:
            gamma_min: Minimum γ value
            gamma_max: Maximum γ value
            resolution: Number of γ points

        Returns:
            BifurcationData with equilibrium positions

        Equation References:
            - Eq 8: γ* = 12/17 (critical threshold)
            - Eq 41: p^sym = 0.5 (always exists)
            - Eq 44-47: p^± = 0.5 ± δ(γ) for γ < γ*
            - Eq 45: δ(γ) ~ √(γ* - γ) (bifurcation amplitude)
        """
        gamma_grid = np.linspace(gamma_min, gamma_max, resolution)

        # Preallocate arrays
        p_symmetric = np.full(resolution, 0.5)
        p_asymmetric_high = np.full(resolution, np.nan)
        p_asymmetric_low = np.full(resolution, np.nan)
        stable_symmetric = np.zeros(resolution, dtype=bool)
        stable_asymmetric = np.zeros(resolution, dtype=bool)

        # Find equilibria for each γ
        for i, gamma in enumerate(gamma_grid):
            equilibria = self.finder.find_equilibria(gamma)

            for eq in equilibria:
                if eq.type == EquilibriumType.SYMMETRIC:
                    stable_symmetric[i] = eq.stable
                elif eq.type == EquilibriumType.ASYMMETRIC_HIGH:
                    p_asymmetric_high[i] = eq.p_star
                    stable_asymmetric[i] = eq.stable
                elif eq.type == EquilibriumType.ASYMMETRIC_LOW:
                    p_asymmetric_low[i] = eq.p_star

        return BifurcationData(
            gamma_grid=gamma_grid,
            p_symmetric=p_symmetric,
            p_asymmetric_high=p_asymmetric_high,
            p_asymmetric_low=p_asymmetric_low,
            stable_mask_symmetric=stable_symmetric,
            stable_mask_asymmetric=stable_asymmetric,
        )

    def detect_critical_slowing_down(
        self, gamma_values: np.ndarray, results: List
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect critical slowing down near γ*.

        Args:
            gamma_values: γ parameter values
            results: Simulation results for each γ

        Returns:
            (variances, autocorrelations) near γ*

        Equation References:
            - Eq 57: Var[pₜ] ~ (γ - γ*)⁻¹ as γ → γ*⁺
            - Eq 58: AC[pₜ] ~ 1 - (γ - γ*) as γ → γ*⁺

        Critical Behavior:
            - Variance increases as γ approaches γ* from above
            - Autocorrelation approaches 1 (slow decay)
            - System takes longer to equilibrate
        """
        variances = []
        autocorrelations = []

        for result in results:
            # Compute variance
            var = np.var(result.p1_trajectory)
            variances.append(var)

            # Compute autocorrelation at lag 1
            traj = result.p1_trajectory
            if len(traj) > 1:
                ac1 = np.corrcoef(traj[:-1], traj[1:])[0, 1]
                autocorrelations.append(ac1)
            else:
                autocorrelations.append(np.nan)

        return np.array(variances), np.array(autocorrelations)

    def find_bifurcation_point(
        self, gamma_min: float = 0.65, gamma_max: float = 0.75, tol: float = 1e-6
    ) -> float:
        """
        Find precise bifurcation point γ*.

        Args:
            gamma_min: Lower bound for search
            gamma_max: Upper bound for search
            tol: Convergence tolerance

        Returns:
            γ* value where bifurcation occurs

        Method:
            Binary search for γ where transition from 1 to 3 equilibria occurs.
        """
        # We know analytically that γ* = 12/17, but this demonstrates the method

        while gamma_max - gamma_min > tol:
            gamma_mid = (gamma_min + gamma_max) / 2.0
            equilibria = self.finder.find_equilibria(gamma_mid)

            if len(equilibria) == 3:
                # Below critical point (3 equilibria)
                gamma_min = gamma_mid
            else:
                # Above critical point (1 equilibrium)
                gamma_max = gamma_mid

        return (gamma_min + gamma_max) / 2.0

    def compute_bifurcation_amplitude(self, gamma: float) -> float:
        """
        Compute bifurcation amplitude δ(γ).

        Args:
            gamma: Individual taste weight (must be < γ*)

        Returns:
            δ(γ): Distance of asymmetric equilibria from 0.5

        Equation:
            Eq 45: δ(γ) ~ √(γ* - γ) for γ < γ*

        Note:
            This is an approximation valid near the bifurcation point.
        """
        if gamma >= GAMMA_STAR:
            return 0.0  # No bifurcation

        # Theoretical approximation
        delta_approx = np.sqrt(GAMMA_STAR - gamma)

        # Get actual equilibria
        equilibria = self.finder.find_equilibria(gamma)
        p_high = None
        for eq in equilibria:
            if eq.type == EquilibriumType.ASYMMETRIC_HIGH:
                p_high = eq.p_star
                break

        if p_high is not None:
            delta_actual = abs(p_high - 0.5)
            return delta_actual
        else:
            return delta_approx
