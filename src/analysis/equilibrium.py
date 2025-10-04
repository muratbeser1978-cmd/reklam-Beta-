"""
Equilibrium State and Analysis.

Defines equilibrium types and implements equilibrium finder.
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from scipy.optimize import root_scalar
from src.analysis.best_response import best_response, br_derivative
from src.distributions.constants import GAMMA_STAR


class EquilibriumType(Enum):
    """Classification of equilibrium states (Eq 41-47)."""
    SYMMETRIC = "symmetric"  # p* = 0.5, occurs for γ > γ*
    ASYMMETRIC_HIGH = "asymmetric_high"  # p* > 0.5, occurs for γ < γ*
    ASYMMETRIC_LOW = "asymmetric_low"  # p* < 0.5, occurs for γ < γ*


@dataclass
class EquilibriumState:
    """
    Equilibrium characterization for given γ.

    Equation References:
        - p_star: Eq 38 (fixed point p* = BR(p*))
        - type: Eq 41-47 (classification)
        - stable: Eq 39-40 (stability via BR'(p*) < 1)
        - gamma: Eq 3 (parameter value)
        - basin_of_attraction: Eq 48-50 (initial conditions converging to p*)
    """

    p_star: float  # Equilibrium market share
    type: EquilibriumType  # Classification
    stable: bool  # Stability flag
    gamma: float  # Parameter value
    basin_of_attraction: Tuple[float, float]  # (lower, upper) bounds

    def __post_init__(self):
        """Validate equilibrium properties."""
        if not (0.0 <= self.p_star <= 1.0):
            raise ValueError(f"p_star={self.p_star} out of range [0, 1]")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma={self.gamma} out of range [0, 1]")
        l, u = self.basin_of_attraction
        if not (0.0 <= l <= u <= 1.0):
            raise ValueError(f"Invalid basin [{l}, {u}]")


class EquilibriumFinder:
    """Find and classify equilibria."""

    def find_equilibria(self, gamma: float, tol: float = 1e-7) -> List[EquilibriumState]:
        # --- ONE-TIME DIAGNOSTIC ---
        if abs(gamma - 0.6) < 0.01:
            g = lambda p: best_response(p, gamma) - p
            print("\n--- FINAL DIAGNOSTIC (gamma=0.6) ---")
            print(f"g(0.0) = BR(0) - 0 = {g(0.0):.6f}")
            print(f"g(0.5) = BR(0.5) - 0.5 = {g(0.5):.6f}")
            print(f"g(1.0) = BR(1.0) - 1.0 = {g(1.0):.6f}")
            print("------------------------------------\n")
        # --- END DIAGNOSTIC ---

        """
        Find all equilibria for given γ.

        Args:
            gamma: Individual taste weight
            tol: Tolerance for fixed point solver

        Returns:
            List of equilibrium states (1 if γ > γ*, 3 if γ < γ*)

        Algorithm:
            1. Always includes symmetric equilibrium p* = 0.5 (Eq 41)
            2. If γ < γ* = 12/17, find asymmetric equilibria numerically:
               - Solve BR(p) = p for p ≠ 0.5 (Eq 38)
               - Approximate: p^± ≈ 0.5 ± √(γ* - γ) (Eq 44-45)
            3. Check stability via BR'(p*) < 1 (Eq 39-40)

        Guarantees:
            - Always returns at least 1 equilibrium (symmetric)
            - For γ < γ*: returns 3 equilibria (1 unstable symmetric, 2 stable asymmetric)
            - For γ > γ*: returns 1 equilibrium (stable symmetric)
        """
        equilibria = []

        # Symmetric equilibrium always exists
        p_sym = 0.5
        br_derivative_sym = br_derivative(p_sym, gamma)
        stable_sym = abs(br_derivative_sym) < 1.0

        basin_sym = (0.0, 1.0) if stable_sym else (0.5 - tol, 0.5 + tol)

        equilibria.append(
            EquilibriumState(
                p_star=p_sym,
                type=EquilibriumType.SYMMETRIC,
                stable=stable_sym,
                gamma=gamma,
                basin_of_attraction=basin_sym,
            )
        )

        # Check for asymmetric equilibria
        if gamma < GAMMA_STAR:
            # Approximate initial guess (Eq 44-45)
            delta = np.sqrt(GAMMA_STAR - gamma)
            p_high_init = 0.5 + delta
            p_low_init = 0.5 - delta

            found_points = set()

            # Find equilibria from both high and low starting points
            p1 = self._find_fixed_point(gamma, p_high_init, tol)
            if p1 is not None and abs(p1 - 0.5) > tol:
                found_points.add(round(p1, 5))

            p2 = self._find_fixed_point(gamma, p_low_init, tol)
            if p2 is not None and abs(p2 - 0.5) > tol:
                found_points.add(round(p2, 5))

            # Classify and add the unique equilibria found
            for p_star in found_points:
                br_deriv = br_derivative(p_star, gamma)
                is_stable = abs(br_deriv) < 1.0

                if p_star > 0.5:
                    eq_type = EquilibriumType.ASYMMETRIC_HIGH
                    basin = (0.5, 1.0)
                else:
                    eq_type = EquilibriumType.ASYMMETRIC_LOW
                    basin = (0.0, 0.5)

                equilibria.append(
                    EquilibriumState(
                        p_star=p_star,
                        type=eq_type,
                        stable=is_stable,
                        gamma=gamma,
                        basin_of_attraction=basin,
                    )
                )

        return equilibria

    def _find_fixed_point(
        self, gamma: float, p_init: float, tol: float = 1e-7
    ) -> float:
        """
        Find fixed point p* where BR(p*) = p* using a robust solver.
        """
        def g(p):
            return best_response(p, gamma) - p

        # Bracket the root search based on the initial guess
        if p_init > 0.5:
            bracket = [0.5 + 1e-6, 1.0]
        else:
            bracket = [0.0, 0.5 - 1e-6]

        try:
            sol = root_scalar(g, bracket=bracket, x0=p_init, method='bisect', xtol=tol)
            if sol.converged:
                return sol.root
        except ValueError:
            pass

        return None