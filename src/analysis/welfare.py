"""
Social Welfare Analysis.

Implements welfare calculations and welfare loss measurement.
"""
import numpy as np
from dataclasses import dataclass
from src.simulation.utilities import Q


@dataclass
class WelfareAnalysis:
    """
    Social welfare measures.

    Equation References:
        - welfare_trajectory: W(t) over time (Eq 64)
        - welfare_optimal: W^opt at p=0.5 (Eq 65)
        - welfare_equilibrium: W^eq at p* (Eq 66)
        - welfare_loss: ΔW = W^opt - W^eq (Eq 67)
    """

    welfare_trajectory: np.ndarray  # W(t) for all t, Eq 64
    welfare_optimal: float  # Eq 65
    welfare_equilibrium: float  # Eq 66
    welfare_loss: float  # Eq 67

    def __post_init__(self):
        """Verify welfare calculations."""
        if self.welfare_loss < -1e-10:  # Allow small numerical error
            raise ValueError(f"Welfare loss cannot be negative: ΔW = {self.welfare_loss}")
        if self.welfare_optimal < self.welfare_equilibrium - 1e-10:
            raise ValueError("Optimal welfare must be ≥ equilibrium welfare")


class WelfareCalculator:
    """Compute social welfare measures."""

    def compute_welfare(
        self, p_trajectory: np.ndarray, theta: np.ndarray, gamma: float
    ) -> WelfareAnalysis:
        """
        Calculate welfare over simulation.

        Args:
            p_trajectory: Market share p₁(t) over time
            theta: True consumer preferences (N, 2)
            gamma: Individual taste weight

        Returns:
            Welfare analysis
        """
        T = len(p_trajectory)
        welfare_trajectory = np.zeros(T)

        # Correctly calculate welfare at each time step
        for t in range(T):
            p1 = p_trajectory[t]
            p2 = 1.0 - p1
            utility_brand_1 = Q(theta[:, 0], p1, gamma)
            utility_brand_2 = Q(theta[:, 1], p2, gamma)
            # Welfare is the average of the maximum utility each consumer can achieve
            welfare_trajectory[t] = np.mean(np.maximum(utility_brand_1, utility_brand_2))

        # Find the true optimal welfare by maximizing the welfare function W(p)
        p_grid = np.linspace(0, 1, 501)  # Grid to search for optimal p
        welfare_grid = [
            np.mean(np.maximum(Q(theta[:, 0], p, gamma), Q(theta[:, 1], 1 - p, gamma)))
            for p in p_grid
        ]
        welfare_optimal = np.max(welfare_grid)

        # Equilibrium welfare (use final value as proxy for equilibrium)
        welfare_equilibrium = welfare_trajectory[-1]

        # Welfare loss
        welfare_loss = welfare_optimal - welfare_equilibrium

        return WelfareAnalysis(
            welfare_trajectory=welfare_trajectory,
            welfare_optimal=welfare_optimal,
            welfare_equilibrium=welfare_equilibrium,
            welfare_loss=welfare_loss,
        )
