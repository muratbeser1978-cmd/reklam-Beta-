"""
Social Welfare Analysis (Corrected for Eq 64).

Implements welfare calculations following exact model specification.

Key Difference from Previous Version:
    OLD: W(t) = E[max{U₁, U₂}]  (ex-ante potential welfare)
    NEW: W(t) = Σₐ pₐ(t)·Q̄ₐ(t)  (ex-post realized welfare, Eq 64)

Equation References:
    - Eq 63: Q̄ₐ(t) = E[q_{i,a}(t) | cᵢ(t)=a]  (average utility of choosers)
    - Eq 64: W(t) = Σₐ pₐ(t) · Q̄ₐ(t)         (total welfare)
    - Eq 65: W^opt at p=0.5
    - Eq 66: W^eq at equilibrium p*
    - Eq 67: ΔW = W^opt - W^eq
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.simulation.utilities import Q


@dataclass
class WelfareAnalysisCorrected:
    """
    Social welfare measures (Eq 64 compliant).

    Attributes:
        welfare_trajectory: W(t) over time (Eq 64)
        welfare_optimal: W^opt at p=0.5 (Eq 65)
        welfare_equilibrium: W^eq at p* (Eq 66)
        welfare_loss: ΔW = W^opt - W^eq (Eq 67)
        Q_bar_trajectories: (T, 2) array of Q̄ₐ(t) for both brands
        ex_ante_welfare: E[max{U₁, U₂}] for comparison
    """

    welfare_trajectory: np.ndarray  # (T,)
    welfare_optimal: float
    welfare_equilibrium: float
    welfare_loss: float
    Q_bar_trajectories: np.ndarray  # (T, 2)
    ex_ante_welfare: np.ndarray  # (T,) for comparison

    def __post_init__(self):
        """Verify welfare calculations."""
        if self.welfare_loss < -1e-6:
            raise ValueError(f"Welfare loss cannot be negative: ΔW = {self.welfare_loss}")
        if self.welfare_optimal < self.welfare_equilibrium - 1e-6:
            raise ValueError("Optimal welfare must be ≥ equilibrium welfare")


class WelfareCalculatorCorrected:
    """
    Compute social welfare following Eq 64.

    Usage:
        calculator = WelfareCalculatorCorrected()
        welfare = calculator.compute_welfare(
            p_trajectory, theta, gamma, choices
        )
    """

    def compute_welfare(
        self,
        p_trajectory: np.ndarray,
        theta: np.ndarray,
        gamma: float,
        choices: Optional[np.ndarray] = None,
    ) -> WelfareAnalysisCorrected:
        """
        Calculate welfare following Eq 64.

        Args:
            p_trajectory: (T+1,) market share p₁(t)
            theta: (N, 2) true preferences
            gamma: Individual taste weight
            choices: (N, T) choice history (if available)

        Returns:
            WelfareAnalysisCorrected object

        Implementation Notes:
            - If choices available: Compute exact Q̄ₐ(t) from actual choosers
            - If choices unavailable: Estimate Q̄ₐ(t) from population statistics

        Example:
            >>> from src.simulation.result import SimulationResult
            >>> welfare = calculator.compute_welfare(
            ...     result.p1_trajectory,
            ...     result.consumer_state.theta,
            ...     gamma=0.7,
            ...     choices=result.choices
            ... )
            >>> print(f"Welfare loss: {welfare.welfare_loss:.4f}")
        """
        T = len(p_trajectory) - 1  # trajectory is (T+1,), choices are (N, T)
        N = theta.shape[0]

        # Storage
        welfare_traj = np.zeros(T + 1)
        Q_bar_traj = np.zeros((T + 1, 2))
        ex_ante_traj = np.zeros(T + 1)

        if choices is not None and choices.shape[1] != T:
            raise ValueError(f"Choices shape {choices.shape} inconsistent with trajectory length {T}")

        # Compute welfare at each timestep
        for t in range(T + 1):
            p1 = p_trajectory[t]
            p2 = 1.0 - p1

            # True utilities for all consumers
            q_i_1 = Q(theta[:, 0], p1, gamma)  # (N,)
            q_i_2 = Q(theta[:, 1], p2, gamma)  # (N,)

            if choices is not None and t < T:
                # Use actual choices at time t
                choices_t = choices[:, t]  # (N,)

                # Q̄₁(t): Average utility of those who chose brand 1
                brand_1_choosers = choices_t == 0
                Q_bar_1 = q_i_1[brand_1_choosers].mean() if brand_1_choosers.any() else 0.0

                # Q̄₂(t): Average utility of those who chose brand 2
                brand_2_choosers = choices_t == 1
                Q_bar_2 = q_i_2[brand_2_choosers].mean() if brand_2_choosers.any() else 0.0

            else:
                # Estimate: Assume consumers choose based on expected utility
                # Q̄ₐ ≈ E[qᵢ,ₐ | qᵢ,ₐ > qᵢ,ₐ']
                # Simplified: Use full population mean
                Q_bar_1 = q_i_1.mean()
                Q_bar_2 = q_i_2.mean()

            # Store Q̄ values
            Q_bar_traj[t, 0] = Q_bar_1
            Q_bar_traj[t, 1] = Q_bar_2

            # Welfare (Eq 64): W(t) = p₁·Q̄₁ + p₂·Q̄₂
            welfare_traj[t] = p1 * Q_bar_1 + p2 * Q_bar_2

            # Ex-ante welfare (for comparison): E[max{q₁, q₂}]
            ex_ante_traj[t] = np.maximum(q_i_1, q_i_2).mean()

        # Optimal welfare (Eq 65): W^opt at p=0.5
        # Search over grid to find true optimum
        p_grid = np.linspace(0, 1, 501)
        welfare_grid = []

        for p in p_grid:
            q1 = Q(theta[:, 0], p, gamma)
            q2 = Q(theta[:, 1], 1 - p, gamma)

            if choices is not None:
                # Use last timestep choices as proxy
                choices_last = choices[:, -1]
                choosers_1 = choices_last == 0
                choosers_2 = choices_last == 1

                Q_bar_1_opt = q1[choosers_1].mean() if choosers_1.any() else q1.mean()
                Q_bar_2_opt = q2[choosers_2].mean() if choosers_2.any() else q2.mean()
            else:
                Q_bar_1_opt = q1.mean()
                Q_bar_2_opt = q2.mean()

            W_p = p * Q_bar_1_opt + (1 - p) * Q_bar_2_opt
            welfare_grid.append(W_p)

        welfare_optimal = np.max(welfare_grid)

        # Equilibrium welfare (Eq 66): Use final 10% average
        equilibrium_window = welfare_traj[-max(1, len(welfare_traj) // 10) :]
        welfare_equilibrium = equilibrium_window.mean()

        # Welfare loss (Eq 67)
        welfare_loss = welfare_optimal - welfare_equilibrium

        return WelfareAnalysisCorrected(
            welfare_trajectory=welfare_traj,
            welfare_optimal=welfare_optimal,
            welfare_equilibrium=welfare_equilibrium,
            welfare_loss=welfare_loss,
            Q_bar_trajectories=Q_bar_traj,
            ex_ante_welfare=ex_ante_traj,
        )

    def compute_welfare_at_equilibrium(
        self, p_star: float, theta: np.ndarray, gamma: float
    ) -> float:
        """
        Compute welfare at given equilibrium.

        Args:
            p_star: Equilibrium market share
            theta: (N, 2) preferences
            gamma: Individual taste weight

        Returns:
            W(p*): Welfare at equilibrium

        Use Case:
            Compare welfare across different equilibria.
        """
        q1 = Q(theta[:, 0], p_star, gamma)
        q2 = Q(theta[:, 1], 1 - p_star, gamma)

        # Assume rational choice: each consumer picks higher utility
        Q_bar_1 = q1[q1 >= q2].mean() if (q1 >= q2).any() else q1.mean()
        Q_bar_2 = q2[q2 > q1].mean() if (q2 > q1).any() else q2.mean()

        W = p_star * Q_bar_1 + (1 - p_star) * Q_bar_2
        return W

    def compare_welfare_methods(
        self,
        p_trajectory: np.ndarray,
        theta: np.ndarray,
        gamma: float,
        choices: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compare old vs new welfare calculation methods.

        Args:
            p_trajectory: Market share trajectory
            theta: Preferences
            gamma: Parameter
            choices: Choice history

        Returns:
            Dictionary with comparison statistics

        Purpose:
            Validate that new method (Eq 64) gives different/correct results.
        """
        # New method (Eq 64 compliant)
        welfare_new = self.compute_welfare(p_trajectory, theta, gamma, choices)

        # Old method (ex-ante)
        T = len(p_trajectory)
        welfare_old = np.zeros(T)
        for t in range(T):
            p1 = p_trajectory[t]
            p2 = 1.0 - p1
            q1 = Q(theta[:, 0], p1, gamma)
            q2 = Q(theta[:, 1], p2, gamma)
            welfare_old[t] = np.maximum(q1, q2).mean()

        return {
            "new_method_equilibrium": welfare_new.welfare_equilibrium,
            "old_method_equilibrium": welfare_old[-1],
            "difference": welfare_new.welfare_equilibrium - welfare_old[-1],
            "new_method_trajectory": welfare_new.welfare_trajectory,
            "old_method_trajectory": welfare_old,
            "ex_ante_trajectory": welfare_new.ex_ante_welfare,
        }
