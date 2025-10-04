"""
Unit tests for Welfare calculations.

Verifies social welfare computations and welfare loss.
"""
import pytest
import numpy as np
from src.analysis.welfare import WelfareCalculator, WelfareAnalysis


class TestWelfareCalculator:
    """Test welfare calculation functions."""

    def test_optimal_greater_than_equilibrium(self):
        """W^opt ≥ W^eq always (Eq 65-66)."""
        calculator = WelfareCalculator()

        # Create test data
        N = 100
        T = 50
        theta = np.random.beta(2, 2, size=(N, 2))
        gamma = 0.7

        # Trajectory converging to asymmetric equilibrium
        p_trajectory = np.linspace(0.6, 0.65, T)

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma)

        assert welfare.welfare_optimal >= welfare.welfare_equilibrium - 1e-10, (
            f"Optimal welfare must be ≥ equilibrium welfare. "
            f"W_opt = {welfare.welfare_optimal:.4f}, "
            f"W_eq = {welfare.welfare_equilibrium:.4f}"
        )

    def test_welfare_loss_non_negative(self):
        """ΔW ≥ 0 always (Eq 67)."""
        calculator = WelfareCalculator()

        N = 100
        T = 50
        theta = np.random.beta(2, 2, size=(N, 2))
        gamma = 0.6

        # Various trajectories
        trajectories = [
            np.full(T, 0.5),  # Optimal (symmetric)
            np.full(T, 0.7),  # Asymmetric
            np.linspace(0.3, 0.7, T),  # Transitioning
        ]

        for p_traj in trajectories:
            welfare = calculator.compute_welfare(p_traj, theta, gamma)

            assert welfare.welfare_loss >= -1e-10, (
                f"Welfare loss cannot be negative: ΔW = {welfare.welfare_loss:.6f}"
            )

    def test_zero_welfare_loss_at_symmetric(self):
        """ΔW = 0 iff p* = 0.5 (symmetric equilibrium) (Eq 67)."""
        calculator = WelfareCalculator()

        N = 100
        T = 50
        theta = np.random.beta(2, 2, size=(N, 2))
        gamma = 0.7

        # Trajectory at symmetric equilibrium
        p_symmetric = np.full(T, 0.5)

        welfare = calculator.compute_welfare(p_symmetric, theta, gamma)

        # Welfare loss should be very small (near zero)
        assert welfare.welfare_loss < 0.01, (
            f"At symmetric equilibrium, welfare loss should be ≈0. "
            f"Got ΔW = {welfare.welfare_loss:.6f}"
        )

    def test_positive_welfare_loss_at_asymmetric(self):
        """ΔW > 0 for asymmetric equilibria."""
        calculator = WelfareCalculator()

        N = 100
        T = 50
        theta = np.random.beta(2, 2, size=(N, 2))
        gamma = 0.6

        # Trajectory at asymmetric equilibrium
        p_asymmetric = np.full(T, 0.7)

        welfare = calculator.compute_welfare(p_asymmetric, theta, gamma)

        # Welfare loss should be positive (departure from optimum)
        assert welfare.welfare_loss > 0.0, (
            f"At asymmetric equilibrium, welfare loss should be > 0. "
            f"Got ΔW = {welfare.welfare_loss:.6f}"
        )

    def test_welfare_trajectory_length(self):
        """Welfare trajectory should match input length."""
        calculator = WelfareCalculator()

        N = 50
        T = 100
        theta = np.random.beta(2, 2, size=(N, 2))
        gamma = 0.75
        p_trajectory = np.linspace(0.4, 0.6, T)

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma)

        assert len(welfare.welfare_trajectory) == T, (
            f"Welfare trajectory length {len(welfare.welfare_trajectory)} != {T}"
        )

    def test_welfare_values_bounded(self):
        """Welfare values should be bounded [0, 1] since utilities are in [0, 1]."""
        calculator = WelfareCalculator()

        N = 100
        T = 50
        theta = np.random.beta(2, 2, size=(N, 2))
        gamma = 0.7
        p_trajectory = np.random.uniform(0.3, 0.7, size=T)

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma)

        # Welfare is average of utilities, so should be in [0, 1]
        assert np.all((welfare.welfare_trajectory >= 0) & (welfare.welfare_trajectory <= 1)), (
            f"Welfare trajectory values out of bounds [0, 1]: "
            f"min={welfare.welfare_trajectory.min():.4f}, "
            f"max={welfare.welfare_trajectory.max():.4f}"
        )

        assert 0.0 <= welfare.welfare_optimal <= 1.0, (
            f"Optimal welfare out of bounds: {welfare.welfare_optimal:.4f}"
        )

        assert 0.0 <= welfare.welfare_equilibrium <= 1.0, (
            f"Equilibrium welfare out of bounds: {welfare.welfare_equilibrium:.4f}"
        )

    def test_welfare_dataclass_validation(self):
        """WelfareAnalysis should validate constraints."""
        # Valid case
        valid = WelfareAnalysis(
            welfare_trajectory=np.array([0.5, 0.6, 0.7]),
            welfare_optimal=0.8,
            welfare_equilibrium=0.7,
            welfare_loss=0.1,
        )
        assert valid.welfare_loss >= 0

        # Invalid: negative welfare loss
        with pytest.raises(ValueError, match="Welfare loss cannot be negative"):
            WelfareAnalysis(
                welfare_trajectory=np.array([0.5, 0.6, 0.7]),
                welfare_optimal=0.7,
                welfare_equilibrium=0.8,
                welfare_loss=-0.1,
            )

        # Invalid: optimal < equilibrium
        with pytest.raises(ValueError, match="Optimal welfare must be"):
            WelfareAnalysis(
                welfare_trajectory=np.array([0.5, 0.6, 0.7]),
                welfare_optimal=0.6,
                welfare_equilibrium=0.8,
                welfare_loss=0.0,  # Set to 0 to pass first check
            )
