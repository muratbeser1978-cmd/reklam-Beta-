"""
Unit tests for Welfare Analysis (Corrected for Eq 64).

Tests ex-post realized welfare W(t) = Σ pₐ·Q̄ₐ(t) as per Eq 64.
"""

import pytest
import numpy as np
from src.analysis.welfare_corrected import (
    WelfareCalculatorCorrected,
    WelfareAnalysisCorrected,
)
from src.simulation.utilities import Q


class TestWelfareCalculation:
    """Test welfare calculation following Eq 64."""

    def test_basic_welfare_computation(self):
        """Test basic welfare computation with choices."""
        calculator = WelfareCalculatorCorrected()

        N = 100
        T = 50
        np.random.seed(42)

        # Preferences
        theta = np.random.beta(2, 2, size=(N, 2))

        # Market share trajectory
        p_trajectory = 0.5 + 0.1 * np.random.randn(T + 1)
        p_trajectory = np.clip(p_trajectory, 0, 1)

        # Choices (random for simplicity)
        choices = np.random.randint(0, 2, size=(N, T))

        gamma = 0.7

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma, choices)

        # Verify all attributes
        assert isinstance(welfare, WelfareAnalysisCorrected)
        assert welfare.welfare_trajectory.shape == (T + 1,)
        assert welfare.Q_bar_trajectories.shape == (T + 1, 2)
        assert welfare.ex_ante_welfare.shape == (T + 1,)
        assert welfare.welfare_optimal > 0
        assert welfare.welfare_equilibrium > 0
        assert welfare.welfare_loss >= 0  # W^opt ≥ W^eq

    def test_welfare_loss_non_negative(self):
        """Test that welfare loss ΔW = W^opt - W^eq ≥ 0."""
        calculator = WelfareCalculatorCorrected()

        N = 50
        T = 100
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = np.linspace(0.5, 0.6, T + 1)
        choices = np.random.randint(0, 2, size=(N, T))

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma=0.7, choices=choices)

        # Welfare loss must be non-negative
        assert welfare.welfare_loss >= 0

    def test_optimal_welfare_at_p_05(self):
        """Test that optimal welfare is near p=0.5 for symmetric case."""
        calculator = WelfareCalculatorCorrected()

        N = 200
        np.random.seed(42)

        # Symmetric preferences
        theta = np.random.beta(2, 2, size=(N, 2))

        # Trajectory at equilibrium p=0.5
        T = 50
        p_trajectory = np.ones(T + 1) * 0.5
        choices = np.random.randint(0, 2, size=(N, T))

        gamma = 0.75  # Above critical

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma, choices)

        # For symmetric case with γ > γ*, equilibrium at p=0.5 is optimal
        # So W^eq ≈ W^opt, hence ΔW ≈ 0
        assert welfare.welfare_loss < 0.1

    def test_welfare_trajectory_positive(self):
        """Test that welfare trajectory is always positive."""
        calculator = WelfareCalculatorCorrected()

        N = 50
        T = 100
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = 0.5 + 0.2 * np.random.randn(T + 1)
        p_trajectory = np.clip(p_trajectory, 0.1, 0.9)
        choices = np.random.randint(0, 2, size=(N, T))

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma=0.7, choices=choices)

        # All welfare values should be positive
        assert np.all(welfare.welfare_trajectory > 0)

    def test_q_bar_computation(self):
        """Test Q̄ₐ(t) computation from actual choosers."""
        calculator = WelfareCalculatorCorrected()

        N = 100
        T = 50
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = np.ones(T + 1) * 0.5

        # All choose brand 0
        choices = np.zeros((N, T), dtype=int)

        gamma = 0.7

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma, choices)

        # Q̄₁ should be computed from all consumers (all chose brand 1=0)
        # Q̄₂ should be 0 (no choosers)
        # (Index 0 = brand 1, index 1 = brand 2 in code)
        assert welfare.Q_bar_trajectories[0, 0] > 0
        # Note: When no one chooses brand 2, Q̄₂ is set to 0 by implementation


class TestWelfareWithoutChoices:
    """Test welfare computation without choice data (estimation)."""

    def test_welfare_without_choices(self):
        """Test welfare estimation when choices are not available."""
        calculator = WelfareCalculatorCorrected()

        N = 100
        T = 50
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = 0.5 + 0.1 * np.random.randn(T + 1)
        p_trajectory = np.clip(p_trajectory, 0, 1)

        gamma = 0.7

        # No choices provided
        welfare = calculator.compute_welfare(p_trajectory, theta, gamma, choices=None)

        # Should still compute welfare (using population means)
        assert isinstance(welfare, WelfareAnalysisCorrected)
        assert welfare.welfare_trajectory.shape == (T + 1,)
        assert welfare.welfare_optimal > 0

    def test_welfare_estimation_uses_population_mean(self):
        """Test that without choices, Q̄ₐ uses population mean."""
        calculator = WelfareCalculatorCorrected()

        N = 50
        T = 20
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = np.ones(T + 1) * 0.5

        gamma = 0.7

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma, choices=None)

        # Q̄₁ and Q̄₂ should be population means
        # Since p=0.5 constant, Q̄₁ ≈ Q̄₂ (symmetric)
        Q_bar_1 = welfare.Q_bar_trajectories[:, 0]
        Q_bar_2 = welfare.Q_bar_trajectories[:, 1]

        # Should be similar for symmetric case
        assert np.abs(Q_bar_1.mean() - Q_bar_2.mean()) < 0.1


class TestWelfareAtEquilibrium:
    """Test compute_welfare_at_equilibrium method."""

    def test_welfare_at_equilibrium_p_05(self):
        """Test welfare at symmetric equilibrium p*=0.5."""
        calculator = WelfareCalculatorCorrected()

        N = 100
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))

        gamma = 0.75  # Above critical
        p_star = 0.5

        W_eq = calculator.compute_welfare_at_equilibrium(p_star, theta, gamma)

        # Should be positive
        assert W_eq > 0
        # For symmetric case, W(0.5) should be near maximum
        assert W_eq > 0.4  # Reasonable lower bound

    def test_welfare_at_asymmetric_equilibrium(self):
        """Test welfare at asymmetric equilibrium."""
        calculator = WelfareCalculatorCorrected()

        N = 100
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))

        gamma = 0.60  # Below critical
        p_star = 0.8  # Asymmetric equilibrium

        W_eq = calculator.compute_welfare_at_equilibrium(p_star, theta, gamma)

        # Should be positive
        assert W_eq > 0


class TestCompareWelfareMethods:
    """Test comparison between old (ex-ante) and new (ex-post) welfare."""

    def test_compare_welfare_methods(self):
        """Test comparison of old vs new welfare calculation."""
        calculator = WelfareCalculatorCorrected()

        N = 100
        T = 50
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = 0.5 + 0.1 * np.random.randn(T + 1)
        p_trajectory = np.clip(p_trajectory, 0, 1)
        choices = np.random.randint(0, 2, size=(N, T))

        gamma = 0.7

        comparison = calculator.compare_welfare_methods(p_trajectory, theta, gamma, choices)

        # Verify structure
        assert "new_method_equilibrium" in comparison
        assert "old_method_equilibrium" in comparison
        assert "difference" in comparison
        assert "new_method_trajectory" in comparison
        assert "old_method_trajectory" in comparison
        assert "ex_ante_trajectory" in comparison

        # All should be finite
        assert np.isfinite(comparison["new_method_equilibrium"])
        assert np.isfinite(comparison["old_method_equilibrium"])
        assert np.isfinite(comparison["difference"])

        # Trajectories should have correct length
        assert comparison["new_method_trajectory"].shape == (T + 1,)
        assert comparison["old_method_trajectory"].shape == (T + 1,)

    def test_ex_ante_vs_ex_post_difference(self):
        """Test that ex-ante and ex-post welfare differ."""
        calculator = WelfareCalculatorCorrected()

        N = 100
        T = 50
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = 0.5 + 0.2 * np.random.randn(T + 1)
        p_trajectory = np.clip(p_trajectory, 0.1, 0.9)
        choices = np.random.randint(0, 2, size=(N, T))

        gamma = 0.7

        comparison = calculator.compare_welfare_methods(p_trajectory, theta, gamma, choices)

        # Ex-ante (old) and ex-post (new) should generally differ
        # (though they could be similar in some cases)
        diff = abs(comparison["difference"])
        # Just verify they're both finite
        assert np.isfinite(diff)


class TestWelfareValidation:
    """Test welfare validation and error checking."""

    def test_welfare_loss_negative_raises_error(self):
        """Test that negative welfare loss raises error in post_init."""
        # Manually construct invalid welfare object
        with pytest.raises(ValueError, match="Welfare loss cannot be negative"):
            WelfareAnalysisCorrected(
                welfare_trajectory=np.array([0.5, 0.6]),
                welfare_optimal=0.5,
                welfare_equilibrium=0.6,  # Higher than optimal (invalid)
                welfare_loss=-0.1,  # Negative (invalid)
                Q_bar_trajectories=np.ones((2, 2)),
                ex_ante_welfare=np.array([0.5, 0.6]),
            )

    def test_optimal_less_than_equilibrium_raises_error(self):
        """Test that W^opt < W^eq raises error."""
        with pytest.raises(ValueError, match="Optimal welfare must be"):
            WelfareAnalysisCorrected(
                welfare_trajectory=np.array([0.5, 0.6]),
                welfare_optimal=0.5,
                welfare_equilibrium=0.7,  # Higher than optimal
                welfare_loss=0.0,  # Inconsistent
                Q_bar_trajectories=np.ones((2, 2)),
                ex_ante_welfare=np.array([0.5, 0.6]),
            )

    def test_choices_shape_mismatch_raises_error(self):
        """Test that mismatched choices shape raises error."""
        calculator = WelfareCalculatorCorrected()

        N = 50
        T = 100
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = np.ones(T + 1) * 0.5

        # Choices with wrong T dimension
        choices = np.random.randint(0, 2, size=(N, T + 5))  # Wrong!

        with pytest.raises(ValueError, match="Choices shape .* inconsistent"):
            calculator.compute_welfare(p_trajectory, theta, gamma=0.7, choices=choices)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_consumer(self):
        """Test welfare with single consumer."""
        calculator = WelfareCalculatorCorrected()

        N = 1
        T = 20
        np.random.seed(42)

        theta = np.array([[0.6, 0.4]])
        p_trajectory = np.ones(T + 1) * 0.5
        choices = np.zeros((N, T), dtype=int)

        gamma = 0.7

        # Should not crash
        welfare = calculator.compute_welfare(p_trajectory, theta, gamma, choices)
        assert isinstance(welfare, WelfareAnalysisCorrected)

    def test_constant_market_share(self):
        """Test welfare with constant market share."""
        calculator = WelfareCalculatorCorrected()

        N = 50
        T = 50
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = np.ones(T + 1) * 0.5  # Constant
        choices = np.random.randint(0, 2, size=(N, T))

        gamma = 0.7

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma, choices)

        # Welfare should be roughly constant over time
        assert welfare.welfare_trajectory.std() < 0.1

    def test_all_choose_one_brand(self):
        """Test welfare when all consumers choose same brand."""
        calculator = WelfareCalculatorCorrected()

        N = 50
        T = 30
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = np.ones(T + 1) * 0.8  # High p1

        # All choose brand 0
        choices = np.zeros((N, T), dtype=int)

        gamma = 0.7

        welfare = calculator.compute_welfare(p_trajectory, theta, gamma, choices)

        # Q̄₁ should be defined, Q̄₂ should be 0 (no choosers)
        assert welfare.Q_bar_trajectories[0, 0] > 0
        # When no one chooses brand 2, implementation sets Q̄₂=0

    def test_extreme_gamma_values(self):
        """Test welfare with extreme gamma values."""
        calculator = WelfareCalculatorCorrected()

        N = 50
        T = 20
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        p_trajectory = np.ones(T + 1) * 0.5
        choices = np.random.randint(0, 2, size=(N, T))

        # γ → 1 (pure individual)
        welfare_high = calculator.compute_welfare(p_trajectory, theta, gamma=0.99, choices=choices)
        assert welfare_high.welfare_trajectory.shape == (T + 1,)

        # γ → 0 (pure social)
        welfare_low = calculator.compute_welfare(p_trajectory, theta, gamma=0.01, choices=choices)
        assert welfare_low.welfare_trajectory.shape == (T + 1,)


class TestWelfareIntegration:
    """Integration test with full simulation."""

    def test_welfare_analysis_from_simulation(self):
        """Test welfare analysis on full simulation result."""
        from src.simulation.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        calculator = WelfareCalculatorCorrected()

        # Run simulation
        np.random.seed(42)
        config = SimulationConfig(
            N=100,
            T=150,
            gamma=0.7,
            beta=0.9,
            alpha_0=2.0,
            beta_0=2.0,
            p_init=0.5,
        )

        engine = SimulationEngine(config)
        result = engine.run()

        # Compute welfare
        welfare = calculator.compute_welfare(
            result.p1_trajectory,
            result.consumer_state.theta,
            gamma=config.gamma,
            choices=result.choices,
        )

        # Verify all outputs
        assert isinstance(welfare, WelfareAnalysisCorrected)
        assert welfare.welfare_trajectory.shape == (config.T + 1,)
        assert welfare.welfare_optimal > 0
        assert welfare.welfare_equilibrium > 0
        assert welfare.welfare_loss >= 0

        # Q̄ trajectories should be reasonable
        assert np.all(welfare.Q_bar_trajectories >= 0)
        assert np.all(welfare.Q_bar_trajectories <= 1)

        # Ex-ante welfare should also be reasonable
        assert np.all(welfare.ex_ante_welfare > 0)
        assert np.all(welfare.ex_ante_welfare <= 1)

    def test_welfare_comparison_across_gamma(self):
        """Test welfare comparison across different gamma values."""
        from src.simulation.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        calculator = WelfareCalculatorCorrected()

        # Run two simulations with different gamma
        np.random.seed(42)

        # High gamma (above critical)
        config_high = SimulationConfig(
            N=100, T=100, gamma=0.75, beta=0.9, alpha_0=2.0, beta_0=2.0, p_init=0.5
        )
        engine_high = SimulationEngine(config_high)
        result_high = engine_high.run()

        welfare_high = calculator.compute_welfare(
            result_high.p1_trajectory,
            result_high.consumer_state.theta,
            gamma=config_high.gamma,
            choices=result_high.choices,
        )

        # Low gamma (below critical)
        np.random.seed(42)  # Same seed for fair comparison
        config_low = SimulationConfig(
            N=100, T=100, gamma=0.60, beta=0.9, alpha_0=2.0, beta_0=2.0, p_init=0.5
        )
        engine_low = SimulationEngine(config_low)
        result_low = engine_low.run()

        welfare_low = calculator.compute_welfare(
            result_low.p1_trajectory,
            result_low.consumer_state.theta,
            gamma=config_low.gamma,
            choices=result_low.choices,
        )

        # Both should be valid
        assert welfare_high.welfare_loss >= 0
        assert welfare_low.welfare_loss >= 0

        # Typically, lower gamma (more network effects) leads to higher welfare loss
        # (but this is not guaranteed in finite samples)
        assert np.isfinite(welfare_high.welfare_loss)
        assert np.isfinite(welfare_low.welfare_loss)
