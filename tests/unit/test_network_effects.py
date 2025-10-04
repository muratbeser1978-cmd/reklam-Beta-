"""
Unit tests for Network Effects Analysis.

Tests social multiplier, utility decomposition, tipping points, and bandwagon effects.
"""

import pytest
import numpy as np
from src.analysis.network_effects import (
    NetworkEffectsAnalyzer,
    UtilityDecomposition,
    SocialMultiplierAnalysis,
    TippingAnalysis,
    estimate_critical_mass,
)
from src.simulation.consumer import ConsumerState
from src.simulation.market import MarketState


class TestUtilityDecomposition:
    """Test utility decomposition into individual and social components."""

    def test_basic_decomposition(self):
        """Test basic utility decomposition."""
        analyzer = NetworkEffectsAnalyzer(gamma_star=12/17)

        # Create consumer state
        N = 100
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))
        consumer_state = ConsumerState(
            theta=theta,
            alpha=np.ones((N, 2)) * 5,
            beta_param=np.ones((N, 2)) * 5,
            theta_hat=theta,
            n_eff=np.ones((N, 2)) * 6,
        )

        market_state = MarketState(p1=0.6, p2=0.4)
        gamma = 0.7

        decomp = analyzer.decompose_utility(consumer_state, market_state, gamma)

        # Verify all attributes
        assert isinstance(decomp, UtilityDecomposition)
        assert decomp.individual_utility_mean > 0
        assert decomp.social_utility_mean > 0
        assert 0 <= decomp.individual_contribution <= 1
        assert 0 <= decomp.social_contribution <= 1
        # Contributions should sum to 1
        assert abs(decomp.individual_contribution + decomp.social_contribution - 1.0) < 0.01
        assert decomp.gamma == gamma

    def test_high_gamma_individual_dominant(self):
        """Test that high γ leads to individual-dominated utility."""
        analyzer = NetworkEffectsAnalyzer()

        N = 100
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))
        consumer_state = ConsumerState(
            theta=theta,
            alpha=np.ones((N, 2)) * 5,
            beta_param=np.ones((N, 2)) * 5,
            theta_hat=theta,
            n_eff=np.ones((N, 2)) * 6,
        )

        market_state = MarketState(p1=0.5, p2=0.5)
        gamma = 0.9  # High gamma

        decomp = analyzer.decompose_utility(consumer_state, market_state, gamma)

        # Individual contribution should dominate
        assert decomp.individual_contribution > 0.7

    def test_low_gamma_social_dominant(self):
        """Test that low γ leads to social-dominated utility."""
        analyzer = NetworkEffectsAnalyzer()

        N = 100
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))
        consumer_state = ConsumerState(
            theta=theta,
            alpha=np.ones((N, 2)) * 5,
            beta_param=np.ones((N, 2)) * 5,
            theta_hat=theta,
            n_eff=np.ones((N, 2)) * 6,
        )

        market_state = MarketState(p1=0.5, p2=0.5)
        gamma = 0.3  # Low gamma

        decomp = analyzer.decompose_utility(consumer_state, market_state, gamma)

        # Social contribution should dominate
        assert decomp.social_contribution > 0.5


class TestSocialMultiplier:
    """Test social multiplier M(γ) = (1-γ)/γ analysis."""

    def test_social_multiplier_computation(self):
        """Test social multiplier calculation."""
        analyzer = NetworkEffectsAnalyzer()

        gamma = 0.6
        mult = analyzer.compute_social_multiplier(gamma)

        assert isinstance(mult, SocialMultiplierAnalysis)
        assert mult.gamma == gamma
        # M(0.6) = 0.4/0.6 = 2/3 ≈ 0.667
        assert abs(mult.multiplier - 2/3) < 0.01
        assert mult.regime in ["individual_dominant", "social_dominant", "balanced"]
        assert mult.amplification_factor > 0

    def test_gamma_05_balanced(self):
        """Test that γ=0.5 gives balanced multiplier M=1."""
        analyzer = NetworkEffectsAnalyzer()

        mult = analyzer.compute_social_multiplier(gamma=0.5)

        # M(0.5) = 0.5/0.5 = 1.0
        assert abs(mult.multiplier - 1.0) < 0.01
        assert mult.regime == "balanced"

    def test_gamma_high_individual_dominant(self):
        """Test high γ gives individual-dominant regime."""
        analyzer = NetworkEffectsAnalyzer()

        mult = analyzer.compute_social_multiplier(gamma=0.8)

        # M(0.8) = 0.2/0.8 = 0.25 < 1
        assert mult.multiplier < 1
        assert mult.regime == "individual_dominant"

    def test_gamma_low_social_dominant(self):
        """Test low γ gives social-dominant regime."""
        analyzer = NetworkEffectsAnalyzer()

        mult = analyzer.compute_social_multiplier(gamma=0.3)

        # M(0.3) = 0.7/0.3 ≈ 2.33 > 1
        assert mult.multiplier > 1
        assert mult.regime == "social_dominant"

    def test_gamma_zero_infinite_multiplier(self):
        """Test γ=0 gives infinite multiplier."""
        analyzer = NetworkEffectsAnalyzer()

        mult = analyzer.compute_social_multiplier(gamma=0.0)

        assert mult.multiplier == np.inf
        assert mult.regime == "social_dominant"

    def test_gamma_one_zero_multiplier(self):
        """Test γ=1 gives zero multiplier."""
        analyzer = NetworkEffectsAnalyzer()

        mult = analyzer.compute_social_multiplier(gamma=1.0)

        assert mult.multiplier == 0.0
        assert mult.regime == "individual_dominant"


class TestTippingAnalysis:
    """Test tipping point and basin of attraction analysis."""

    def test_tipping_below_critical(self):
        """Test tipping point detection for γ < γ*."""
        analyzer = NetworkEffectsAnalyzer(gamma_star=12/17)

        # Below critical: expect tipping at p=0.5
        gamma = 0.65

        # Simulated outcomes
        initial_conditions = np.linspace(0.0, 1.0, 21)
        # Below 0.5 → converge to ~0.0, above 0.5 → converge to ~1.0
        final_outcomes = np.where(initial_conditions < 0.5, 0.1, 0.9)

        tipping = analyzer.analyze_tipping_point(
            initial_conditions, final_outcomes, gamma, threshold_tol=0.05
        )

        assert isinstance(tipping, TippingAnalysis)
        # Tipping threshold should be 0.5
        assert abs(tipping.tipping_threshold - 0.5) < 0.1
        assert tipping.hysteresis_detected
        assert tipping.gamma == gamma
        # Low and high equilibria should be different
        assert abs(tipping.above_threshold_outcome - tipping.below_threshold_outcome) > 0.5

    def test_no_tipping_above_critical(self):
        """Test no tipping for γ > γ*."""
        analyzer = NetworkEffectsAnalyzer(gamma_star=12/17)

        # Above critical: no tipping
        gamma = 0.75

        # All converge to 0.5
        initial_conditions = np.linspace(0.0, 1.0, 21)
        final_outcomes = np.ones(21) * 0.5

        tipping = analyzer.analyze_tipping_point(
            initial_conditions, final_outcomes, gamma
        )

        # No tipping threshold
        assert np.isnan(tipping.tipping_threshold)
        assert not tipping.hysteresis_detected
        # Both equilibria should be 0.5
        assert abs(tipping.below_threshold_outcome - 0.5) < 0.1
        assert abs(tipping.above_threshold_outcome - 0.5) < 0.1

    def test_basin_size_computation(self):
        """Test basin of attraction size computation."""
        analyzer = NetworkEffectsAnalyzer(gamma_star=12/17)

        gamma = 0.60

        # 30% converge to low, 70% converge to high
        initial_conditions = np.linspace(0, 1, 20)
        final_outcomes = np.where(initial_conditions < 0.3, 0.1, 0.9)

        tipping = analyzer.analyze_tipping_point(
            initial_conditions, final_outcomes, gamma
        )

        # Basin size (fraction leading to high equilibrium)
        assert 0.6 < tipping.basin_size < 0.8


class TestBandwagonEffect:
    """Test bandwagon effect measurement."""

    def test_strong_bandwagon_effect(self):
        """Test detection of strong bandwagon effect."""
        analyzer = NetworkEffectsAnalyzer()

        # Strong correlation: p(t) predicts choice(t+1)
        T = 100
        p_trajectory = np.linspace(0.3, 0.7, T)
        # Choice fraction follows market share with small lag
        choice_fractions = p_trajectory.copy()

        bandwagon = analyzer.measure_bandwagon_effect(p_trajectory, choice_fractions)

        # Should detect strong bandwagon (high correlation)
        assert bandwagon > 0.8

    def test_weak_bandwagon_effect(self):
        """Test detection of weak bandwagon effect."""
        analyzer = NetworkEffectsAnalyzer()

        # No correlation: random choices
        np.random.seed(42)
        T = 100
        p_trajectory = 0.5 + 0.1 * np.random.randn(T)
        choice_fractions = 0.5 + 0.1 * np.random.randn(T)

        bandwagon = analyzer.measure_bandwagon_effect(p_trajectory, choice_fractions)

        # Should detect weak bandwagon (low correlation)
        assert bandwagon < 0.5

    def test_bandwagon_short_trajectory(self):
        """Test handling of very short trajectory."""
        analyzer = NetworkEffectsAnalyzer()

        p_trajectory = np.array([0.5])
        choice_fractions = np.array([0.5])

        bandwagon = analyzer.measure_bandwagon_effect(p_trajectory, choice_fractions)

        # Should return 0 (not enough data)
        assert bandwagon == 0.0

    def test_bandwagon_mismatched_lengths_raises_error(self):
        """Test that mismatched trajectory lengths raise error."""
        analyzer = NetworkEffectsAnalyzer()

        p_trajectory = np.array([0.5, 0.6, 0.7])
        choice_fractions = np.array([0.5, 0.6])

        with pytest.raises(ValueError, match="must have same length"):
            analyzer.measure_bandwagon_effect(p_trajectory, choice_fractions)


class TestSocialWelfareDecomposition:
    """Test social welfare decomposition into individual and network components."""

    def test_welfare_decomposition(self):
        """Test welfare decomposition."""
        analyzer = NetworkEffectsAnalyzer()

        N = 100
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))
        consumer_state = ConsumerState(
            theta=theta,
            alpha=np.ones((N, 2)) * 5,
            beta_param=np.ones((N, 2)) * 5,
            theta_hat=theta,
            n_eff=np.ones((N, 2)) * 6,
        )

        market_state = MarketState(p1=0.5, p2=0.5)
        gamma = 0.7

        welfare_decomp = analyzer.compute_social_welfare_decomposition(
            consumer_state, market_state, gamma
        )

        # Verify structure
        assert "total_welfare" in welfare_decomp
        assert "individual_welfare" in welfare_decomp
        assert "network_welfare" in welfare_decomp
        assert "network_fraction" in welfare_decomp

        # All components should be non-negative
        assert welfare_decomp["total_welfare"] > 0
        assert welfare_decomp["individual_welfare"] >= 0
        assert welfare_decomp["network_welfare"] >= 0
        assert 0 <= welfare_decomp["network_fraction"] <= 1

    def test_high_gamma_low_network_welfare(self):
        """Test that high γ leads to low network welfare contribution."""
        analyzer = NetworkEffectsAnalyzer()

        N = 100
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))
        consumer_state = ConsumerState(
            theta=theta,
            alpha=np.ones((N, 2)) * 5,
            beta_param=np.ones((N, 2)) * 5,
            theta_hat=theta,
            n_eff=np.ones((N, 2)) * 6,
        )

        market_state = MarketState(p1=0.5, p2=0.5)
        gamma = 0.9  # High individual weight

        welfare_decomp = analyzer.compute_social_welfare_decomposition(
            consumer_state, market_state, gamma
        )

        # Network fraction should be low
        assert welfare_decomp["network_fraction"] < 0.3


class TestCriticalMassEstimation:
    """Test critical mass estimation function."""

    def test_estimate_critical_mass_below_gamma_star(self):
        """Test critical mass estimation for γ < γ*."""
        # Simulate tipping thresholds below critical
        gamma_values = np.array([0.60, 0.62, 0.64, 0.66, 0.68])
        tipping_thresholds = np.array([0.50, 0.51, 0.49, 0.50, 0.50])  # Around 0.5

        avg_critical_mass, fitted_curve = estimate_critical_mass(
            gamma_values, tipping_thresholds
        )

        # Average should be around 0.5
        assert 0.45 < avg_critical_mass < 0.55
        # Fitted curve should be constant 0.5 for all γ < γ*
        assert np.all(fitted_curve == 0.5)

    def test_estimate_critical_mass_above_gamma_star(self):
        """Test critical mass estimation for γ > γ*."""
        gamma_star = 12/17

        # Above critical: no tipping threshold (NaN)
        gamma_values = np.array([0.72, 0.75, 0.80])
        tipping_thresholds = np.array([np.nan, np.nan, np.nan])

        avg_critical_mass, fitted_curve = estimate_critical_mass(
            gamma_values, tipping_thresholds
        )

        # Should return NaN (no valid data)
        assert np.isnan(avg_critical_mass)
        # Fitted curve should be NaN for γ > γ*
        assert np.all(np.isnan(fitted_curve))

    def test_estimate_critical_mass_mixed(self):
        """Test critical mass with mixed γ values."""
        gamma_values = np.array([0.60, 0.65, 0.70, 0.75, 0.80])
        # Below γ*: 0.5, above γ*: NaN
        gamma_star = 12/17
        tipping_thresholds = np.where(
            gamma_values < gamma_star,
            0.5,
            np.nan
        )

        avg_critical_mass, fitted_curve = estimate_critical_mass(
            gamma_values, tipping_thresholds
        )

        # Average should be 0.5 (from valid values)
        assert 0.45 < avg_critical_mass < 0.55
        # Fitted: 0.5 for γ<γ*, NaN for γ>γ*
        assert fitted_curve[0] == 0.5
        assert np.isnan(fitted_curve[-1])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_variance_utility(self):
        """Test decomposition with zero variance in preferences."""
        analyzer = NetworkEffectsAnalyzer()

        N = 100
        # All consumers identical
        theta = np.ones((N, 2)) * 0.5
        consumer_state = ConsumerState(
            theta=theta,
            alpha=np.ones((N, 2)) * 5,
            beta_param=np.ones((N, 2)) * 5,
            theta_hat=theta,
            n_eff=np.ones((N, 2)) * 6,
        )

        market_state = MarketState(p1=0.5, p2=0.5)
        gamma = 0.7

        # Should not crash
        decomp = analyzer.decompose_utility(consumer_state, market_state, gamma)
        assert isinstance(decomp, UtilityDecomposition)

    def test_market_state_without_p1_attribute(self):
        """Test handling when market_state lacks p1 attribute."""
        analyzer = NetworkEffectsAnalyzer()

        N = 50
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))
        consumer_state = ConsumerState(
            theta=theta,
            alpha=np.ones((N, 2)) * 5,
            beta_param=np.ones((N, 2)) * 5,
            theta_hat=theta,
            n_eff=np.ones((N, 2)) * 6,
        )

        # Market state without p1 attribute
        market_state = type('obj', (object,), {})()

        # Should default to p1=0.5
        decomp = analyzer.decompose_utility(consumer_state, market_state, gamma=0.7)
        assert isinstance(decomp, UtilityDecomposition)


class TestIntegration:
    """Integration tests for network effects analysis."""

    def test_full_network_effects_workflow(self):
        """Test complete network effects analysis workflow."""
        from src.simulation.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        analyzer = NetworkEffectsAnalyzer(gamma_star=12/17)

        # Run simulation
        np.random.seed(42)
        config = SimulationConfig(
            N=100,
            T=150,
            gamma=0.65,  # Below critical
            beta=0.9,
            alpha_0=2.0,
            beta_0=2.0,
            p_init=0.6,  # Start above tipping threshold
        )

        engine = SimulationEngine(config)
        result = engine.run()

        # 1. Utility decomposition
        decomp = analyzer.decompose_utility(
            result.consumer_state, result.market_state, config.gamma
        )
        assert isinstance(decomp, UtilityDecomposition)
        assert 0 <= decomp.individual_contribution <= 1

        # 2. Social multiplier
        mult = analyzer.compute_social_multiplier(config.gamma)
        assert isinstance(mult, SocialMultiplierAnalysis)
        assert mult.multiplier > 0

        # 3. Welfare decomposition
        welfare_decomp = analyzer.compute_social_welfare_decomposition(
            result.consumer_state, result.market_state, config.gamma
        )
        assert welfare_decomp["total_welfare"] > 0

        # 4. Bandwagon effect
        # Create choice fractions from market shares
        choice_fractions = result.p1_trajectory.copy()
        bandwagon = analyzer.measure_bandwagon_effect(
            result.p1_trajectory, choice_fractions
        )
        assert 0 <= bandwagon <= 1

    def test_tipping_analysis_multiple_runs(self):
        """Test tipping analysis across multiple initial conditions."""
        from src.simulation.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        analyzer = NetworkEffectsAnalyzer(gamma_star=12/17)

        # Run simulations from different initial conditions
        np.random.seed(42)
        gamma = 0.60  # Below critical

        initial_conditions = [0.2, 0.4, 0.5, 0.6, 0.8]
        final_outcomes = []

        for p_init in initial_conditions:
            config = SimulationConfig(
                N=50,
                T=100,
                gamma=gamma,
                beta=0.9,
                alpha_0=2.0,
                beta_0=2.0,
                p_init=p_init,
            )
            engine = SimulationEngine(config)
            result = engine.run()
            final_outcomes.append(result.p1_trajectory[-1])

        # Analyze tipping
        tipping = analyzer.analyze_tipping_point(
            initial_conditions, final_outcomes, gamma
        )

        assert isinstance(tipping, TippingAnalysis)
        # Should detect tipping dynamics for γ < γ*
        assert abs(tipping.tipping_threshold - 0.5) < 0.2 or np.isnan(tipping.tipping_threshold)
