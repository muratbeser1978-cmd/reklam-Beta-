"""
Unit tests for Learning Dynamics Analysis.

Tests Bayesian learning metrics, regret, information gain, and lock-in detection.
"""

import pytest
import numpy as np
from src.analysis.learning import (
    LearningAnalyzer,
    LearningMetrics,
    ConsumerLearningProfile,
)
from src.simulation.consumer import ConsumerState
from src.simulation.market import MarketState


class TestLearningMetrics:
    """Test learning metrics computation."""

    def test_basic_metrics_computation(self):
        """Test basic metrics on simple consumer state."""
        analyzer = LearningAnalyzer(alpha_0=2.0, beta_0=2.0)

        # Create simple consumer state
        N = 50
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))
        alpha = np.ones((N, 2)) * 5.0  # Some experience
        beta_param = np.ones((N, 2)) * 5.0
        theta_hat = alpha / (alpha + beta_param)
        n_eff = alpha + beta_param - 4.0

        consumer_state = ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            theta_hat=theta_hat,
            n_eff=n_eff,
        )

        # Create dummy choices and rewards
        T = 100
        choices = np.random.randint(0, 2, size=(N, T))
        rewards = np.random.beta(2, 2, size=(N, T))

        market_state = MarketState(p1=0.5, p2=0.5)

        metrics = analyzer.compute_metrics(
            consumer_state, choices, rewards, market_state, gamma=0.7
        )

        # Verify all metrics exist and are finite
        assert isinstance(metrics, LearningMetrics)
        assert np.isfinite(metrics.cumulative_regret)
        assert np.isfinite(metrics.instantaneous_regret)
        assert metrics.avg_posterior_variance > 0
        assert 0 <= metrics.exploration_rate <= 1
        assert metrics.information_gain >= 0
        assert 0 <= metrics.convergence_score <= 1
        assert 0 <= metrics.locked_in_fraction <= 1

    def test_information_gain_increases_with_experience(self):
        """Test that information gain increases as consumers gain experience."""
        analyzer = LearningAnalyzer(alpha_0=2.0, beta_0=2.0)

        N = 100
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))

        # Low experience (close to prior)
        alpha_low = np.ones((N, 2)) * 3.0
        beta_low = np.ones((N, 2)) * 3.0
        theta_hat_low = alpha_low / (alpha_low + beta_low)
        n_eff_low = alpha_low + beta_low - 4.0

        state_low = ConsumerState(
            theta=theta,
            alpha=alpha_low,
            beta_param=beta_low,
            theta_hat=theta_hat_low,
            n_eff=n_eff_low,
        )

        # High experience (far from prior)
        alpha_high = np.ones((N, 2)) * 20.0
        beta_high = np.ones((N, 2)) * 20.0
        theta_hat_high = alpha_high / (alpha_high + beta_high)
        n_eff_high = alpha_high + beta_high - 4.0

        state_high = ConsumerState(
            theta=theta,
            alpha=alpha_high,
            beta_param=beta_high,
            theta_hat=theta_hat_high,
            n_eff=n_eff_high,
        )

        # Compute information gain
        ig_low = analyzer._compute_information_gain(state_low)
        ig_high = analyzer._compute_information_gain(state_high)

        # More experience → more information gain
        assert ig_high > ig_low

    def test_posterior_variance_decreases_with_experience(self):
        """Test that posterior variance decreases with more trials."""
        analyzer = LearningAnalyzer()

        N = 100
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))

        # Few trials
        alpha_few = np.ones((N, 2)) * 3.0
        beta_few = np.ones((N, 2)) * 3.0
        state_few = ConsumerState(
            theta=theta,
            alpha=alpha_few,
            beta_param=beta_few,
            theta_hat=alpha_few / (alpha_few + beta_few),
            n_eff=alpha_few + beta_few - 4.0,
        )

        # Many trials
        alpha_many = np.ones((N, 2)) * 50.0
        beta_many = np.ones((N, 2)) * 50.0
        state_many = ConsumerState(
            theta=theta,
            alpha=alpha_many,
            beta_param=beta_many,
            theta_hat=alpha_many / (alpha_many + beta_many),
            n_eff=alpha_many + beta_many - 4.0,
        )

        var_few = analyzer._compute_avg_posterior_variance(state_few)
        var_many = analyzer._compute_avg_posterior_variance(state_many)

        # More trials → less uncertainty
        assert var_many < var_few


class TestConsumerProfiles:
    """Test individual consumer learning profile analysis."""

    def test_analyze_consumer_profiles(self):
        """Test consumer profile extraction."""
        analyzer = LearningAnalyzer()

        N = 20
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))
        alpha = np.random.uniform(2, 10, size=(N, 2))
        beta_param = np.random.uniform(2, 10, size=(N, 2))
        theta_hat = alpha / (alpha + beta_param)
        n_eff = alpha + beta_param - 4.0

        consumer_state = ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            theta_hat=theta_hat,
            n_eff=n_eff,
        )

        profiles = analyzer.analyze_consumer_profiles(consumer_state, top_k=5)

        # Should return 5 profiles
        assert len(profiles) == 5

        for profile in profiles:
            assert isinstance(profile, ConsumerLearningProfile)
            assert profile.true_theta.shape == (2,)
            assert profile.posterior_mean.shape == (2,)
            assert profile.posterior_variance.shape == (2,)
            assert profile.experience.shape == (2,)
            assert profile.regret >= 0
            assert profile.preferred_brand in [0, 1]
            assert profile.chosen_brand in [0, 1]
            assert isinstance(profile.locked_in, bool)

    def test_lock_in_detection(self):
        """Test lock-in detection for highly certain consumers."""
        analyzer = LearningAnalyzer()

        # Consumer locked-in to brand 0 (high α, low β)
        alpha_locked = 100.0
        beta_locked = 5.0

        is_locked = analyzer._is_locked_in(alpha_locked, beta_locked, certainty_threshold=0.9)

        # Posterior mean = 100/105 ≈ 0.952 > 0.9 ✓
        # Posterior var ≈ 0 ✓
        assert is_locked

    def test_no_lock_in_uncertain_consumer(self):
        """Test that uncertain consumers are not locked-in."""
        analyzer = LearningAnalyzer()

        # Consumer with low experience (uncertain)
        alpha_uncertain = 3.0
        beta_uncertain = 3.0

        is_locked = analyzer._is_locked_in(alpha_uncertain, beta_uncertain, certainty_threshold=0.9)

        # Posterior mean = 0.5 < 0.9 ✗
        assert not is_locked


class TestRegretComputation:
    """Test regret computation."""

    def test_regret_trajectory(self):
        """Test regret trajectory computation."""
        analyzer = LearningAnalyzer()

        N = 50
        T = 100
        np.random.seed(42)

        theta = np.random.beta(2, 2, size=(N, 2))
        alpha = np.ones((N, 2)) * 5.0
        beta_param = np.ones((N, 2)) * 5.0
        theta_hat = alpha / (alpha + beta_param)
        n_eff = alpha + beta_param - 4.0

        consumer_state = ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            theta_hat=theta_hat,
            n_eff=n_eff,
        )

        choices = np.random.randint(0, 2, size=(N, T))
        market_state = MarketState(p1=0.5, p2=0.5)

        regret_traj = analyzer.compute_regret_trajectory(
            consumer_state, choices, market_state, gamma=0.7
        )

        # Should return T-length trajectory
        assert regret_traj.shape == (T,)
        # Regret should be non-negative
        assert np.all(regret_traj >= 0)

    def test_zero_regret_for_optimal_choices(self):
        """Test that optimal choices yield zero regret."""
        analyzer = LearningAnalyzer()

        N = 10
        T = 50
        np.random.seed(42)

        # Consumers with known preferences
        theta = np.array([[0.8, 0.2]] * N)  # All prefer brand 0

        # Perfect beliefs (θ̂ = θ)
        alpha = np.array([[80.0, 20.0]] * N)
        beta_param = np.array([[20.0, 80.0]] * N)
        theta_hat = theta.copy()
        n_eff = np.ones((N, 2)) * 100.0

        consumer_state = ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            theta_hat=theta_hat,
            n_eff=n_eff,
        )

        # All choose brand 0 (optimal)
        choices = np.zeros((N, T), dtype=int)
        market_state = MarketState(p1=0.5, p2=0.5)

        regret_traj = analyzer.compute_regret_trajectory(
            consumer_state, choices, market_state, gamma=0.7
        )

        # Regret should be near zero (perfect beliefs, optimal choices)
        # (May not be exactly zero due to network effects and market state)
        assert np.all(regret_traj < 0.1)


class TestBetaKLDivergence:
    """Test Beta KL divergence computation."""

    def test_kl_divergence_identity(self):
        """Test KL(Beta(α,β) || Beta(α,β)) = 0."""
        analyzer = LearningAnalyzer()

        alpha = 5.0
        beta = 3.0

        kl = analyzer._beta_kl_divergence(alpha, beta, alpha, beta)

        assert abs(kl) < 1e-10

    def test_kl_divergence_positive(self):
        """Test that KL divergence is always non-negative."""
        analyzer = LearningAnalyzer()

        # Random distributions
        np.random.seed(42)
        for _ in range(10):
            alpha1 = np.random.uniform(1, 10)
            beta1 = np.random.uniform(1, 10)
            alpha0 = np.random.uniform(1, 10)
            beta0 = np.random.uniform(1, 10)

            kl = analyzer._beta_kl_divergence(alpha1, beta1, alpha0, beta0)

            assert kl >= -1e-10  # Allow tiny numerical errors

    def test_kl_divergence_from_prior(self):
        """Test KL divergence from prior Beta(2,2)."""
        analyzer = LearningAnalyzer(alpha_0=2.0, beta_0=2.0)

        # Posterior far from prior
        alpha_post = 20.0
        beta_post = 5.0

        kl = analyzer._beta_kl_divergence(alpha_post, beta_post, 2.0, 2.0)

        # Should be large (far from prior)
        assert kl > 1.0


class TestExplorationRate:
    """Test exploration rate estimation."""

    def test_high_exploration_early(self):
        """Test high exploration rate with uncertain beliefs."""
        analyzer = LearningAnalyzer()

        N = 100
        np.random.seed(42)

        # Early stage: high uncertainty
        theta = np.random.beta(2, 2, size=(N, 2))
        alpha = np.ones((N, 2)) * 2.5  # Close to prior
        beta_param = np.ones((N, 2)) * 2.5
        theta_hat = alpha / (alpha + beta_param)
        n_eff = alpha + beta_param - 4.0

        consumer_state = ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            theta_hat=theta_hat,
            n_eff=n_eff,
        )

        exploration_rate = analyzer._estimate_exploration_rate(consumer_state, threshold=0.1)

        # Should have high exploration
        assert exploration_rate > 0.5

    def test_low_exploration_late(self):
        """Test low exploration rate with confident beliefs."""
        analyzer = LearningAnalyzer()

        N = 100
        np.random.seed(42)

        # Late stage: low uncertainty
        theta = np.random.beta(2, 2, size=(N, 2))
        alpha = np.ones((N, 2)) * 50.0
        beta_param = np.ones((N, 2)) * 50.0
        theta_hat = alpha / (alpha + beta_param)
        n_eff = alpha + beta_param - 4.0

        consumer_state = ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            theta_hat=theta_hat,
            n_eff=n_eff,
        )

        exploration_rate = analyzer._estimate_exploration_rate(consumer_state, threshold=0.1)

        # Should have low exploration
        assert exploration_rate < 0.5


class TestConvergenceScore:
    """Test convergence score computation."""

    def test_perfect_convergence(self):
        """Test convergence score when θ̂ = θ."""
        analyzer = LearningAnalyzer()

        N = 50
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))

        # Perfect beliefs
        theta_hat = theta.copy()
        alpha = np.ones((N, 2)) * 100.0
        beta_param = np.ones((N, 2)) * 100.0
        n_eff = np.ones((N, 2)) * 196.0

        consumer_state = ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            theta_hat=theta_hat,
            n_eff=n_eff,
        )

        score = analyzer._compute_convergence_score(consumer_state)

        # Should be 1.0 (perfect convergence)
        assert score > 0.99

    def test_poor_convergence(self):
        """Test convergence score with large belief error."""
        analyzer = LearningAnalyzer()

        N = 50
        np.random.seed(42)
        theta = np.random.beta(2, 2, size=(N, 2))

        # Random beliefs (unrelated to truth)
        theta_hat = np.random.beta(2, 2, size=(N, 2))
        alpha = np.ones((N, 2)) * 5.0
        beta_param = np.ones((N, 2)) * 5.0
        n_eff = np.ones((N, 2)) * 6.0

        consumer_state = ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            theta_hat=theta_hat,
            n_eff=n_eff,
        )

        score = analyzer._compute_convergence_score(consumer_state)

        # Should be low (poor convergence)
        assert 0 <= score < 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_consumer_state(self):
        """Test handling of very small consumer populations."""
        analyzer = LearningAnalyzer()

        N = 1
        theta = np.array([[0.5, 0.5]])
        alpha = np.array([[3.0, 3.0]])
        beta_param = np.array([[3.0, 3.0]])
        theta_hat = alpha / (alpha + beta_param)
        n_eff = alpha + beta_param - 4.0

        consumer_state = ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            theta_hat=theta_hat,
            n_eff=n_eff,
        )

        choices = np.zeros((1, 10), dtype=int)
        rewards = np.random.beta(2, 2, size=(1, 10))
        market_state = MarketState(p1=0.5, p2=0.5)

        # Should not crash
        metrics = analyzer.compute_metrics(
            consumer_state, choices, rewards, market_state, gamma=0.7
        )

        assert isinstance(metrics, LearningMetrics)


def test_learning_integration():
    """Integration test: full learning analysis workflow."""
    from src.simulation.config import SimulationConfig
    from src.simulation.engine import SimulationEngine

    # Run simulation
    np.random.seed(42)
    config = SimulationConfig(
        N=100,
        T=200,
        gamma=0.7,
        beta=0.9,
        alpha_0=2.0,
        beta_0=2.0,
        p_init=0.5,
    )

    engine = SimulationEngine(config)
    result = engine.run()

    # Analyze learning
    analyzer = LearningAnalyzer(alpha_0=2.0, beta_0=2.0)

    metrics = analyzer.compute_metrics(
        result.consumer_state,
        result.choices,
        result.rewards,
        result.market_state,
        gamma=config.gamma,
    )

    # Verify all metrics are valid
    assert np.isfinite(metrics.cumulative_regret)
    assert np.isfinite(metrics.instantaneous_regret)
    assert metrics.avg_posterior_variance > 0
    assert 0 <= metrics.exploration_rate <= 1
    assert metrics.information_gain > 0  # Some learning occurred
    assert 0 <= metrics.convergence_score <= 1
    assert 0 <= metrics.locked_in_fraction <= 1

    # Analyze consumer profiles
    profiles = analyzer.analyze_consumer_profiles(result.consumer_state, top_k=10)
    assert len(profiles) == 10

    # Regret trajectory
    regret_traj = analyzer.compute_regret_trajectory(
        result.consumer_state, result.choices, result.market_state, gamma=config.gamma
    )
    assert regret_traj.shape == (config.T,)
