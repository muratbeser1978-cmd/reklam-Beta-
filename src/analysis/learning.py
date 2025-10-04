"""
Learning Dynamics Analysis.

Analyzes Bayesian learning process and Thompson Sampling performance.

Key Metrics:
    - Regret: Suboptimality of choices
    - Information gain: KL divergence from prior
    - Exploration vs. exploitation balance
    - Lock-in detection: Consumers stuck on suboptimal brand

Theoretical Background:
    - Thompson Sampling is Bayesian optimal for multi-armed bandits
    - Regret bound: O(√T log T) for Beta-Bernoulli case
    - Information gain measures learning progress
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from scipy import stats
from scipy.special import betaln, digamma


@dataclass
class LearningMetrics:
    """
    Learning performance metrics.

    Attributes:
        cumulative_regret: Total regret up to time T
        instantaneous_regret: Regret at final timestep
        avg_posterior_variance: Mean posterior uncertainty
        exploration_rate: Fraction of exploratory choices
        information_gain: KL divergence from prior
        convergence_score: How close beliefs are to truth
        locked_in_fraction: Fraction of consumers locked-in
    """

    cumulative_regret: float
    instantaneous_regret: float
    avg_posterior_variance: float
    exploration_rate: float
    information_gain: float
    convergence_score: float
    locked_in_fraction: float


@dataclass
class ConsumerLearningProfile:
    """
    Individual consumer learning profile.

    Attributes:
        consumer_id: Consumer index
        true_theta: True preferences (θ₁, θ₂)
        posterior_mean: Posterior means (θ̂₁, θ̂₂)
        posterior_variance: Posterior variances
        experience: Number of trials per brand
        regret: Cumulative regret
        locked_in: Whether consumer is locked-in
        preferred_brand: True optimal brand (0 or 1)
        chosen_brand: Currently preferred brand
    """

    consumer_id: int
    true_theta: np.ndarray  # (2,)
    posterior_mean: np.ndarray  # (2,)
    posterior_variance: np.ndarray  # (2,)
    experience: np.ndarray  # (2,)
    regret: float
    locked_in: bool
    preferred_brand: int
    chosen_brand: int


class LearningAnalyzer:
    """
    Analyze Bayesian learning dynamics.

    Usage:
        analyzer = LearningAnalyzer()
        metrics = analyzer.compute_metrics(result, consumer_state)
        profiles = analyzer.analyze_consumer_profiles(consumer_state)
    """

    def __init__(self, alpha_0: float = 2.0, beta_0: float = 2.0):
        """
        Initialize analyzer.

        Args:
            alpha_0: Prior success parameter
            beta_0: Prior failure parameter
        """
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

    def compute_metrics(
        self,
        consumer_state,
        choices: np.ndarray,
        rewards: np.ndarray,
        market_state,
        gamma: float,
    ) -> LearningMetrics:
        """
        Compute all learning metrics.

        Args:
            consumer_state: ConsumerState object with beliefs
            choices: (N, T) array of choices over time
            rewards: (N, T) array of rewards over time
            market_state: MarketState trajectory
            gamma: Individual taste weight

        Returns:
            LearningMetrics object

        Example:
            >>> from src.simulation.result import SimulationResult
            >>> metrics = analyzer.compute_metrics(
            ...     result.consumer_state,
            ...     result.choices,
            ...     result.rewards,
            ...     result.market_state,
            ...     gamma=0.7
            ... )
        """
        N, T = choices.shape

        # 1. Cumulative regret
        cumulative_regret = self._compute_cumulative_regret(
            consumer_state, choices, rewards, market_state, gamma
        )

        # 2. Instantaneous regret (last timestep)
        instantaneous_regret = self._compute_instantaneous_regret(
            consumer_state, market_state, gamma
        )

        # 3. Posterior variance (measure of uncertainty)
        avg_posterior_variance = self._compute_avg_posterior_variance(consumer_state)

        # 4. Exploration rate
        exploration_rate = self._estimate_exploration_rate(consumer_state)

        # 5. Information gain (KL from prior)
        information_gain = self._compute_information_gain(consumer_state)

        # 6. Convergence score (distance from truth)
        convergence_score = self._compute_convergence_score(consumer_state)

        # 7. Lock-in fraction
        locked_in_fraction = self._compute_locked_in_fraction(consumer_state)

        return LearningMetrics(
            cumulative_regret=cumulative_regret,
            instantaneous_regret=instantaneous_regret,
            avg_posterior_variance=avg_posterior_variance,
            exploration_rate=exploration_rate,
            information_gain=information_gain,
            convergence_score=convergence_score,
            locked_in_fraction=locked_in_fraction,
        )

    def analyze_consumer_profiles(
        self, consumer_state, top_k: int = 10
    ) -> list[ConsumerLearningProfile]:
        """
        Analyze individual consumer learning profiles.

        Args:
            consumer_state: ConsumerState object
            top_k: Number of consumers to analyze in detail

        Returns:
            List of ConsumerLearningProfile objects

        Example:
            >>> profiles = analyzer.analyze_consumer_profiles(consumer_state, top_k=5)
            >>> for prof in profiles:
            ...     print(f"Consumer {prof.consumer_id}: Regret={prof.regret:.2f}")
        """
        N = consumer_state.theta.shape[0]
        profiles = []

        for i in range(min(N, top_k)):
            # True preferences
            true_theta = consumer_state.theta[i]
            posterior_mean = consumer_state.theta_hat[i]

            # Posterior variance: Var[Beta(α,β)] = αβ/((α+β)²(α+β+1))
            alpha = consumer_state.alpha[i]
            beta_param = consumer_state.beta_param[i]
            posterior_var = (alpha * beta_param) / (
                (alpha + beta_param) ** 2 * (alpha + beta_param + 1)
            )

            # Experience
            experience = consumer_state.n_eff[i]

            # Regret (computed approximately)
            optimal_brand = int(true_theta[0] < true_theta[1])
            current_brand = int(posterior_mean[0] < posterior_mean[1])
            regret = abs(true_theta[optimal_brand] - true_theta[current_brand])

            # Lock-in detection
            locked_in = self._is_locked_in(alpha[current_brand], beta_param[current_brand])

            profiles.append(
                ConsumerLearningProfile(
                    consumer_id=i,
                    true_theta=true_theta,
                    posterior_mean=posterior_mean,
                    posterior_variance=posterior_var,
                    experience=experience,
                    regret=regret,
                    locked_in=locked_in,
                    preferred_brand=optimal_brand,
                    chosen_brand=current_brand,
                )
            )

        return profiles

    def compute_regret_trajectory(
        self, consumer_state, choices: np.ndarray, market_state, gamma: float
    ) -> np.ndarray:
        """
        Compute regret over time.

        Args:
            consumer_state: ConsumerState object
            choices: (N, T) choices array
            market_state: MarketState trajectory
            gamma: Individual taste weight

        Returns:
            (T,) array of average regret at each timestep

        Use Case:
            Plot regret trajectory to visualize learning progress.
        """
        from src.simulation.utilities import Q

        N, T = choices.shape
        regret_trajectory = np.zeros(T)

        # This is a simplified computation - ideally needs full state history
        # For now, estimate from final state
        true_theta = consumer_state.theta  # (N, 2)

        # Compute optimal action for each consumer
        # (requires knowing market state at each t, simplified here)
        for t in range(T):
            # Placeholder: use current market state
            p1 = market_state.p1 if hasattr(market_state, "p1") else 0.5

            # Optimal utility for each brand
            q_brand_0 = Q(true_theta[:, 0], p1, gamma)
            q_brand_1 = Q(true_theta[:, 1], 1 - p1, gamma)

            # Optimal action
            optimal_actions = (q_brand_0 < q_brand_1).astype(int)

            # Actual actions
            actual_actions = choices[:, t]

            # Regret = utility of optimal - utility of actual
            regret = np.where(
                optimal_actions == 0,
                q_brand_0 - q_brand_1,  # Optimal was 0, chose differently
                q_brand_1 - q_brand_0,  # Optimal was 1, chose differently
            )
            regret[actual_actions == optimal_actions] = 0

            regret_trajectory[t] = regret.mean()

        return regret_trajectory

    def _compute_cumulative_regret(
        self, consumer_state, choices, rewards, market_state, gamma
    ) -> float:
        """
        Cumulative regret over all timesteps.

        Regret = Σₜ (reward_optimal - reward_actual)
        """
        # Simplified: use final state
        # Proper implementation needs full trajectory
        N, T = choices.shape
        return T * self._compute_instantaneous_regret(consumer_state, market_state, gamma)

    def _compute_instantaneous_regret(self, consumer_state, market_state, gamma) -> float:
        """
        Instantaneous regret at current time.

        Regret = E[reward_optimal - reward_current]
        """
        from src.simulation.utilities import Q

        true_theta = consumer_state.theta
        N = true_theta.shape[0]

        # Current market state
        p1 = market_state.p1 if hasattr(market_state, "p1") else 0.5

        # Optimal utilities
        q0 = Q(true_theta[:, 0], p1, gamma)
        q1 = Q(true_theta[:, 1], 1 - p1, gamma)
        optimal_utility = np.maximum(q0, q1)

        # Current posterior means
        theta_hat = consumer_state.theta_hat

        # Current utilities (based on beliefs)
        q0_hat = Q(theta_hat[:, 0], p1, gamma)
        q1_hat = Q(theta_hat[:, 1], 1 - p1, gamma)
        current_utility = np.maximum(q0_hat, q1_hat)

        regret = optimal_utility - current_utility
        return regret.mean()

    def _compute_avg_posterior_variance(self, consumer_state) -> float:
        """
        Average posterior variance across all consumers and brands.
        """
        alpha = consumer_state.alpha
        beta_param = consumer_state.beta_param

        # Var[Beta(α,β)] = αβ/((α+β)²(α+β+1))
        var = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
        return var.mean()

    def _estimate_exploration_rate(self, consumer_state, threshold: float = 0.1) -> float:
        """
        Estimate fraction of exploratory choices.

        High posterior variance → exploration
        Low posterior variance → exploitation
        """
        alpha = consumer_state.alpha
        beta_param = consumer_state.beta_param

        # Posterior variance
        var = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))

        # Count high-variance consumers (exploring)
        exploring = var > threshold
        return exploring.mean()

    def _compute_information_gain(self, consumer_state) -> float:
        """
        Information gain: KL divergence from prior to posterior.

        KL(Beta(α,β) || Beta(α₀,β₀))
        """
        alpha = consumer_state.alpha
        beta_param = consumer_state.beta_param

        # KL divergence for Beta distributions
        kl = self._beta_kl_divergence(alpha, beta_param, self.alpha_0, self.beta_0)
        return kl.mean()

    def _beta_kl_divergence(self, alpha1, beta1, alpha0, beta0):
        """
        KL(Beta(α₁,β₁) || Beta(α₀,β₀)).

        Formula:
            KL = log[B(α₀,β₀)/B(α₁,β₁)] + (α₁-α₀)ψ(α₁) + (β₁-β₀)ψ(β₁)
                 - (α₁+β₁-α₀-β₀)ψ(α₁+β₁)

        where ψ is digamma function, B is beta function.
        """
        # Beta function: B(α,β) = Γ(α)Γ(β)/Γ(α+β) = exp(betaln(α,β))
        log_B_ratio = betaln(alpha0, beta0) - betaln(alpha1, beta1)

        # Digamma terms
        psi_alpha1 = digamma(alpha1)
        psi_beta1 = digamma(beta1)
        psi_sum1 = digamma(alpha1 + beta1)

        kl = (
            log_B_ratio
            + (alpha1 - alpha0) * psi_alpha1
            + (beta1 - beta0) * psi_beta1
            - (alpha1 + beta1 - alpha0 - beta0) * psi_sum1
        )

        return kl

    def _compute_convergence_score(self, consumer_state) -> float:
        """
        Convergence score: 1 - ||θ̂ - θ||/√2.

        Score ∈ [0, 1], higher = better convergence.
        """
        true_theta = consumer_state.theta
        theta_hat = consumer_state.theta_hat

        # L2 distance
        distance = np.linalg.norm(true_theta - theta_hat, axis=1)

        # Normalize by maximum possible distance (√2)
        normalized_distance = distance / np.sqrt(2)

        # Score: 1 - distance
        score = 1 - normalized_distance.mean()
        return max(0, min(1, score))

    def _compute_locked_in_fraction(
        self, consumer_state, certainty_threshold: float = 0.9
    ) -> float:
        """
        Fraction of consumers locked-in to a brand.

        Lock-in criterion:
            - High posterior certainty (low variance)
            - Committed to one brand
        """
        alpha = consumer_state.alpha
        beta_param = consumer_state.beta_param

        locked_in_count = 0
        N = alpha.shape[0]

        for i in range(N):
            # Check both brands
            for a in range(2):
                if self._is_locked_in(alpha[i, a], beta_param[i, a], certainty_threshold):
                    locked_in_count += 1
                    break  # Consumer is locked-in

        return locked_in_count / N

    def _is_locked_in(
        self, alpha: float, beta: float, certainty_threshold: float = 0.9
    ) -> bool:
        """
        Check if consumer is locked-in based on posterior.

        Criteria:
            1. Posterior mean > threshold (strong belief)
            2. Posterior variance < 0.01 (low uncertainty)
        """
        posterior_mean = alpha / (alpha + beta)
        posterior_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        return posterior_mean > certainty_threshold and posterior_var < 0.01
