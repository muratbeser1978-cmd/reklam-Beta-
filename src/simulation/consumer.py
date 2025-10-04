"""
Consumer State Management.

Tracks the complete state of all N consumers including preferences,
beliefs, and demographic status.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class ConsumerState:
    """
    Complete state of all N consumers at time t.

    Attributes:
        theta: True preferences θᵢ,ₐ (N, 2) - unknown to consumers (Eq 9-10)
        alpha: Posterior success parameters αᵢ,ₐ(t) (N, 2) (Eq 15)
        beta_param: Posterior failure parameters βᵢ,ₐ(t) (N, 2) (Eq 16)
        n_eff: Effective experience nᵢ,ₐ^eff(t) (N, 2) (Eq 29)
        theta_hat: Posterior means θ̂ᵢ,ₐ(t) (N, 2) (Eq 30)
        active: Consumer active status (N,) (Eq 62)

    Note:
        beta_param is named to avoid conflict with beta (retention parameter).
    """

    theta: np.ndarray  # (N, 2), true preferences
    alpha: np.ndarray  # (N, 2), posterior success counts
    beta_param: np.ndarray  # (N, 2), posterior failure counts
    n_eff: np.ndarray  # (N, 2), effective experience
    theta_hat: np.ndarray  # (N, 2), posterior means
    active: np.ndarray  # (N,), active status

    def __post_init__(self):
        """Validate consumer state invariants."""
        N = self.theta.shape[0]

        # Shape checks
        assert self.theta.shape == (N, 2), f"theta shape {self.theta.shape} != ({N}, 2)"
        assert self.alpha.shape == (N, 2), f"alpha shape {self.alpha.shape} != ({N}, 2)"
        assert self.beta_param.shape == (
            N,
            2,
        ), f"beta_param shape {self.beta_param.shape} != ({N}, 2)"
        assert self.n_eff.shape == (N, 2), f"n_eff shape {self.n_eff.shape} != ({N}, 2)"
        assert self.theta_hat.shape == (
            N,
            2,
        ), f"theta_hat shape {self.theta_hat.shape} != ({N}, 2)"
        assert self.active.shape == (N,), f"active shape {self.active.shape} != ({N},)"

        # Value constraints
        assert np.all(self.alpha > 0), "alpha must be > 0"
        assert np.all(self.beta_param > 0), "beta_param must be > 0"
        assert np.all((self.theta >= 0) & (self.theta <= 1)), "theta must be in [0, 1]"
        assert np.all(
            (self.theta_hat >= 0) & (self.theta_hat <= 1)
        ), "theta_hat must be in [0, 1]"

    @property
    def Z(self) -> np.ndarray:
        """
        Preference difference Z = θ₁ - θ₂ (Eq 11).

        Returns:
            np.ndarray: (N,) array of preference differences
        """
        return self.theta[:, 0] - self.theta[:, 1]

    def update_beliefs(self, choices: np.ndarray, rewards: np.ndarray):
        """
        Bayesian belief update (Eq 27-28).

        Args:
            choices: (N,) array of choices {0, 1}
            rewards: (N,) array of rewards {0, 1}

        Updates:
            - If reward=1: α_{i,a} += 1 (Eq 27a)
            - If reward=0: β_{i,a} += 1 (Eq 27b)
            - Recompute θ̂ = α/(α+β) (Eq 30)
            - Recompute n_eff = α + β - (α₀ + β₀) (Eq 29)
        """
        N = len(choices)
        for i in range(N):
            a = choices[i]
            if rewards[i] == 1:
                self.alpha[i, a] += 1
            else:
                self.beta_param[i, a] += 1

        # Update derived quantities
        self.theta_hat = self.alpha / (self.alpha + self.beta_param)
        # n_eff = total observations - prior observations
        alpha_0_beta_0_sum = self.alpha[0, 0] + self.beta_param[0, 0]  # Initial prior sum
        for i in range(N):
            for a in range(2):
                # Find initial prior sum (same for all consumers initially)
                initial_sum = 4.0  # alpha_0 + beta_0 = 2 + 2
                self.n_eff[i, a] = self.alpha[i, a] + self.beta_param[i, a] - initial_sum

    def apply_churn(self, beta_retention: float, rng: np.random.RandomState, alpha_0: float, beta_0: float):
        """
        Apply demographic churn (Eq 62).

        With probability (1-β), replace consumer with new one.

        Args:
            beta_retention: Retention probability β
            rng: Random number generator
            alpha_0: Prior success parameter
            beta_0: Prior failure parameter

        Updates:
            - Churned consumers: resample θ ~ Beta(α₀, β₀)
            - Reset beliefs: α = α₀, β = β₀
            - Mark as active
        """
        N = self.theta.shape[0]

        # Determine which consumers churn
        churn_prob = 1.0 - beta_retention
        churn_mask = rng.random(N) < churn_prob
        n_churned = churn_mask.sum()

        if n_churned > 0:
            # Resample preferences for churned consumers
            self.theta[churn_mask] = rng.beta(alpha_0, beta_0, size=(n_churned, 2))

            # Reset beliefs to prior
            self.alpha[churn_mask] = alpha_0
            self.beta_param[churn_mask] = beta_0
            self.theta_hat[churn_mask] = alpha_0 / (alpha_0 + beta_0)
            self.n_eff[churn_mask] = 0

            # Mark as active (replacement consumers)
            self.active[churn_mask] = True

    @staticmethod
    def initialize(N: int, alpha_0: float, beta_0: float, rng: np.random.RandomState):
        """
        Initialize consumer state (Eq 9-10, 15-16).

        Args:
            N: Number of consumers
            alpha_0: Prior success parameter
            beta_0: Prior failure parameter
            rng: Random number generator

        Returns:
            ConsumerState: Initialized state
        """
        # Sample true preferences θ ~ Beta(α₀, β₀)
        theta = rng.beta(alpha_0, beta_0, size=(N, 2))

        # Initialize beliefs at prior
        alpha = np.full((N, 2), alpha_0, dtype=float)
        beta_param = np.full((N, 2), beta_0, dtype=float)

        # Posterior means at prior
        theta_hat = np.full((N, 2), alpha_0 / (alpha_0 + beta_0), dtype=float)

        # No effective experience yet
        n_eff = np.zeros((N, 2), dtype=float)

        # All consumers active
        active = np.ones(N, dtype=bool)

        return ConsumerState(
            theta=theta,
            alpha=alpha,
            beta_param=beta_param,
            n_eff=n_eff,
            theta_hat=theta_hat,
            active=active,
        )
