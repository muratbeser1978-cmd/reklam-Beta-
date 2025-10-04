"""
Thompson Sampling for Consumer Choice.

Implements Thompson Sampling algorithm for bandit learning.
"""
import numpy as np
from src.simulation.utilities import Q


def thompson_sample(consumer_state, market_state, rng, gamma):
    """
    Thompson Sampling for all consumers.

    Args:
        consumer_state: Current consumer state
        market_state: Current market state
        rng: Random number generator
        gamma: Individual taste weight

    Returns:
        np.ndarray: (N,) array of choices {0, 1}

    Algorithm:
        1. Draw θ̃ᵢ,ₐ ~ Beta(αᵢ,ₐ, βᵢ,ₐ) for all i, a  [Eq 23]
        2. Compute q̃ᵢ,ₐ = Q(θ̃ᵢ,ₐ, pₐ, γ) for each brand  [Eq 24]
        3. Choose cᵢ = argmax_a q̃ᵢ,ₐ  [Eq 25]
    """
    N = consumer_state.theta.shape[0]

    # Step 1: Thompson sampling - draw from posterior
    # θ̃ ~ Beta(α, β) for all consumers and both brands
    theta_samples = rng.beta(consumer_state.alpha, consumer_state.beta_param)  # (N, 2)

    # Step 2: Compute utilities for both brands
    # Brand 0 (index 0): market share p1
    # Brand 1 (index 1): market share p2
    q_tilde = np.zeros((N, 2))
    q_tilde[:, 0] = Q(theta_samples[:, 0], market_state.p1, gamma)
    q_tilde[:, 1] = Q(theta_samples[:, 1], market_state.p2, gamma)

    # Step 3: Choose brand with highest utility
    choices = np.argmax(q_tilde, axis=1)  # (N,)

    return choices
