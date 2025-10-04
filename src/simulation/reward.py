"""
Reward Generation.

Implements stochastic reward generation based on true preferences.
"""
import numpy as np
from src.simulation.utilities import Q


def generate_rewards(consumer_state, choices, market_state, gamma, rng):
    """
    Generate rewards based on true preferences (Eq 26).

    Args:
        consumer_state: Current consumer state
        choices: (N,) array of consumer choices {0, 1}
        market_state: Current market state
        gamma: Individual taste weight
        rng: Random number generator

    Returns:
        np.ndarray: (N,) array of rewards {0, 1}

    Equation:
        Eq 26: Rᵢ(t) ~ Bernoulli(qᵢ,cᵢ(t))
        where qᵢ,ₐ = Q(θᵢ,ₐ, pₐ, γ) is the true utility
    """
    N = len(choices)
    rewards = np.zeros(N, dtype=int)

    for i in range(N):
        a = choices[i]

        # True preference for chosen brand
        theta_i_a = consumer_state.theta[i, a]

        # Market share for chosen brand
        p_a = market_state.p1 if a == 0 else market_state.p2

        # True utility (reward probability)
        q_i_a = Q(theta_i_a, p_a, gamma)

        # Bernoulli reward
        rewards[i] = 1 if rng.random() < q_i_a else 0

    return rewards
