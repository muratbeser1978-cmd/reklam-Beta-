"""
Simulation Configuration.

Immutable configuration for a single simulation run.
"""
import hashlib
import json
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable simulation configuration.

    Equation References:
        - N: consumer population size
        - T: simulation timesteps
        - gamma: Eq 3 (individual taste weight γ ∈ [0,1])
        - beta: Eq 4 (retention probability β ∈ [0,1))
        - alpha_0: Eq 1 (prior success parameter, fixed at 2.0)
        - beta_0: Eq 2 (prior failure parameter, fixed at 2.0)
        - seed: random seed for reproducibility
        - p_init: Eq 13 (initial market share p₁(0))
    """

    N: int  # [100, 100000]
    T: int  # [1, 10000]
    gamma: float  # [0.0, 1.0]
    beta: float  # [0.0, 1.0)
    alpha_0: float = 2.0  # Fixed per specification
    beta_0: float = 2.0  # Fixed per specification
    seed: int = 42
    p_init: float = 0.5

    def __post_init__(self):
        """Validate configuration constraints."""
        if not (100 <= self.N <= 100_000):
            raise ValueError(f"N={self.N} out of range [100, 100000]")
        if not (1 <= self.T <= 10_000):
            raise ValueError(f"T={self.T} out of range [1, 10000]")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma={self.gamma} out of range [0, 1]")
        if not (0.0 <= self.beta < 1.0):
            raise ValueError(f"beta={self.beta} out of range [0, 1)")
        if self.alpha_0 != 2.0:
            raise ValueError("alpha_0 must be 2.0 (fixed per spec)")
        if self.beta_0 != 2.0:
            raise ValueError("beta_0 must be 2.0 (fixed per spec)")
        if not (0.0 <= self.p_init <= 1.0):
            raise ValueError(f"p_init={self.p_init} out of range [0, 1]")

    def create_rng(self) -> np.random.RandomState:
        """
        Create isolated RNG instance.

        Returns:
            np.random.RandomState: Random number generator with this config's seed

        Note:
            Uses legacy RandomState for cross-platform reproducibility.
            See research.md decision 3.
        """
        return np.random.RandomState(self.seed)

    def to_dict(self) -> dict:
        """Serialize configuration to dictionary."""
        return {
            "N": self.N,
            "T": self.T,
            "gamma": self.gamma,
            "beta": self.beta,
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
            "seed": self.seed,
            "p_init": self.p_init,
        }

    def to_hash(self) -> str:
        """
        Generate unique hash for this configuration.

        Returns:
            str: SHA256 hash (first 16 chars) for provenance tracking
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
