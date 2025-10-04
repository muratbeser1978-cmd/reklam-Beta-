"""
Parameter Sweep Configuration.

Defines configuration for parameter sweeps across γ values and seeds.
"""
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List
from src.simulation.config import SimulationConfig


@dataclass
class SweepConfig:
    """
    Configuration for parameter sweep.

    Attributes:
        gamma_values: Grid of γ ∈ [γ_min, γ_max]
        seeds: Random seeds (for averaging across runs)
        base_config: Template SimulationConfig (N, T, beta fixed)
        output_dir: Directory for results
    """

    gamma_values: np.ndarray
    seeds: List[int]
    base_config: SimulationConfig
    output_dir: Path

    def __post_init__(self):
        """Validate sweep configuration."""
        if len(self.gamma_values) == 0:
            raise ValueError("gamma_values cannot be empty")
        if len(self.seeds) == 0:
            raise ValueError("seeds cannot be empty")
        if not all(0.0 <= g <= 1.0 for g in self.gamma_values):
            raise ValueError("All gamma values must be in [0, 1]")

    @property
    def total_runs(self) -> int:
        """Total number of simulation runs."""
        return len(self.gamma_values) * len(self.seeds)

    def generate_configs(self) -> List[SimulationConfig]:
        """
        Generate all simulation configurations for the sweep.

        Returns:
            List of SimulationConfig objects (gamma × seeds combinations)
        """
        configs = []

        for gamma in self.gamma_values:
            for seed in self.seeds:
                config = SimulationConfig(
                    N=self.base_config.N,
                    T=self.base_config.T,
                    gamma=gamma,
                    beta=self.base_config.beta,
                    alpha_0=self.base_config.alpha_0,
                    beta_0=self.base_config.beta_0,
                    seed=seed,
                    p_init=self.base_config.p_init,
                )
                configs.append(config)

        return configs

    @classmethod
    def create_gamma_sweep(
        cls,
        gamma_min: float,
        gamma_max: float,
        n_gamma: int,
        seeds: List[int],
        base_config: SimulationConfig,
        output_dir: str = "outputs/sweep",
    ):
        """
        Create sweep configuration for γ parameter.

        Args:
            gamma_min: Minimum γ value
            gamma_max: Maximum γ value
            n_gamma: Number of γ points
            seeds: Random seeds for averaging
            base_config: Base configuration (N, T, beta)
            output_dir: Output directory

        Returns:
            SweepConfig instance

        Example:
            >>> base = SimulationConfig(N=1000, T=500, gamma=0.5, beta=0.9, seed=42)
            >>> sweep = SweepConfig.create_gamma_sweep(
            ...     gamma_min=0.5, gamma_max=0.9, n_gamma=50,
            ...     seeds=[42, 123, 456], base_config=base
            ... )
            >>> print(f"Total runs: {sweep.total_runs}")  # 50 × 3 = 150
        """
        gamma_values = np.linspace(gamma_min, gamma_max, n_gamma)

        return cls(
            gamma_values=gamma_values,
            seeds=seeds,
            base_config=base_config,
            output_dir=Path(output_dir),
        )
