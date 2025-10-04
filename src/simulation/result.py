"""
Simulation Result Storage.

Stores complete trajectory and metadata from a simulation run.
"""
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class SimulationResult:
    """
    Complete simulation trajectory and metadata.

    Fields:
        p1_trajectory: Market share p₁(t) for t ∈ [0, T] (T+1,)
        p2_trajectory: Market share p₂(t) = 1 - p₁(t) (T+1,)
        choices: Consumer choices cᵢ(t) ∈ {0, 1} (N, T)
        rewards: Consumer rewards Rᵢ(t) ∈ {0, 1} (N, T)
        config: Original configuration
        metadata: Runtime info (timestamp, git commit, etc.)
    """

    p1_trajectory: np.ndarray  # (T+1,)
    p2_trajectory: np.ndarray  # (T+1,)
    choices: np.ndarray  # (N, T)
    rewards: np.ndarray  # (N, T)
    config: "SimulationConfig"  # type: ignore
    metadata: dict

    def __post_init__(self):
        """Verify result invariants."""
        T = self.config.T
        N = self.config.N

        # Shape checks
        assert self.p1_trajectory.shape == (
            T + 1,
        ), f"p1_trajectory shape {self.p1_trajectory.shape} != ({T+1},)"
        assert self.p2_trajectory.shape == (
            T + 1,
        ), f"p2_trajectory shape {self.p2_trajectory.shape} != ({T+1},)"
        assert self.choices.shape == (N, T), f"choices shape {self.choices.shape} != ({N}, {T})"
        assert self.rewards.shape == (N, T), f"rewards shape {self.rewards.shape} != ({N}, {T})"

        # Conservation law: p₁ + p₂ = 1.0
        total = self.p1_trajectory + self.p2_trajectory
        max_error = np.abs(total - 1.0).max()
        assert max_error < 1e-10, f"Market shares do not sum to 1.0 (max error = {max_error})"

        # Range checks
        assert np.all(
            (self.p1_trajectory >= 0) & (self.p1_trajectory <= 1)
        ), "p1_trajectory contains values outside [0, 1]"
        assert np.all(
            (self.choices == 0) | (self.choices == 1)
        ), "choices contains values other than {0, 1}"

    def save(self, path: str):
        """
        Save result to disk (research.md decision 8).

        Args:
            path: Base path (without extension)

        Creates:
            {path}.npz: Array data (compressed)
            {path}.json: Metadata
        """
        base_path = Path(path).with_suffix('')
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Save arrays to .npz (compressed)
        np.savez_compressed(
            f"{base_path}.npz",
            p1_trajectory=self.p1_trajectory,
            p2_trajectory=self.p2_trajectory,
            choices=self.choices,
            rewards=self.rewards,
        )

        # Save metadata to JSON
        metadata_full = {
            **self.metadata,
            "config": self.config.to_dict(),
        }
        with open(f"{base_path}.json", "w") as f:
            json.dump(metadata_full, f, indent=2)

    @staticmethod
    def load(path: str):
        """
        Load result from disk.

        Args:
            path: Base path (without extension)

        Returns:
            SimulationResult: Loaded result
        """
        from src.simulation.config import SimulationConfig

        base_path = Path(path).with_suffix('')

        # Load arrays
        data = np.load(f"{base_path}.npz")
        p1_trajectory = data["p1_trajectory"]
        p2_trajectory = data["p2_trajectory"]
        choices = data["choices"]
        rewards = data["rewards"]

        # Load metadata
        with open(f"{base_path}.json", "r") as f:
            metadata_full = json.load(f)

        config_dict = metadata_full.pop("config")
        config = SimulationConfig(**config_dict)

        return SimulationResult(
            p1_trajectory=p1_trajectory,
            p2_trajectory=p2_trajectory,
            choices=choices,
            rewards=rewards,
            config=config,
            metadata=metadata_full,
        )
