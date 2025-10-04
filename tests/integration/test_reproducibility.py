"""
Integration Test: Reproducibility (Scenario 1 from quickstart.md).

Verifies that identical configuration produces identical output.
"""
import pytest
import numpy as np
from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine


class TestReproducibility:
    """Test reproducibility: same seed → same output."""

    def test_same_config_same_output(self):
        """Scenario 1: Same config produces identical results."""
        config = SimulationConfig(N=100, T=100, gamma=0.75, beta=0.9, seed=42)

        engine = SimulationEngine()
        result1 = engine.run(config)
        result2 = engine.run(config)

        # Verify identical outputs
        assert np.array_equal(
            result1.p1_trajectory, result2.p1_trajectory
        ), "FAIL: Non-reproducible (different trajectories)"
        assert np.array_equal(
            result1.choices, result2.choices
        ), "FAIL: Non-reproducible (different choices)"
        assert np.array_equal(
            result1.rewards, result2.rewards
        ), "FAIL: Non-reproducible (different rewards)"

    def test_different_seeds_different_output(self):
        """Different seeds should produce different results."""
        config1 = SimulationConfig(N=100, T=100, gamma=0.75, beta=0.9, seed=42)
        config2 = SimulationConfig(N=100, T=100, gamma=0.75, beta=0.9, seed=123)

        engine = SimulationEngine()
        result1 = engine.run(config1)
        result2 = engine.run(config2)

        # Should be different
        assert not np.array_equal(
            result1.p1_trajectory, result2.p1_trajectory
        ), "Different seeds produced identical trajectories"
