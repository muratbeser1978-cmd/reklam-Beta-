"""
Integration Test: Conservation Laws (Scenario 4 from quickstart.md).

Verifies mathematical invariants hold throughout simulation.
"""
import pytest
import numpy as np
from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine


class TestConservationLaws:
    """Test conservation laws and invariants."""

    def test_market_share_conservation(self):
        """
        Scenario 4: p₁ + p₂ = 1.0 always.

        Equations Tested:
            - Market share conservation: p₁ + p₂ = 1.0
            - Probability bounds: all probabilities ∈ [0, 1]
            - Positive parameters: α, β > 0
        """
        config = SimulationConfig(N=500, T=200, gamma=0.7, beta=0.85, seed=1024)

        engine = SimulationEngine()
        result = engine.run(config)

        # Test 1: Market shares sum to 1
        total = result.p1_trajectory + result.p2_trajectory
        max_error = np.abs(total - 1.0).max()

        assert max_error < 1e-10, (
            f"FAIL: Conservation violated. " f"Max error in p₁ + p₂ = 1.0 is {max_error:.2e}"
        )

        # Test 2: Market shares in [0, 1]
        assert np.all(
            (result.p1_trajectory >= 0) & (result.p1_trajectory <= 1)
        ), "FAIL: p1_trajectory out of bounds [0, 1]"

        assert np.all(
            (result.p2_trajectory >= 0) & (result.p2_trajectory <= 1)
        ), "FAIL: p2_trajectory out of bounds [0, 1]"

        # Test 3: Choices are {0, 1}
        assert np.all(
            (result.choices == 0) | (result.choices == 1)
        ), "FAIL: Invalid choice values (not in {0, 1})"

        # Test 4: Rewards are {0, 1}
        assert np.all(
            (result.rewards == 0) | (result.rewards == 1)
        ), "FAIL: Invalid reward values (not in {0, 1})"

    def test_conservation_across_parameter_ranges(self):
        """Conservation should hold for all valid parameter combinations."""
        engine = SimulationEngine()

        test_cases = [
            (0.5, 0.8),  # Mid gamma, high retention
            (0.9, 0.5),  # High gamma, mid retention
            (0.3, 0.95),  # Low gamma, very high retention
        ]

        for gamma, beta in test_cases:
            config = SimulationConfig(N=200, T=100, gamma=gamma, beta=beta, seed=42)
            result = engine.run(config)

            total = result.p1_trajectory + result.p2_trajectory
            max_error = np.abs(total - 1.0).max()

            assert max_error < 1e-10, (
                f"Conservation violated for gamma={gamma}, beta={beta}. "
                f"Max error = {max_error:.2e}"
            )
