"""
Integration Test: Asymmetric Equilibrium (Scenario 3 from quickstart.md).

Verifies path-dependent convergence when γ < γ*.
"""
import pytest
import numpy as np
from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.distributions.constants import GAMMA_STAR


class TestAsymmetricEquilibrium:
    """Test asymmetric equilibrium and path dependence (γ < γ*)."""

    def test_path_dependence(self):
        """
        Scenario 3: γ < γ* → convergence depends on initial condition.

        Equations Tested:
            - Eq 43-47: Asymmetric equilibria p^± for γ < γ*
            - Eq 48-50: Basins of attraction, tipping point at 0.5
            - Eq 61: Path dependence
        """
        # γ = 0.60 < γ* = 0.706
        gamma_subcritical = 0.60

        assert gamma_subcritical < GAMMA_STAR, f"Test setup error: gamma should be < γ*"

        engine = SimulationEngine()

        # Run from p(0) > 0.5 (upper basin)
        config_high = SimulationConfig(
            N=500, T=500, gamma=gamma_subcritical, beta=0.9, seed=456, p_init=0.6
        )
        result_high = engine.run(config_high)
        p1_high = result_high.p1_trajectory[-1]

        # Run from p(0) < 0.5 (lower basin)
        config_low = SimulationConfig(
            N=500, T=500, gamma=gamma_subcritical, beta=0.9, seed=789, p_init=0.4
        )
        result_low = engine.run(config_low)
        p1_low = result_low.p1_trajectory[-1]

        # Verify asymmetry
        assert p1_high > 0.52, f"FAIL: Expected p⁺ > 0.52, got {p1_high:.4f}"
        assert p1_low < 0.48, f"FAIL: Expected p⁻ < 0.48, got {p1_low:.4f}"

        # Verify they're different equilibria
        assert abs(p1_high - p1_low) > 0.05, (
            f"FAIL: Expected distinct equilibria, got p⁺={p1_high:.4f}, p⁻={p1_low:.4f}"
        )

    def test_tipping_point_sensitivity(self):
        """Initial condition near 0.5 shows sensitivity."""
        gamma = 0.60  # < γ*
        engine = SimulationEngine()

        # Slightly above 0.5
        config_above = SimulationConfig(
            N=300, T=400, gamma=gamma, beta=0.9, seed=100, p_init=0.51
        )
        result_above = engine.run(config_above)

        # Slightly below 0.5
        config_below = SimulationConfig(
            N=300, T=400, gamma=gamma, beta=0.9, seed=200, p_init=0.49
        )
        result_below = engine.run(config_below)

        # Should diverge to different equilibria
        p1_above = result_above.p1_trajectory[-1]
        p1_below = result_below.p1_trajectory[-1]

        # May or may not show sensitivity (stochastic), but should not both be at 0.5
        # This is a weaker test due to stochasticity near tipping point
        assert not (
            abs(p1_above - 0.5) < 0.05 and abs(p1_below - 0.5) < 0.05
        ), "Both converged to 0.5, but unstable equilibrium should not attract"
