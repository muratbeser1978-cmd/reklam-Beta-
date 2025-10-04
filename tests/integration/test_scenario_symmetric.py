"""
Integration Test: Symmetric Equilibrium (Scenario 2 from quickstart.md).

Verifies convergence to p* = 0.5 when γ > γ*.
"""
import pytest
import numpy as np
from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.distributions.constants import GAMMA_STAR


class TestSymmetricEquilibrium:
    """Test convergence to symmetric equilibrium (γ > γ*)."""

    def test_convergence_to_half(self):
        """
        Scenario 2: γ > γ* → convergence to p* = 0.5.

        Equations Tested:
            - Eq 8: γ* = 12/17 ≈ 0.706
            - Eq 41-42: Symmetric equilibrium p^sym = 0.5, stable for γ > γ*
            - Eq 60: Long-run convergence
        """
        # γ = 0.80 > γ* = 0.706
        config = SimulationConfig(N=500, T=500, gamma=0.80, beta=0.9, seed=123, p_init=0.6)

        assert config.gamma > GAMMA_STAR, f"Test setup error: gamma should be > γ*"

        engine = SimulationEngine()
        result = engine.run(config)

        # Check final market share
        p1_final = result.p1_trajectory[-1]
        deviation = abs(p1_final - 0.5)

        assert deviation < 0.1, (
            f"FAIL: Did not converge to symmetric equilibrium. "
            f"|p₁(T) - 0.5| = {deviation:.4f} (should be < 0.1)"
        )

    def test_multiple_initial_conditions_converge(self):
        """Multiple initial conditions should all converge to 0.5 when γ > γ*."""
        gamma = 0.80  # > γ*
        initial_conditions = [0.2, 0.3, 0.5, 0.7, 0.8]

        engine = SimulationEngine()

        for p_init in initial_conditions:
            config = SimulationConfig(
                N=200, T=300, gamma=gamma, beta=0.9, seed=42, p_init=p_init
            )
            result = engine.run(config)

            p1_final = result.p1_trajectory[-1]
            deviation = abs(p1_final - 0.5)

            assert deviation < 0.15, (
                f"Starting from p_init={p_init}, did not converge to 0.5. "
                f"Final p1 = {p1_final:.4f}"
            )
