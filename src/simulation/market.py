"""
Market State and Dynamics.

Tracks aggregate market share and implements market dynamics.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class MarketState:
    """
    Aggregate market configuration at time t.

    Attributes:
        p1: Market share of brand 1 (Eq 53)
        t: Current timestep

    Properties:
        p2: Market share of brand 2 (= 1 - p1)
    """

    p1: float
    t: int

    def __post_init__(self):
        """Validate market state invariants."""
        if not (0.0 <= self.p1 <= 1.0):
            raise ValueError(f"p1={self.p1} out of range [0, 1]")
        if not (abs(self.p1 + self.p2 - 1.0) < 1e-10):
            raise ValueError(f"Conservation violated: p1 + p2 = {self.p1 + self.p2} != 1.0")

    @property
    def p2(self) -> float:
        """Market share of brand 2."""
        return 1.0 - self.p1

    def update(self, choices: np.ndarray, beta_retention: float):
        """
        Update market share based on consumer choices (Eq 53, 2.7a-2.7b).

        Args:
            choices: (N,) array of consumer choices {0, 1}
            beta_retention: Retention probability β

        Equation:
            p₁(t+1) = β·p₁(t) + (1-β)·BR(p₁(t)) + β·(BR(p₁(t)) - p₁(t))
                    = BR(p₁(t))  [Eq 2.7*]

        Implementation:
            Simplified dynamics: market share = fraction choosing brand 1
        """
        # Compute fraction choosing brand 1 (choice=0 is brand 1)
        fraction_brand_1 = (choices == 0).mean()

        # Update market share
        # In equilibrium dynamics, p₁(t+1) approaches BR(p₁(t))
        # For simulation, we use the direct fraction
        self.p1 = fraction_brand_1

        # Increment timestep
        self.t += 1

        # Ensure conservation
        if abs(self.p1 + self.p2 - 1.0) >= 1e-10:
            # Numerical precision: renormalize
            self.p1 = min(max(self.p1, 0.0), 1.0)
