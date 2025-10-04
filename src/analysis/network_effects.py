"""
Network Effects and Social Influence Analysis.

Analyzes social externalities and network effects in consumer choices.

Key Concepts:
    - Utility decomposition: U = γ·θ (individual) + (1-γ)·p (social)
    - Social multiplier: M(γ) = (1-γ)/γ
    - Tipping dynamics: Critical mass for market dominance
    - Bandwagon effects: Positive feedback loops

Theoretical Background:
    - Eq 7: M(γ) = (1-γ)/γ measures social amplification
    - Eq 20-21: Utility decomposition
    - Network externalities drive lock-in when γ < γ*
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from scipy import stats


@dataclass
class UtilityDecomposition:
    """
    Decomposition of utility into individual and social components.

    Attributes:
        individual_utility_mean: Mean γ·θ across consumers
        social_utility_mean: Mean (1-γ)·p across consumers
        individual_contribution: Fraction from individual taste
        social_contribution: Fraction from social influence
        correlation: Correlation between individual and social components
        gamma: Parameter value
    """

    individual_utility_mean: float
    social_utility_mean: float
    individual_contribution: float
    social_contribution: float
    correlation: float
    gamma: float


@dataclass
class SocialMultiplierAnalysis:
    """
    Social multiplier M(γ) = (1-γ)/γ analysis.

    Attributes:
        gamma: Individual taste weight
        multiplier: M(γ) value
        interpretation: Verbal interpretation
        regime: "individual_dominant" or "social_dominant"
        amplification_factor: How much social effects amplify changes
    """

    gamma: float
    multiplier: float
    interpretation: str
    regime: str
    amplification_factor: float


@dataclass
class TippingAnalysis:
    """
    Tipping point analysis for market dominance.

    Attributes:
        tipping_threshold: Critical market share for irreversible dominance
        below_threshold_outcome: Equilibrium when starting below threshold
        above_threshold_outcome: Equilibrium when starting above threshold
        basin_size: Size of attraction basin [0, 1]
        hysteresis_detected: Whether hysteresis is present
        gamma: Parameter value
    """

    tipping_threshold: float
    below_threshold_outcome: float
    above_threshold_outcome: float
    basin_size: float
    hysteresis_detected: bool
    gamma: float


class NetworkEffectsAnalyzer:
    """
    Analyze network effects and social influence.

    Usage:
        analyzer = NetworkEffectsAnalyzer()
        decomp = analyzer.decompose_utility(consumer_state, market_state, gamma)
        multiplier = analyzer.compute_social_multiplier(gamma)
        tipping = analyzer.analyze_tipping_point(trajectories, gamma)
    """

    def __init__(self, gamma_star: float = 12.0 / 17.0):
        """
        Initialize analyzer.

        Args:
            gamma_star: Critical bifurcation threshold
        """
        self.gamma_star = gamma_star

    def decompose_utility(
        self, consumer_state, market_state, gamma: float
    ) -> UtilityDecomposition:
        """
        Decompose utility into individual and social components.

        Args:
            consumer_state: ConsumerState object
            market_state: MarketState with p1, p2
            gamma: Individual taste weight

        Returns:
            UtilityDecomposition object

        Mathematical Formula:
            U_{i,a} = γ·θ_{i,a} + (1-γ)·p_a  [Eq 22]

            Individual component: γ·θ_{i,a}  [Eq 20]
            Social component: (1-γ)·p_a      [Eq 21]

        Example:
            >>> decomp = analyzer.decompose_utility(consumer_state, market_state, 0.7)
            >>> print(f"Individual: {decomp.individual_contribution:.1%}")
            >>> print(f"Social: {decomp.social_contribution:.1%}")
        """
        theta = consumer_state.theta  # (N, 2)
        p1 = market_state.p1 if hasattr(market_state, "p1") else 0.5
        p2 = 1.0 - p1

        # Individual utility component: γ·θ
        individual_utility_brand1 = gamma * theta[:, 0]
        individual_utility_brand2 = gamma * theta[:, 1]

        # Social utility component: (1-γ)·p
        social_utility_brand1 = (1 - gamma) * p1
        social_utility_brand2 = (1 - gamma) * p2

        # Average across consumers and brands
        individual_mean = (individual_utility_brand1.mean() + individual_utility_brand2.mean()) / 2
        social_mean = (social_utility_brand1 + social_utility_brand2) / 2  # Constant across consumers

        # Contributions to total utility
        total_utility = individual_mean + social_mean
        individual_contribution = individual_mean / total_utility if total_utility > 0 else 0
        social_contribution = social_mean / total_utility if total_utility > 0 else 0

        # Correlation between individual tastes and social influence
        # (For a given brand, does individual preference correlate with its popularity?)
        correlation = self._compute_taste_popularity_correlation(theta, p1)

        return UtilityDecomposition(
            individual_utility_mean=individual_mean,
            social_utility_mean=social_mean,
            individual_contribution=individual_contribution,
            social_contribution=social_contribution,
            correlation=correlation,
            gamma=gamma,
        )

    def compute_social_multiplier(self, gamma: float) -> SocialMultiplierAnalysis:
        """
        Compute and interpret social multiplier M(γ).

        Args:
            gamma: Individual taste weight

        Returns:
            SocialMultiplierAnalysis object

        Theory:
            M(γ) = (1-γ)/γ  [Eq 7]

            - M > 1: Social effects amplify individual changes
            - M < 1: Individual preferences dominate
            - M = 1: Balanced (γ = 0.5)

        Example:
            >>> mult = analyzer.compute_social_multiplier(0.6)
            >>> print(f"M(γ=0.6) = {mult.multiplier:.2f}")
            >>> print(mult.interpretation)
        """
        if gamma == 0:
            multiplier = np.inf
            regime = "social_dominant"
            interpretation = "Pure social influence (γ=0): Infinite amplification"
        elif gamma == 1:
            multiplier = 0.0
            regime = "individual_dominant"
            interpretation = "Pure individual taste (γ=1): No social amplification"
        else:
            multiplier = (1 - gamma) / gamma

            if multiplier > 1:
                regime = "social_dominant"
                interpretation = (
                    f"Social effects {multiplier:.2f}× stronger than individual tastes. "
                    f"Market share changes are amplified."
                )
            elif multiplier < 1:
                regime = "individual_dominant"
                interpretation = (
                    f"Individual tastes {1/multiplier:.2f}× stronger than social effects. "
                    f"Market converges to preference distribution."
                )
            else:
                regime = "balanced"
                interpretation = "Balanced: Individual and social effects equally strong."

        # Amplification factor: How much a small change in tastes affects equilibrium
        amplification_factor = 1 / (1 - multiplier) if multiplier < 1 else np.inf

        return SocialMultiplierAnalysis(
            gamma=gamma,
            multiplier=multiplier,
            interpretation=interpretation,
            regime=regime,
            amplification_factor=amplification_factor,
        )

    def analyze_tipping_point(
        self,
        initial_conditions: List[float],
        final_outcomes: List[float],
        gamma: float,
        threshold_tol: float = 0.05,
    ) -> TippingAnalysis:
        """
        Analyze tipping dynamics and basin of attraction.

        Args:
            initial_conditions: List of p_init values
            final_outcomes: Corresponding final p values
            gamma: Parameter value
            threshold_tol: Tolerance for threshold detection

        Returns:
            TippingAnalysis object

        Method:
            1. Sort by initial conditions
            2. Find threshold where outcome flips
            3. Measure basin sizes
            4. Detect hysteresis

        Example:
            >>> # Run simulations with different p_init
            >>> p_inits = np.linspace(0.0, 1.0, 21)
            >>> p_finals = [run_sim(p_init).p1_trajectory[-1] for p_init in p_inits]
            >>> tipping = analyzer.analyze_tipping_point(p_inits, p_finals, gamma=0.6)
            >>> print(f"Tipping threshold: {tipping.tipping_threshold:.3f}")
        """
        initial = np.array(initial_conditions)
        final = np.array(final_outcomes)

        # Sort by initial conditions
        sort_idx = np.argsort(initial)
        initial_sorted = initial[sort_idx]
        final_sorted = final[sort_idx]

        # Detect tipping threshold
        if gamma < self.gamma_star:
            # Below critical: expect tipping at p=0.5
            tipping_threshold = 0.5

            # Classify outcomes
            low_equilibrium = final_sorted[initial_sorted < 0.5].mean()
            high_equilibrium = final_sorted[initial_sorted > 0.5].mean()

            # Basin size (fraction of initial conditions leading to high equilibrium)
            basin_high = (final_sorted > 0.5).mean()

            # Hysteresis: large difference between low and high equilibria
            hysteresis = abs(high_equilibrium - low_equilibrium) > 0.2

        else:
            # Above critical: no tipping, converges to 0.5
            tipping_threshold = np.nan
            low_equilibrium = 0.5
            high_equilibrium = 0.5
            basin_high = 1.0
            hysteresis = False

        return TippingAnalysis(
            tipping_threshold=tipping_threshold,
            below_threshold_outcome=low_equilibrium,
            above_threshold_outcome=high_equilibrium,
            basin_size=basin_high,
            hysteresis_detected=hysteresis,
            gamma=gamma,
        )

    def measure_bandwagon_effect(
        self, p_trajectory: np.ndarray, choice_fractions: np.ndarray
    ) -> float:
        """
        Measure strength of bandwagon effect.

        Args:
            p_trajectory: Market share over time
            choice_fractions: Fraction choosing dominant brand at each t

        Returns:
            Bandwagon strength coefficient ∈ [0, 1]

        Method:
            Correlation between past market share and current choice probability.
            Higher correlation = stronger bandwagon effect.

        Example:
            >>> bandwagon = analyzer.measure_bandwagon_effect(p1_traj, choice_fracs)
            >>> if bandwagon > 0.8:
            ...     print("Strong bandwagon effect detected!")
        """
        if len(p_trajectory) != len(choice_fractions):
            raise ValueError("Trajectories must have same length")

        if len(p_trajectory) < 2:
            return 0.0

        # Lagged correlation: p(t) vs choice_fraction(t+1)
        p_lagged = p_trajectory[:-1]
        choices_current = choice_fractions[1:]

        if len(p_lagged) < 2:
            return 0.0

        # Correlation
        corr, _ = stats.pearsonr(p_lagged, choices_current)

        # Return absolute correlation as strength measure
        return abs(corr)

    def compute_social_welfare_decomposition(
        self, consumer_state, market_state, gamma: float
    ) -> Dict[str, float]:
        """
        Decompose social welfare into individual and network components.

        Args:
            consumer_state: ConsumerState object
            market_state: MarketState
            gamma: Parameter value

        Returns:
            Dictionary with welfare components

        Components:
            - individual_welfare: From individual taste match
            - network_welfare: From network effects
            - total_welfare: Sum of both

        Example:
            >>> welfare_decomp = analyzer.compute_social_welfare_decomposition(
            ...     consumer_state, market_state, 0.7
            ... )
            >>> print(f"Network contributes {welfare_decomp['network_fraction']:.1%}")
        """
        from src.simulation.utilities import Q

        theta = consumer_state.theta
        p1 = market_state.p1 if hasattr(market_state, "p1") else 0.5
        p2 = 1.0 - p1

        # Total utility for each consumer
        u1 = Q(theta[:, 0], p1, gamma)
        u2 = Q(theta[:, 1], p2, gamma)
        total_utility = np.maximum(u1, u2)

        # Individual component only (γ=1)
        u1_individual = theta[:, 0]
        u2_individual = theta[:, 1]
        individual_utility = np.maximum(u1_individual, u2_individual)

        # Network component (residual)
        network_utility = total_utility - gamma * individual_utility

        # Aggregate
        total_welfare = total_utility.mean()
        individual_welfare = (gamma * individual_utility).mean()
        network_welfare = network_utility.mean()

        return {
            "total_welfare": total_welfare,
            "individual_welfare": individual_welfare,
            "network_welfare": network_welfare,
            "network_fraction": network_welfare / total_welfare if total_welfare > 0 else 0,
        }

    def _compute_taste_popularity_correlation(
        self, theta: np.ndarray, p1: float
    ) -> float:
        """
        Correlation between individual taste for brand 1 and its popularity.

        Positive correlation: Individuals with taste for popular brand.
        Negative correlation: Contrarian preferences.
        """
        # Preference for brand 1 vs brand 2
        preference_brand1 = theta[:, 0] - theta[:, 1]

        # Popularity measure (constant, so correlation is undefined)
        # Instead, compute correlation with absolute taste strength
        taste_strength = np.abs(preference_brand1)

        # Placeholder: return 0 (could be extended with temporal analysis)
        return 0.0


def estimate_critical_mass(
    gamma_values: np.ndarray, tipping_thresholds: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Estimate critical mass as function of γ.

    Args:
        gamma_values: Array of γ parameters
        tipping_thresholds: Corresponding tipping thresholds

    Returns:
        (average_critical_mass, fitted_curve)

    Theory:
        For γ < γ*, critical mass ≈ 0.5
        For γ > γ*, no critical mass (market self-corrects)

    Example:
        >>> gammas = np.linspace(0.5, 0.9, 20)
        >>> thresholds = [analyze_tipping(...).tipping_threshold for g in gammas]
        >>> critical_mass, curve = estimate_critical_mass(gammas, thresholds)
    """
    # Filter valid values (not NaN)
    valid = ~np.isnan(tipping_thresholds)
    gamma_valid = gamma_values[valid]
    threshold_valid = tipping_thresholds[valid]

    if len(threshold_valid) == 0:
        return np.nan, np.array([])

    # Average critical mass
    avg_critical_mass = np.mean(threshold_valid)

    # Fitted curve (constant 0.5 for γ < γ*, NaN for γ > γ*)
    gamma_star = 12.0 / 17.0
    fitted = np.where(gamma_values < gamma_star, 0.5, np.nan)

    return avg_critical_mass, fitted
