"""
Main Simulation Engine.

Executes complete simulation from t=0 to t=T.
"""
import numpy as np
from datetime import datetime
from typing import List
from multiprocessing import Pool, cpu_count
from src.simulation.config import SimulationConfig
from src.simulation.consumer import ConsumerState
from src.simulation.market import MarketState
from src.simulation.result import SimulationResult
from src.simulation.thompson import thompson_sample
from src.simulation.reward import generate_rewards


class SimulationEngine:
    """
    Simulation engine implementing Protocol from contracts/simulation_engine.py.

    Guarantees:
        - Deterministic: run(config) == run(config) for same seed
        - Conservation: ∀t: p₁(t) + p₂(t) = 1.0 (within 1e-10)
        - Bounds: ∀t: p₁(t), p₂(t) ∈ [0, 1]
    """

    def run(self, config: SimulationConfig) -> SimulationResult:
        """
        Execute simulation from t=0 to t=T.

        Args:
            config: Simulation parameters

        Returns:
            Complete trajectory and metadata

        Algorithm:
            Initialization (t=0):
                - θᵢ,ₐ ~ Beta(α₀, β₀)  [Eq 9-10]
                - αᵢ,ₐ(0) = α₀, βᵢ,ₐ(0) = β₀  [Eq 15-16]
                - p₁(0) = p_init  [Eq 13]

            Each timestep t → t+1:
                - θ̃ᵢ,ₐ(t) ~ Beta(αᵢ,ₐ(t), βᵢ,ₐ(t))  [Eq 23]
                - q̃ᵢ,ₐ(t) = Q(θ̃ᵢ,ₐ(t), pₐ(t))  [Eq 24]
                - cᵢ(t) = argmax_a q̃ᵢ,ₐ(t)  [Eq 25]
                - Rᵢ(t) ~ Bernoulli(qᵢ,cᵢ(t)(t))  [Eq 26]
                - Bayesian update  [Eq 27-28]
                - Market share update  [Eq 53]
                - Demographic churn  [Eq 62]
        """
        start_time = datetime.now()

        # Create isolated RNG
        rng = config.create_rng()

        # Initialize consumer state (Eq 9-10, 15-16)
        consumer_state = ConsumerState.initialize(
            N=config.N, alpha_0=config.alpha_0, beta_0=config.beta_0, rng=rng
        )

        # Initialize market state (Eq 13)
        market_state = MarketState(p1=config.p_init, t=0)

        # Storage for trajectories
        p1_trajectory = np.zeros(config.T + 1)
        p2_trajectory = np.zeros(config.T + 1)
        choices_history = np.zeros((config.N, config.T), dtype=np.int8)
        rewards_history = np.zeros((config.N, config.T), dtype=np.int8)

        # Store initial state
        p1_trajectory[0] = market_state.p1
        p2_trajectory[0] = market_state.p2

        # Main simulation loop
        for t in range(config.T):
            # 1. Thompson sampling → choices (Eq 23-25)
            choices = thompson_sample(consumer_state, market_state, rng, config.gamma)

            # 2. Generate rewards (Eq 26)
            rewards = generate_rewards(consumer_state, choices, market_state, config.gamma, rng)

            # 3. Update beliefs (Eq 27-28)
            consumer_state.update_beliefs(choices, rewards)

            # 4. Update market share (Eq 53)
            market_state.update(choices, config.beta)

            # 5. Apply churn (Eq 62)
            consumer_state.apply_churn(config.beta, rng, config.alpha_0, config.beta_0)

            # Store trajectory
            p1_trajectory[t + 1] = market_state.p1
            p2_trajectory[t + 1] = market_state.p2
            choices_history[:, t] = choices
            rewards_history[:, t] = rewards

        # Create metadata
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()

        metadata = {
            "timestamp": start_time.isoformat(),
            "runtime_seconds": runtime,
            "final_p1": float(p1_trajectory[-1]),
            "final_p2": float(p2_trajectory[-1]),
        }

        return SimulationResult(
            p1_trajectory=p1_trajectory,
            p2_trajectory=p2_trajectory,
            choices=choices_history,
            rewards=rewards_history,
            config=config,
            metadata=metadata,
        )

    def run_batch(
        self, configs: List[SimulationConfig], n_workers: int = None
    ) -> List[SimulationResult]:
        """
        Execute multiple simulations in parallel.

        Args:
            configs: List of simulation configurations to run
            n_workers: Number of parallel workers (default: cpu_count())

        Returns:
            List of results in same order as configs

        Implementation:
            - Uses multiprocessing.Pool for parallelization
            - Each worker runs independent simulation with isolated RNG
            - Results returned in original order
        """
        if n_workers is None:
            n_workers = cpu_count()

        # Use multiprocessing.Pool for parallel execution
        with Pool(processes=n_workers) as pool:
            results = pool.map(self.run, configs)

        return results
