"""
Basic Simulation Example.

Demonstrates how to run a single simulation and visualize results.
"""
from pathlib import Path
from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.visualization import plot_trajectory

# Create output directory
output_dir = Path("outputs/examples")
output_dir.mkdir(parents=True, exist_ok=True)

# Configure simulation
config = SimulationConfig(
    N=1000,           # 1000 consumers
    T=500,            # 500 time steps
    gamma=0.75,       # Learning weight (γ > γ* = 0.706)
    beta=0.9,         # Demographic churn rate
    seed=42,          # For reproducibility
    p_init=0.5        # Initial market share
)

print("Running simulation...")
print(f"  N = {config.N} consumers")
print(f"  T = {config.T} time steps")
print(f"  γ = {config.gamma}")
print(f"  β = {config.beta}")

# Run simulation
engine = SimulationEngine()
result = engine.run(config)

print(f"\n✓ Simulation complete!")
print(f"  Runtime: {result.metadata['runtime_seconds']:.2f} seconds")
print(f"  Final market share p₁ = {result.p1_trajectory[-1]:.4f}")
print(f"  Final market share p₂ = {result.p2_trajectory[-1]:.4f}")

# Create visualization
print(f"\nCreating trajectory plot...")
plot_path = output_dir / "basic_trajectory.pdf"
plot_trajectory(result, save_path=plot_path, show_equilibria=True)

print(f"\n✓ Done! Check {plot_path}")
