"""
Full Analysis Example.

Complete analysis pipeline: simulation → equilibrium → statistics → welfare → plots.
"""
from pathlib import Path
import numpy as np
from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.analysis.equilibrium import EquilibriumFinder
from src.analysis.statistics import StatisticalAnalyzer
from src.analysis.welfare import WelfareCalculator
from src.visualization import (
    plot_trajectory,
    plot_potential_slices,
    plot_welfare_trajectory
)

# Create output directory
output_dir = Path("outputs/examples/full_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

print("Full Analysis Pipeline")
print("=" * 60)

# 1. RUN SIMULATION
print("\n[1] Running simulation...")
config = SimulationConfig(
    N=1000,
    T=1000,
    gamma=0.65,      # Below γ* → asymmetric equilibria
    beta=0.9,
    seed=42,
    p_init=0.6       # Start near upper asymmetric equilibrium
)

engine = SimulationEngine()
result = engine.run(config)

print(f"  ✓ Simulation complete")
print(f"    Runtime: {result.metadata['runtime_seconds']:.2f}s")
print(f"    Final p₁: {result.p1_trajectory[-1]:.4f}")

# 2. EQUILIBRIUM ANALYSIS
print("\n[2] Equilibrium analysis...")
finder = EquilibriumFinder()
equilibria = finder.find_equilibria(gamma=config.gamma)

print(f"  ✓ Found {len(equilibria)} equilibria:")
for eq in equilibria:
    stability = "stable" if eq.stable else "unstable"
    print(f"    • {eq.type.value}: p* = {eq.p_star:.4f} ({stability})")

# 3. STATISTICAL ANALYSIS
print("\n[3] Statistical analysis...")
analyzer = StatisticalAnalyzer()

# Determine which equilibrium to use
# Find closest stable equilibrium to final state
stable_eq = [eq for eq in equilibria if eq.stable]

if not stable_eq:
    print("  ✗ No stable equilibrium found. Cannot proceed with statistical analysis.")
    # Sections 4, 5, and 6 depend on this, so we exit.
    exit()

final_p = result.p1_trajectory[-1]
closest_eq = min(stable_eq, key=lambda eq: abs(eq.p_star - final_p))

stats = analyzer.compute_statistics(
    result.p1_trajectory,
    equilibrium_p=closest_eq.p_star,
    threshold=0.02
)

print(f"  ✓ Statistics computed:")
print(f"    Mean: {stats.mean:.4f}")
print(f"    Variance: {stats.variance:.6f}")
print(f"    Std: {stats.std:.6f}")
print(f"    Skewness: {stats.skewness:.4f}")
print(f"    Convergence time: {stats.convergence_time}")
print(f"    Autocorrelation [1,5,10,20]: {stats.autocorrelation}")

# 4. WELFARE ANALYSIS
print("\n[4] Welfare analysis...")

# Generate consumer preferences (approximate - true theta not stored)
np.random.seed(42)
theta = np.random.beta(config.alpha_0, config.beta_0, size=(config.N, 2))

calculator = WelfareCalculator()
welfare = calculator.compute_welfare(
    result.p1_trajectory,
    theta,
    config.gamma
)

print(f"  ✓ Welfare computed:")
print(f"    Optimal welfare: {welfare.welfare_optimal:.4f}")
print(f"    Equilibrium welfare: {welfare.welfare_equilibrium:.4f}")
print(f"    Welfare loss: {welfare.welfare_loss:.4f}")
print(f"    Loss (% of optimal): {welfare.welfare_loss/welfare.welfare_optimal*100:.2f}%")

# 5. VISUALIZATIONS
print("\n[5] Creating visualizations...")

# Trajectory plot
plot1 = output_dir / "trajectory.pdf"
plot_trajectory(result, save_path=plot1, show_equilibria=True)
print(f"  ✓ Saved: {plot1}")

# Potential function slices
plot2 = output_dir / "potential_slices.pdf"
gamma_values = [0.60, config.gamma, 0.70, 12/17, 0.75]
plot_potential_slices(gamma_values=gamma_values, save_path=plot2)
print(f"  ✓ Saved: {plot2}")

# Welfare trajectory
plot3 = output_dir / "welfare_trajectory.pdf"
plot_welfare_trajectory(result, save_path=plot3)
print(f"  ✓ Saved: {plot3}")

# 6. SUMMARY
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)
print(f"Configuration: N={config.N}, T={config.T}, γ={config.gamma:.3f}, β={config.beta}")
print(f"\nEquilibria: {len(equilibria)} found ({len(stable_eq)} stable)")
print(f"Final state: p₁ = {result.p1_trajectory[-1]:.4f}")
print(f"Converged to: {closest_eq.type.value} equilibrium (p* = {closest_eq.p_star:.4f})")
print(f"Convergence time: {stats.convergence_time} steps")
print(f"\nWelfare loss: {welfare.welfare_loss:.4f} ({welfare.welfare_loss/welfare.welfare_optimal*100:.2f}% of optimal)")
print(f"\nPlots saved in: {output_dir}")
print("=" * 60)

print(f"\n✓ Full analysis complete!")
