"""
Bifurcation Analysis Example.

Demonstrates how to analyze the bifurcation at γ* = 12/17.
"""
from pathlib import Path
from src.analysis.bifurcation import BifurcationAnalyzer
from src.analysis.equilibrium import EquilibriumFinder
from src.visualization import plot_bifurcation, plot_bifurcation_with_amplitude
from src.distributions.constants import GAMMA_STAR

# Create output directory
output_dir = Path("outputs/examples")
output_dir.mkdir(parents=True, exist_ok=True)

print("Bifurcation Analysis")
print("=" * 60)
print(f"Critical point: γ* = {GAMMA_STAR:.6f} (12/17)")
print()

# Analyze equilibria at different gamma values
print("Equilibria at different γ values:")
print("-" * 60)

finder = EquilibriumFinder()

gamma_values = [0.60, 0.65, GAMMA_STAR, 0.75, 0.80]

for gamma in gamma_values:
    equilibria = finder.find_equilibria(gamma)
    print(f"\nγ = {gamma:.4f}:")
    for eq in equilibria:
        stability = "stable" if eq.stable else "unstable"
        print(f"  • {eq.type.value}: p* = {eq.p_star:.4f} ({stability})")

# Compute bifurcation diagram
print("\n" + "=" * 60)
print("Computing bifurcation diagram...")

analyzer = BifurcationAnalyzer()
bif_data = analyzer.compute_bifurcation_diagram(
    gamma_min=0.5,
    gamma_max=0.9,
    resolution=100
)

print(f"✓ Computed {len(bif_data.gamma_grid)} gamma points")

# Create visualizations
print("\nCreating visualizations...")

# 1. Standard bifurcation diagram
plot1_path = output_dir / "bifurcation_diagram.pdf"
plot_bifurcation(bif_data, save_path=plot1_path)
print(f"  ✓ Saved: {plot1_path}")

# 2. Bifurcation with amplitude scaling
plot2_path = output_dir / "bifurcation_amplitude.pdf"
plot_bifurcation_with_amplitude(bif_data, save_path=plot2_path)
print(f"  ✓ Saved: {plot2_path}")

print(f"\n✓ Done! Check outputs in {output_dir}")
