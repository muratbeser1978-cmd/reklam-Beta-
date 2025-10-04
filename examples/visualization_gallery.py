"""
Visualization Gallery Example.

Demonstrates all available visualization functions.
"""
from pathlib import Path
import numpy as np
from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.analysis.bifurcation import BifurcationAnalyzer
from src.visualization import (
    plot_trajectory,
    plot_multiple_trajectories,
    plot_trajectory_with_variance,
    plot_bifurcation,
    plot_bifurcation_with_amplitude,
    plot_potential_landscape,
    plot_potential_slices,
    plot_potential_derivative,
    plot_welfare_loss,
)

def main():
    """Main function to generate all plots."""
    # Create output directory
    output_dir = Path("outputs/examples/gallery")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Visualization Gallery")
    print("=" * 60)
    print("Creating all available plot types...\n")

    # Run a few simulations for various plots
    base_config = SimulationConfig(N=500, T=300, gamma=0.7, beta=0.9, seed=42)
    engine = SimulationEngine()

    print("[1/9] Single trajectory plot...")
    result1 = engine.run(base_config)
    plot_trajectory(result1, save_path=output_dir / "01_trajectory.pdf")

    print("[2/9] Multiple trajectories plot...")
    configs = [
        SimulationConfig(N=500, T=300, gamma=0.7, beta=0.9, seed=s)
        for s in [42, 123, 456, 789, 999]
    ]
    results = engine.run_batch(configs, n_workers=4)
    labels = [f"Seed {s}" for s in [42, 123, 456, 789, 999]]
    plot_multiple_trajectories(results, labels=labels, save_path=output_dir / "02_multiple_trajectories.pdf")

    print("[3/9] Trajectory with variance bands...")
    plot_trajectory_with_variance(results, save_path=output_dir / "03_trajectory_variance.pdf")

    print("[4/9] Bifurcation diagram...")
    analyzer = BifurcationAnalyzer()
    bif_data = analyzer.compute_bifurcation_diagram(
        gamma_min=0.5, gamma_max=0.9, resolution=80
    )
    plot_bifurcation(bif_data, save_path=output_dir / "04_bifurcation_diagram.pdf")

    print("[5/9] Bifurcation with amplitude scaling...")
    plot_bifurcation_with_amplitude(bif_data, save_path=output_dir / "05_bifurcation_amplitude.pdf")

    print("[6/9] Potential landscape (3D surface)...")
    plot_potential_landscape(
        gamma_values=np.linspace(0.5, 0.9, 25),
        p_grid_resolution=80,
        save_path=output_dir / "06_potential_3d.pdf",
        plot_3d=True
    )

    print("[7/9] Potential landscape (2D contour)...")
    plot_potential_landscape(
        gamma_values=np.linspace(0.5, 0.9, 25),
        p_grid_resolution=80,
        save_path=output_dir / "07_potential_contour.pdf",
        plot_3d=False
    )

    print("[8/9] Potential function slices...")
    from src.distributions.constants import GAMMA_STAR
    gamma_slices = [0.60, 0.65, GAMMA_STAR, 0.75, 0.80]
    plot_potential_slices(gamma_values=gamma_slices, save_path=output_dir / "08_potential_slices.pdf")

    print("[9/9] Potential gradient (dV/dp)...")
    plot_potential_derivative(gamma_values=gamma_slices, save_path=output_dir / "09_potential_gradient.pdf")

    # Bonus: Welfare loss plot (needs data)
    print("[Bonus] Welfare loss plot...")
    # Compute welfare loss for different gamma values
    from src.analysis.welfare import WelfareCalculator
    gamma_grid = np.linspace(0.5, 0.9, 20)
    welfare_losses = []
    theta = np.random.beta(2, 2, size=(500, 2))
    calculator = WelfareCalculator()

    for gamma in gamma_grid:
        # Run quick simulation at this gamma
        cfg = SimulationConfig(N=500, T=200, gamma=gamma, beta=0.9, seed=42)
        res = engine.run(cfg)
        welfare = calculator.compute_welfare(res.p1_trajectory, theta, gamma)
        welfare_losses.append(welfare.welfare_loss)

    welfare_losses = np.array(welfare_losses)
    plot_welfare_loss(gamma_grid, welfare_losses, save_path=output_dir / "10_welfare_loss.pdf")

    print("\n" + "=" * 60)
    print(f"✓ All visualizations created!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total plots: 10")
    print("=" * 60)
    print("\nPlot descriptions:")
    print("  01: Single trajectory with equilibrium lines")
    print("  02: Multiple trajectories (different seeds)")
    print("  03: Mean trajectory with ±1 std bands")
    print("  04: Bifurcation diagram (supercritical pitchfork)")
    print("  05: Bifurcation amplitude δ ~ √(γ* - γ)")
    print("  06: Potential landscape V(p,γ) (3D surface)")
    print("  07: Potential landscape V(p,γ) (2D contour)")
    print("  08: Potential slices at different γ values")
    print("  09: Potential gradient dV/dp = BR(p) - p")
    print("  10: Welfare loss ΔW across γ values")

if __name__ == '__main__':
    main()