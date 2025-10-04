"""
Advanced Visualization Gallery (Phase 2).

Demonstrates new class-based plotters with comprehensive examples.
Creates 22 publication-quality plots (300 DPI).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.simulation.consumer import ConsumerState
from src.simulation.market import MarketState

from src.visualization import (
    TrajectoryPlotter,
    PhasePortraitPlotter,
    BifurcationPlotter,
    HeatmapPlotter,
    DistributionPlotter,
)

from src.analysis.welfare_corrected import WelfareCalculatorCorrected
from src.analysis.critical_slowing import CriticalSlowingAnalyzer
from src.analysis.network_effects import NetworkEffectsAnalyzer


def main():
    """Generate comprehensive visualization gallery."""
    # Create output directory
    output_dir = Path("outputs/gallery_phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ADVANCED VISUALIZATION GALLERY - PHASE 2")
    print("=" * 70)
    print(f"Output directory: {output_dir}\n")

    # Initialize plotters
    traj_plotter = TrajectoryPlotter()
    phase_plotter = PhasePortraitPlotter()
    bifurc_plotter = BifurcationPlotter()
    heatmap_plotter = HeatmapPlotter()
    dist_plotter = DistributionPlotter()

    # Run base simulations
    print("[Setup] Running base simulations...")
    np.random.seed(42)
    engine = SimulationEngine()

    # Simulation 1: Below critical (γ < γ*)
    config_low = SimulationConfig(N=200, T=250, gamma=0.65, beta=0.9, seed=42)
    result_low = engine.run(config_low)

    # Generate consumer preferences (approximate - true theta not stored in result)
    np.random.seed(42)
    theta = np.random.beta(config_low.alpha_0, config_low.beta_0, size=(config_low.N, 2))

    # Simulation 2: Above critical (γ > γ*)
    config_high = SimulationConfig(N=200, T=250, gamma=0.75, beta=0.9, seed=42)
    result_high = engine.run(config_high)

    print("✓ Base simulations complete\n")

    # =========================================================================
    # TRAJECTORY PLOTS (5 plots)
    # =========================================================================
    print("TRAJECTORY PLOTS")
    print("-" * 70)

    print("[1/25] Market share trajectory (low γ)...")
    fig = traj_plotter.plot_market_share_trajectory(
        result_low.p1_trajectory, gamma=0.65, burnin=50
    )
    fig.savefig(output_dir / "01_trajectory_low_gamma.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[2/25] Market share trajectory (high γ)...")
    fig = traj_plotter.plot_market_share_trajectory(
        result_high.p1_trajectory, gamma=0.75, burnin=50
    )
    fig.savefig(output_dir / "02_trajectory_high_gamma.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[3/25] Welfare trajectory...")
    welfare_calc = WelfareCalculatorCorrected()
    welfare = welfare_calc.compute_welfare(
        result_low.p1_trajectory,
        theta,
        gamma=0.65,
        choices=result_low.choices
    )
    fig = traj_plotter.plot_welfare_trajectory(welfare, burnin=50, show_components=True)
    fig.savefig(output_dir / "03_welfare_trajectory.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[4/25] Multiple trajectories comparison...")
    # Run 5 simulations with different seeds
    trajectories = []
    for seed in [42, 123, 456, 789, 999]:
        cfg = SimulationConfig(N=200, T=250, gamma=0.65, beta=0.9, seed=seed)
        res = engine.run(cfg)
        trajectories.append(res.p1_trajectory)

    fig = traj_plotter.plot_multi_trajectory(
        trajectories,
        labels=[f'Seed {s}' for s in [42, 123, 456, 789, 999]],
        show_mean=True,
        show_std=True
    )
    fig.savefig(output_dir / "04_multiple_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[5/25] Critical slowing down indicators...")
    csd_analyzer = CriticalSlowingAnalyzer()
    csd_metrics = csd_analyzer.compute_metrics(result_low.p1_trajectory, gamma=0.65, burnin=50)
    fig = traj_plotter.plot_critical_slowing_down(
        result_low.p1_trajectory, csd_metrics, window_size=50
    )
    fig.savefig(output_dir / "05_critical_slowing_down.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # =========================================================================
    # PHASE PORTRAITS (6 plots)
    # =========================================================================
    print("\nPHASE PORTRAITS")
    print("-" * 70)

    print("[6/25] Best response curve (γ < γ*)...")
    fig = phase_plotter.plot_best_response(gamma=0.65, show_stability=True)
    fig.savefig(output_dir / "06_br_curve_low_gamma.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[7/25] Best response curve (γ > γ*)...")
    fig = phase_plotter.plot_best_response(gamma=0.75, show_stability=True)
    fig.savefig(output_dir / "07_br_curve_high_gamma.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[8/25] Potential landscape (γ < γ*)...")
    fig = phase_plotter.plot_potential_landscape(gamma=0.65, show_derivative=True)
    fig.savefig(output_dir / "08_potential_low_gamma.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[9/25] Potential landscape (γ > γ*)...")
    fig = phase_plotter.plot_potential_landscape(gamma=0.75, show_derivative=True)
    fig.savefig(output_dir / "09_potential_high_gamma.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[10/25] Phase space with trajectory...")
    fig = phase_plotter.plot_phase_space(
        gamma=0.65, trajectory=result_low.p1_trajectory, n_arrows=15
    )
    fig.savefig(output_dir / "10_phase_space.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[11/25] Bifurcation comparison (side-by-side)...")
    fig = phase_plotter.plot_bifurcation_comparison(gamma_below=0.65, gamma_above=0.75)
    fig.savefig(output_dir / "11_bifurcation_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # =========================================================================
    # BIFURCATION DIAGRAMS (4 plots)
    # =========================================================================
    print("\nBIFURCATION DIAGRAMS")
    print("-" * 70)

    print("[12/25] Classic bifurcation diagram...")
    gamma_values = np.linspace(0.55, 0.85, 150)
    fig = bifurc_plotter.plot_bifurcation_diagram(
        gamma_values=gamma_values, show_stability=True, show_critical_line=True
    )
    fig.savefig(output_dir / "12_bifurcation_diagram.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[13/25] Variance scaling (critical exponent)...")
    # Compute variances for different gammas
    gamma_above = np.linspace(0.71, 0.85, 20)
    variances = []
    for g in gamma_above:
        cfg = SimulationConfig(N=200, T=300, gamma=g, beta=0.9, seed=42)
        res = engine.run(cfg)
        var = np.var(res.p1_trajectory[-100:])
        variances.append(var)

    fig = bifurc_plotter.plot_variance_scaling(
        gamma_above, np.array(variances), theoretical_exponent=1.0, show_fit=True
    )
    fig.savefig(output_dir / "13_variance_scaling.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[14/25] Parameter sweep (multiple metrics)...")
    gamma_sweep = np.linspace(0.6, 0.8, 15)
    metrics = {
        'Variance': [],
        'Mean p₁': [],
    }
    for g in gamma_sweep:
        cfg = SimulationConfig(N=200, T=200, gamma=g, beta=0.9, seed=42)
        res = engine.run(cfg)
        eq_vals = res.p1_trajectory[-50:]
        metrics['Variance'].append(np.var(eq_vals))
        metrics['Mean p₁'].append(np.mean(eq_vals))

    fig = bifurc_plotter.plot_parameter_sweep(
        gamma_sweep,
        {k: np.array(v) for k, v in metrics.items()},
        ylabel='Metric Value',
        title='Parameter Sweep: Metrics vs γ'
    )
    fig.savefig(output_dir / "14_parameter_sweep.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[15/25] Stability analysis (BR' at equilibria)...")
    gamma_range = np.linspace(0.6, 0.8, 50)
    fig = phase_plotter.plot_stability_analysis(gamma_range)
    fig.savefig(output_dir / "15_stability_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # =========================================================================
    # HEATMAPS (4 plots)
    # =========================================================================
    print("\nHEATMAPS")
    print("-" * 70)

    print("[16/25] Welfare loss heatmap (γ, β)...")
    # Small grid for demonstration
    gamma_grid = np.linspace(0.6, 0.8, 15)
    beta_grid = np.linspace(0.8, 0.95, 12)
    welfare_grid = np.zeros((len(beta_grid), len(gamma_grid)))

    for i, beta in enumerate(beta_grid):
        for j, gamma in enumerate(gamma_grid):
            cfg = SimulationConfig(N=150, T=150, gamma=gamma, beta=beta, seed=42)
            res = engine.run(cfg)
            # Generate theta for this configuration
            theta_grid = np.random.beta(cfg.alpha_0, cfg.beta_0, size=(cfg.N, 2))
            welf = welfare_calc.compute_welfare(
                res.p1_trajectory, theta_grid, gamma, res.choices
            )
            welfare_grid[i, j] = welf.welfare_loss

    fig = heatmap_plotter.plot_welfare_heatmap(
        gamma_grid, beta_grid, welfare_grid, mark_critical=True
    )
    fig.savefig(output_dir / "16_welfare_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[17/25] Variance heatmap (γ, β)...")
    variance_grid = np.zeros((len(beta_grid), len(gamma_grid)))
    for i, beta in enumerate(beta_grid):
        for j, gamma in enumerate(gamma_grid):
            cfg = SimulationConfig(N=150, T=150, gamma=gamma, beta=beta, seed=42)
            res = engine.run(cfg)
            variance_grid[i, j] = np.var(res.p1_trajectory[-50:])

    fig = heatmap_plotter.plot_variance_heatmap(
        gamma_grid, beta_grid, variance_grid, log_scale=True, mark_critical=True
    )
    fig.savefig(output_dir / "17_variance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[18/25] Generic parameter heatmap...")
    fig = heatmap_plotter.plot_parameter_heatmap(
        gamma_grid, beta_grid, welfare_grid,
        xlabel='Individual Taste Weight γ',
        ylabel='Retention Rate β',
        zlabel='Welfare Loss ΔW',
        title='Welfare Loss Across Parameter Space',
        cmap='plasma',
        show_contours=True
    )
    fig.savefig(output_dir / "18_generic_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[19/25] Correlation heatmap...")
    # Dummy correlation matrix
    variables = ['Welfare Loss', 'Variance', 'Mean p₁', 'Autocorr']
    corr_matrix = np.random.rand(4, 4)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Symmetric
    np.fill_diagonal(corr_matrix, 1.0)

    fig = heatmap_plotter.plot_correlation_heatmap(
        variables, corr_matrix, annotate=True
    )
    fig.savefig(output_dir / "19_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # =========================================================================
    # DISTRIBUTIONS (6 plots)
    # =========================================================================
    print("\nDISTRIBUTIONS")
    print("-" * 70)

    print("[20/25] Preference distribution...")
    fig = dist_plotter.plot_preference_distribution(
        theta, show_kde=True, show_theory=True
    )
    fig.savefig(output_dir / "20_preference_distribution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[21/25] Posterior distribution (Brand 1)...")
    # Skip this plot - requires ConsumerState which is not stored in SimulationResult
    # Would need to refactor to track consumer state separately
    print("  (Skipped - requires consumer state tracking)")
    # fig = dist_plotter.plot_posterior_distribution(
    #     result_low.consumer_state, brand=0, show_convergence=True
    # )
    # fig.savefig(output_dir / "21_posterior_distribution.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)

    print("[22/25] Equilibrium distribution (multiple runs)...")
    equilibria = [traj[-1] for traj in trajectories]
    fig = dist_plotter.plot_equilibrium_distribution(
        np.array(equilibria), gamma=0.65, theoretical_equilibria=None
    )
    fig.savefig(output_dir / "22_equilibrium_distribution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[23/25] Q-Q plot (preference distribution)...")
    fig = dist_plotter.plot_qq_plot(
        theta[:, 0], distribution='beta', params=(2, 2)
    )
    fig.savefig(output_dir / "23_qq_plot.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[24/25] Belief scatter (true vs posterior)...")
    # Skip this plot - requires ConsumerState
    print("  (Skipped - requires consumer state tracking)")
    # fig = dist_plotter.plot_belief_scatter(
    #     result_low.consumer_state, n_samples=150, show_diagonal=True
    # )
    # fig.savefig(output_dir / "24_belief_scatter.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)

    print("[25/25] Experience distribution...")
    # Skip this plot - requires ConsumerState
    print("  (Skipped - requires consumer state tracking)")
    # fig = dist_plotter.plot_experience_distribution(result_low.consumer_state)
    # fig.savefig(output_dir / "25_experience_distribution.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ GALLERY COMPLETE!")
    print("=" * 70)
    print(f"\nCreated 22 publication-quality plots in: {output_dir}/")
    print("\nPlot Categories:")
    print("  • Trajectory Plots (1-5): Time series, welfare, critical slowing")
    print("  • Phase Portraits (6-11): BR curves, potentials, dynamics")
    print("  • Bifurcation Diagrams (12-15): Bifurcation, scaling, stability")
    print("  • Heatmaps (16-19): Parameter space, correlations")
    print("  • Distributions (20-23): Preferences, equilibria, Q-Q plots")
    print("\nNote: 3 plots skipped (require consumer state tracking)")
    print("All plots saved as high-resolution PNG (300 DPI)")
    print("=" * 70)


if __name__ == '__main__':
    main()
