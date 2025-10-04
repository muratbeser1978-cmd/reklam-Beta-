"""
Complete Tutorial: RKL Simulation & Analysis (Phase 1 & 2)

This tutorial demonstrates:
1. Core simulation (basic + batch)
2. Phase 1 analysis modules (critical slowing, welfare, etc.)
3. Phase 2 visualization suite (5 plotter classes)

Runtime: ~3-5 minutes
Output: outputs/tutorial/ with analysis results and plots
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine

# Phase 1 analyzers
from src.analysis.critical_slowing import CriticalSlowingAnalyzer
from src.analysis.welfare_corrected import WelfareCalculatorCorrected
from src.analysis.network_effects import NetworkEffectsAnalyzer

# Phase 2 plotters
from src.visualization import (
    TrajectoryPlotter,
    PhasePortraitPlotter,
    BifurcationPlotter,
    HeatmapPlotter,
    DistributionPlotter,
)


def main():
    """Run complete tutorial."""
    # Setup
    output_dir = Path("outputs/tutorial")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RKL COMPLETE TUTORIAL - PHASE 1 & 2")
    print("=" * 80)
    print(f"Output directory: {output_dir}\n")

    # =========================================================================
    # PART 1: BASIC SIMULATION
    # =========================================================================
    print("PART 1: BASIC SIMULATION")
    print("-" * 80)

    # Create configuration
    config = SimulationConfig(
        N=500,          # 500 consumers
        T=300,          # 300 timesteps
        gamma=0.65,     # Individual taste weight (below critical)
        beta=0.9,       # Retention rate
        seed=42         # Reproducibility
    )

    print(f"Configuration:")
    print(f"  N={config.N}, T={config.T}, γ={config.gamma:.3f}, β={config.beta:.2f}")
    print(f"  γ* = {12/17:.4f} (critical threshold)")
    print(f"  Regime: γ < γ* (asymmetric equilibria expected)")
    print()

    # Run simulation
    print("Running simulation...")
    engine = SimulationEngine()
    result = engine.run(config)

    print(f"✓ Simulation complete")
    print(f"  Final market share p₁: {result.p1_trajectory[-1]:.4f}")
    print(f"  Trajectory length: {len(result.p1_trajectory)}")
    print()

    # =========================================================================
    # PART 2: PHASE 1 ANALYSIS
    # =========================================================================
    print("PART 2: PHASE 1 ANALYSIS MODULES")
    print("-" * 80)

    # 2.1 Critical Slowing Down
    print("[1/3] Critical Slowing Down Analysis...")
    csd_analyzer = CriticalSlowingAnalyzer()
    csd_metrics = csd_analyzer.compute_metrics(
        result.p1_trajectory,
        gamma=config.gamma,
        burnin=50
    )

    print(f"  Distance to critical γ*: {csd_metrics.distance_to_critical:.4f}")
    print(f"  Variance: {csd_metrics.variance:.6f}")
    print(f"  Autocorrelation AC(1): {csd_metrics.autocorr_lag1:.4f}")
    if csd_metrics.relaxation_time is not None:
        print(f"  Relaxation time: {csd_metrics.relaxation_time:.1f}")
    print(f"  Skewness: {csd_metrics.skewness:.4f}")
    print()

    # 2.2 Welfare Analysis
    print("[2/3] Welfare Analysis (Corrected - Eq 64)...")
    welfare_calc = WelfareCalculatorCorrected()

    # Generate consumer preferences (not stored in result)
    np.random.seed(42)
    theta = np.random.beta(config.alpha_0, config.beta_0, size=(config.N, 2))

    welfare = welfare_calc.compute_welfare(
        result.p1_trajectory,
        theta,
        gamma=config.gamma,
        choices=result.choices
    )

    print(f"  Ex-post welfare (final): {welfare.welfare_trajectory[-1]:.4f}")
    print(f"  Optimal welfare W*: {welfare.welfare_optimal:.4f}")
    print(f"  Equilibrium welfare W(p*): {welfare.welfare_equilibrium:.4f}")
    print(f"  Welfare loss ΔW: {welfare.welfare_loss:.4f}")
    print(f"  Relative loss: {100 * welfare.welfare_loss / welfare.welfare_optimal:.2f}%")
    print()

    # 2.3 Network Effects
    print("[3/3] Network Effects Analysis...")
    network_analyzer = NetworkEffectsAnalyzer()
    social_mult = network_analyzer.compute_social_multiplier(gamma=config.gamma)

    print(f"  Social multiplier M(γ): {social_mult.multiplier:.3f}")
    print(f"  Regime: {social_mult.regime}")
    print(f"  Network strength (1-γ): {1 - config.gamma:.3f}")
    print(f"  Interpretation: {social_mult.interpretation}")
    print()

    # =========================================================================
    # PART 3: PHASE 2 VISUALIZATION
    # =========================================================================
    print("PART 3: PHASE 2 VISUALIZATION SUITE")
    print("-" * 80)

    # Initialize all plotters
    traj_plotter = TrajectoryPlotter()
    phase_plotter = PhasePortraitPlotter()
    bifurc_plotter = BifurcationPlotter()
    heatmap_plotter = HeatmapPlotter()
    dist_plotter = DistributionPlotter()

    # 3.1 Trajectory Plots
    print("[1/8] Plotting market share trajectory...")
    fig = traj_plotter.plot_market_share_trajectory(
        result.p1_trajectory,
        gamma=config.gamma,
        burnin=50
    )
    fig.savefig(output_dir / "01_trajectory.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[2/8] Plotting welfare trajectory...")
    fig = traj_plotter.plot_welfare_trajectory(
        welfare,
        burnin=50,
        show_components=True
    )
    fig.savefig(output_dir / "02_welfare.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[3/8] Plotting critical slowing down indicators...")
    fig = traj_plotter.plot_critical_slowing_down(
        result.p1_trajectory,
        csd_metrics,
        window_size=50
    )
    fig.savefig(output_dir / "03_critical_slowing.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3.2 Phase Portraits
    print("[4/8] Plotting best response curve...")
    fig = phase_plotter.plot_best_response(
        gamma=config.gamma,
        show_stability=True,
        show_equilibria=True
    )
    fig.savefig(output_dir / "04_br_curve.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("[5/8] Plotting potential landscape...")
    fig = phase_plotter.plot_potential_landscape(
        gamma=config.gamma,
        show_equilibria=True,
        show_derivative=True
    )
    fig.savefig(output_dir / "05_potential.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3.3 Bifurcation Diagram
    print("[6/8] Generating bifurcation diagram...")
    gamma_values = np.linspace(0.60, 0.80, 50)
    fig = bifurc_plotter.plot_bifurcation_diagram(
        gamma_values=gamma_values,
        show_stability=True,
        show_critical_line=True
    )
    fig.savefig(output_dir / "06_bifurcation.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3.4 Parameter Sweep Heatmap
    print("[7/8] Creating parameter space heatmap...")
    # Small grid for demo
    gamma_grid = np.linspace(0.60, 0.75, 10)
    beta_grid = np.linspace(0.85, 0.95, 8)
    variance_grid = np.zeros((len(beta_grid), len(gamma_grid)))

    for i, beta in enumerate(beta_grid):
        for j, gamma in enumerate(gamma_grid):
            cfg = SimulationConfig(N=200, T=200, gamma=gamma, beta=beta, seed=42)
            res = engine.run(cfg)
            variance_grid[i, j] = np.var(res.p1_trajectory[-50:])

    fig = heatmap_plotter.plot_variance_heatmap(
        gamma_grid,
        beta_grid,
        variance_grid,
        log_scale=True,
        mark_critical=True
    )
    fig.savefig(output_dir / "07_variance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3.5 Distribution Plot
    print("[8/8] Plotting preference distribution...")
    fig = dist_plotter.plot_preference_distribution(
        theta,
        show_kde=True,
        show_theory=True
    )
    fig.savefig(output_dir / "08_preferences.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print()

    # =========================================================================
    # PART 4: PARAMETER SWEEP EXAMPLE
    # =========================================================================
    print("PART 4: PARAMETER SWEEP (Batch Execution)")
    print("-" * 80)

    # Create configurations for different gamma values
    gamma_sweep = np.linspace(0.60, 0.80, 11)
    configs = [
        SimulationConfig(N=300, T=250, gamma=g, beta=0.9, seed=42)
        for g in gamma_sweep
    ]

    print(f"Running {len(configs)} simulations in parallel...")
    results_sweep = engine.run_batch(configs, n_workers=4)

    # Analyze each result
    print("Analyzing sweep results...")
    sweep_analysis = {
        'gamma': [],
        'variance': [],
        'autocorr': [],
        'welfare_loss': [],
        'final_p1': []
    }

    for cfg, res in zip(configs, results_sweep):
        # Critical slowing
        csd = csd_analyzer.compute_metrics(res.p1_trajectory, cfg.gamma, burnin=50)

        # Welfare
        theta_sweep = np.random.beta(2, 2, size=(cfg.N, 2))
        welf = welfare_calc.compute_welfare(
            res.p1_trajectory, theta_sweep, cfg.gamma, res.choices
        )

        sweep_analysis['gamma'].append(cfg.gamma)
        sweep_analysis['variance'].append(csd.variance)
        sweep_analysis['autocorr'].append(csd.autocorr_lag1)
        sweep_analysis['welfare_loss'].append(welf.welfare_loss)
        sweep_analysis['final_p1'].append(res.p1_trajectory[-1])

    # Plot sweep results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Variance
    ax1.plot(sweep_analysis['gamma'], sweep_analysis['variance'], 'o-', linewidth=2)
    ax1.axvline(12/17, color='red', linestyle='--', alpha=0.7, label='γ*')
    ax1.set_xlabel('Individual Taste Weight γ')
    ax1.set_ylabel('Variance')
    ax1.set_title('Variance vs γ (Critical Slowing Down)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Autocorrelation
    ax2.plot(sweep_analysis['gamma'], sweep_analysis['autocorr'], 'o-', linewidth=2, color='orange')
    ax2.axvline(12/17, color='red', linestyle='--', alpha=0.7, label='γ*')
    ax2.set_xlabel('Individual Taste Weight γ')
    ax2.set_ylabel('Autocorrelation AC(1)')
    ax2.set_title('Autocorrelation vs γ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Welfare loss
    ax3.plot(sweep_analysis['gamma'], sweep_analysis['welfare_loss'], 'o-', linewidth=2, color='green')
    ax3.axvline(12/17, color='red', linestyle='--', alpha=0.7, label='γ*')
    ax3.set_xlabel('Individual Taste Weight γ')
    ax3.set_ylabel('Welfare Loss ΔW')
    ax3.set_title('Welfare Loss vs γ')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Final equilibrium
    ax4.plot(sweep_analysis['gamma'], sweep_analysis['final_p1'], 'o-', linewidth=2, color='purple')
    ax4.axvline(12/17, color='red', linestyle='--', alpha=0.7, label='γ*')
    ax4.axhline(0.5, color='black', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Individual Taste Weight γ')
    ax4.set_ylabel('Final Market Share p₁')
    ax4.set_title('Equilibrium vs γ')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "09_parameter_sweep_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Sweep complete: analyzed {len(results_sweep)} simulations")
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 80)
    print("✅ TUTORIAL COMPLETE!")
    print("=" * 80)
    print(f"\nCreated 9 plots in: {output_dir}/\n")

    print("Plot Summary:")
    print("  [1] Market share trajectory")
    print("  [2] Welfare evolution (W, Q̄₁, Q̄₂)")
    print("  [3] Critical slowing down indicators")
    print("  [4] Best response curve with stability")
    print("  [5] Potential landscape V(p,γ)")
    print("  [6] Bifurcation diagram")
    print("  [7] Variance heatmap (γ, β)")
    print("  [8] Preference distribution θ ~ Beta(2,2)")
    print("  [9] Parameter sweep analysis (4 panels)")

    print("\nAnalysis Summary:")
    print(f"  Critical Slowing Down:")
    print(f"    - Distance to γ*: {csd_metrics.distance_to_critical:.4f}")
    print(f"    - Variance: {csd_metrics.variance:.6f}")
    print(f"    - AC(1): {csd_metrics.autocorr_lag1:.4f}")
    print(f"  Welfare:")
    print(f"    - Loss ΔW: {welfare.welfare_loss:.4f}")
    if welfare.welfare_optimal > 0:
        print(f"    - Relative: {100 * welfare.welfare_loss / welfare.welfare_optimal:.2f}%")
    print(f"  Network Effects:")
    print(f"    - Social Multiplier: {social_mult.multiplier:.3f}")
    print(f"    - Regime: {social_mult.regime}")

    print("\nNext Steps:")
    print("  1. View plots in outputs/tutorial/")
    print("  2. Read VISUALIZATION_GUIDE.md for detailed plotter documentation")
    print("  3. Read ANALYSIS_GUIDE.md for Phase 1 analysis modules")
    print("  4. Run examples/advanced_visualization_gallery.py for full gallery (22 plots)")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
