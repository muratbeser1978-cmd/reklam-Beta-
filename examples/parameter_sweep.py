"""
Parameter Sweep Example.

Demonstrates how to run parallel parameter sweeps across gamma values.
"""
from pathlib import Path
from src.simulation.config import SimulationConfig
from src.sweep import SweepConfig, run_sweep
from src.sweep.archiver import load_sweep_results
import numpy as np
import shutil

def main():
    """Main function to run the parameter sweep."""
    # Create / clean output directory
    output_dir = Path("outputs/examples/sweep")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print("Parameter Sweep Example")
    print("=" * 60)

    # Create base configuration
    base_config = SimulationConfig(
        N=500,            # 500 consumers (smaller for speed)
        T=200,            # 200 time steps
        gamma=0.7,        # Will be varied in sweep
        beta=0.9,
        seed=42,
        p_init=0.5
    )

    # Create sweep configuration
    # Sweep across 10 gamma values with 3 different seeds
    sweep = SweepConfig.create_gamma_sweep(
        gamma_min=0.60,
        gamma_max=0.80,
        n_gamma=10,              # 10 gamma values
        seeds=[42, 123, 456],    # 3 seeds
        base_config=base_config,
        output_dir=output_dir
    )

    print(f"Sweep configuration:")
    print(f"  Gamma range: [{sweep.gamma_values.min():.3f}, {sweep.gamma_values.max():.3f}]")
    print(f"  Number of gamma values: {len(sweep.gamma_values)}")
    print(f"  Seeds: {sweep.seeds}")
    print(f"  Total simulations: {len(sweep.gamma_values) * len(sweep.seeds)}")
    print(f"  Output directory: {output_dir}")
    print()

    # Run sweep with parallel processing
    print("Running parameter sweep...")
    results = run_sweep(
        sweep,
        n_workers=4,              # Use 4 CPU cores
        save_immediately=True     # Save each result as it completes
    )

    print(f"\n✓ Sweep complete!")
    print(f"  Total simulations: {len(results)}")

    # Analyze results
    print("\nAnalyzing results...")

    # Group by gamma
    gamma_groups = {}
    for result in results:
        gamma = result.config.gamma
        if gamma not in gamma_groups:
            gamma_groups[gamma] = []
        gamma_groups[gamma].append(result.p1_trajectory[-1])

    # Compute mean final market share for each gamma
    print("\nFinal market share by γ:")
    print("-" * 60)
    for gamma in sorted(gamma_groups.keys()):
        final_shares = gamma_groups[gamma]
        mean_share = np.mean(final_shares)
        std_share = np.std(final_shares)
        print(f"  γ = {gamma:.3f}: p₁(T) = {mean_share:.4f} ± {std_share:.4f}")

    # Load results with provenance
    print("\nLoading results with provenance metadata...")
    loaded_results = load_sweep_results(output_dir)
    print(f"✓ Loaded {len(loaded_results)} results with provenance")

    # Show provenance for first result
    if loaded_results:
        result, provenance = loaded_results[0]
        print(f"\nExample provenance:")
        print(f"  Timestamp: {provenance['timestamp']}")
        print(f"  Git commit: {provenance['git_commit'][:8]}...")
        print(f"  Python version: {provenance['environment']['python_version'].split()[0]}")
        print(f"  Runtime: {provenance['runtime_seconds']:.2f}s")

    print(f"\n✓ Done! Results saved in {output_dir}")

if __name__ == '__main__':
    main()