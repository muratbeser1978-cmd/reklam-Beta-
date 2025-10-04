"""
Parameter Sweep Runner.

Executes simulations across parameter grids with parallel processing.
"""
from typing import List, Optional
from pathlib import Path
from multiprocessing import cpu_count
from src.simulation.engine import SimulationEngine
from src.simulation.result import SimulationResult
from src.sweep.config import SweepConfig


def run_sweep(
    sweep_config: SweepConfig,
    n_workers: Optional[int] = None,
    save_immediately: bool = True,
) -> List[SimulationResult]:
    """
    Execute parameter sweep with parallel processing.

    Args:
        sweep_config: Sweep configuration (gamma values × seeds)
        n_workers: Number of parallel workers (default: cpu_count())
        save_immediately: If True, save each result to disk during execution

    Returns:
        List of SimulationResults for all parameter combinations

    Implementation:
        - Generates all configs: gamma_values × seeds
        - Uses multiprocessing.Pool via SimulationEngine.run_batch
        - Chunksize = total_runs / (2 * cpu_count()) for load balancing
        - Optionally saves results to sweep_config.output_dir

    Performance:
        - N=1000, T=1000: ~10s per run
        - 50 gamma × 5 seeds = 250 runs: ~40 minutes on 8 cores
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Generate all simulation configs
    configs = sweep_config.generate_configs()
    total_runs = len(configs)

    print(f"Running parameter sweep: {total_runs} configurations")
    print(f"  Gamma values: {len(sweep_config.gamma_values)}")
    print(f"  Seeds: {len(sweep_config.seeds)}")
    print(f"  Parallel workers: {n_workers}")

    # Create output directory if saving
    if save_immediately:
        sweep_config.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {sweep_config.output_dir}")

    # Execute sweep using batch runner
    engine = SimulationEngine()
    results = engine.run_batch(configs, n_workers=n_workers)

    # Save results if requested
    if save_immediately:
        print(f"\nSaving results...")
        for i, result in enumerate(results):
            gamma = result.config.gamma
            seed = result.config.seed

            from src.sweep.archiver import save_with_provenance

            # Create filename: sweep_gamma{gamma:.4f}_seed{seed}.npz
            filename = f"sweep_gamma{gamma:.4f}_seed{seed:05d}" # Note: no .npz here
            output_path = sweep_config.output_dir / filename

            save_with_provenance(result, output_path)

            if (i + 1) % 10 == 0:
                print(f"  Saved {i + 1}/{total_runs} results")

        print(f"All {total_runs} results saved to {sweep_config.output_dir}")

    print(f"\n✓ Sweep complete: {total_runs} simulations")

    return results


def run_sweep_with_callback(
    sweep_config: SweepConfig,
    callback: callable,
    n_workers: Optional[int] = None,
) -> List[SimulationResult]:
    """
    Execute parameter sweep with custom callback per result.

    Args:
        sweep_config: Sweep configuration
        callback: Function (result) -> None called after each simulation
        n_workers: Number of parallel workers

    Returns:
        List of SimulationResults

    Example:
        def my_callback(result):
            print(f"Completed γ={result.config.gamma:.4f}")

        run_sweep_with_callback(sweep_config, my_callback)
    """
    if n_workers is None:
        n_workers = cpu_count()

    configs = sweep_config.generate_configs()
    engine = SimulationEngine()

    results = engine.run_batch(configs, n_workers=n_workers)

    # Apply callback to each result
    for result in results:
        callback(result)

    return results
