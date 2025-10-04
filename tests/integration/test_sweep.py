"""
Integration tests for parameter sweep functionality.

Tests T046 from tasks.md.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from src.simulation.config import SimulationConfig
from src.sweep.config import SweepConfig
from src.sweep.runner import run_sweep, run_sweep_with_callback
from src.sweep.archiver import (
    save_with_provenance,
    load_with_provenance,
    load_sweep_results,
)


class TestParameterSweep:
    """Test parameter sweep execution and archiving."""

    def test_sweep_execution(self):
        """Run sweep with 5 gamma × 3 seeds = 15 runs."""
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create base config (small for speed)
            base_config = SimulationConfig(
                N=100, T=50, gamma=0.7, beta=0.9, seed=42, p_init=0.5
            )

            # Create sweep: 5 gamma values × 3 seeds = 15 runs
            gamma_values = np.linspace(0.6, 0.8, 5)
            seeds = [42, 123, 456]

            sweep_config = SweepConfig.create_gamma_sweep(
                gamma_min=0.6,
                gamma_max=0.8,
                n_gamma=5,
                seeds=seeds,
                base_config=base_config,
                output_dir=output_dir,
            )

            # Run sweep
            results = run_sweep(sweep_config, n_workers=2, save_immediately=True)

            # Verify 15 results returned
            assert len(results) == 15, f"Expected 15 results, got {len(results)}"

            # Verify 15 .npz files saved
            npz_files = list(output_dir.glob("*.npz"))
            assert len(npz_files) == 15, f"Expected 15 .npz files, got {len(npz_files)}"

            # Verify each result has correct gamma and seed
            gamma_set = set(gamma_values)
            seed_set = set(seeds)

            for result in results:
                assert result.config.gamma in gamma_set, "Unexpected gamma value"
                assert result.config.seed in seed_set, "Unexpected seed value"

                # Verify result is valid
                assert result.p1_trajectory.shape == (51,), "Trajectory length mismatch"
                assert result.choices.shape == (100, 50), "Choices shape mismatch"

    def test_unique_seeds(self):
        """Verify each output has unique seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            base_config = SimulationConfig(
                N=50, T=20, gamma=0.7, beta=0.9, seed=42, p_init=0.5
            )

            # Single gamma, 3 different seeds
            sweep_config = SweepConfig(
                gamma_values=np.array([0.7]),
                seeds=[10, 20, 30],
                base_config=base_config,
                output_dir=output_dir,
            )

            results = run_sweep(sweep_config, n_workers=2, save_immediately=False)

            # Verify 3 results with different seeds
            assert len(results) == 3
            seeds_used = [r.config.seed for r in results]
            assert sorted(seeds_used) == [10, 20, 30], "Seeds not unique"

            # Verify results are different (stochastic variation)
            p1_final_values = [r.p1_trajectory[-1] for r in results]
            # At least one pair should be different (very high probability with different seeds)
            assert not all(
                abs(p1_final_values[i] - p1_final_values[j]) < 1e-10
                for i in range(3)
                for j in range(i + 1, 3)
            ), "All results identical despite different seeds"

    def test_provenance_metadata(self):
        """Check provenance metadata completeness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            config = SimulationConfig(
                N=50, T=20, gamma=0.7, beta=0.9, seed=42, p_init=0.5
            )

            sweep_config = SweepConfig(
                gamma_values=np.array([0.7]),
                seeds=[42],
                base_config=config,
                output_dir=output_dir,
            )

            results = run_sweep(sweep_config, n_workers=1, save_immediately=True)

            # Load with provenance
            npz_files = list(output_dir.glob("*.npz"))
            assert len(npz_files) == 1

            result, provenance = load_with_provenance(npz_files[0])

            # Check required provenance fields
            assert "timestamp" in provenance, "Missing timestamp"
            assert "git_commit" in provenance, "Missing git commit"
            assert "environment" in provenance, "Missing environment info"
            assert "config" in provenance, "Missing config"
            assert "runtime_seconds" in provenance, "Missing runtime"
            assert "final_p1" in provenance, "Missing final_p1"
            assert "final_p2" in provenance, "Missing final_p2"

            # Verify environment info
            env = provenance["environment"]
            assert "python_version" in env
            assert "numpy_version" in env
            assert "scipy_version" in env

            # Verify config matches
            saved_config = provenance["config"]
            assert saved_config["N"] == 50
            assert saved_config["T"] == 20
            assert abs(saved_config["gamma"] - 0.7) < 1e-10
            assert saved_config["seed"] == 42

    def test_load_sweep_results(self):
        """Test loading all results from sweep directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            base_config = SimulationConfig(
                N=50, T=20, gamma=0.7, beta=0.9, seed=42, p_init=0.5
            )

            # Create sweep with 3 gamma × 2 seeds = 6 runs
            sweep_config = SweepConfig.create_gamma_sweep(
                gamma_min=0.6,
                gamma_max=0.8,
                n_gamma=3,
                seeds=[42, 123],
                base_config=base_config,
                output_dir=output_dir,
            )

            run_sweep(sweep_config, n_workers=2, save_immediately=True)

            # Load all results
            loaded_results = load_sweep_results(output_dir)

            assert len(loaded_results) == 6, "Should load 6 results"

            # Verify structure
            for result, provenance in loaded_results:
                assert hasattr(result, "p1_trajectory")
                assert "timestamp" in provenance
                assert "config" in provenance

    def test_sweep_with_callback(self):
        """Test sweep execution with custom callback."""
        callback_count = {"count": 0}

        def my_callback(result):
            callback_count["count"] += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            base_config = SimulationConfig(
                N=50, T=20, gamma=0.7, beta=0.9, seed=42, p_init=0.5
            )

            sweep_config = SweepConfig(
                gamma_values=np.array([0.6, 0.7, 0.8]),
                seeds=[42, 123],
                base_config=base_config,
                output_dir=output_dir,
            )

            results = run_sweep_with_callback(
                sweep_config, callback=my_callback, n_workers=2
            )

            # Callback should be called 6 times (3 gamma × 2 seeds)
            assert callback_count["count"] == 6, "Callback not called correct number of times"
            assert len(results) == 6, "Should return 6 results"

    def test_extra_metadata(self):
        """Test saving with extra custom metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_result.npz"

            config = SimulationConfig(
                N=50, T=20, gamma=0.7, beta=0.9, seed=42, p_init=0.5
            )

            from src.simulation.engine import SimulationEngine

            engine = SimulationEngine()
            result = engine.run(config)

            # Save with extra metadata
            extra = {"experiment_name": "test_sweep", "notes": "Testing extra metadata"}

            save_with_provenance(result, output_path, extra_metadata=extra)

            # Load and verify
            loaded_result, provenance = load_with_provenance(output_path)

            assert "extra" in provenance, "Extra metadata not saved"
            assert provenance["extra"]["experiment_name"] == "test_sweep"
            assert provenance["extra"]["notes"] == "Testing extra metadata"
