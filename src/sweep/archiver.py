"""
Provenance Archiver.

Saves simulation results with complete provenance metadata.
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import numpy as np
import scipy
from src.simulation.result import SimulationResult


def get_git_commit() -> str:
    """Get current git commit hash (or 'unknown' if not in git repo)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return commit.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_environment_info() -> Dict[str, str]:
    """Get Python environment information."""
    import sys

    return {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def save_with_provenance(
    result: SimulationResult, output_path: Path, extra_metadata: Dict[str, Any] = None
) -> None:
    """
    Save simulation result with complete provenance metadata.

    Args:
        result: SimulationResult to save
        output_path: Path to .npz file
        extra_metadata: Optional additional metadata to include

    Output:
        - {output_path}.npz: Compressed NumPy arrays (trajectories, choices, rewards)
        - {output_path}.json: Complete provenance metadata

    Provenance metadata includes:
        - Timestamp (ISO 8601)
        - Git commit hash
        - Python/NumPy/SciPy versions
        - Full simulation config
        - Runtime statistics
        - Custom metadata from extra_metadata
    """
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save arrays to .npz (this is already done by result.save())
    result.save(output_path)

    # Build comprehensive provenance metadata
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "environment": get_environment_info(),
        "config": result.config.to_dict(),
        "runtime_seconds": result.metadata.get("runtime_seconds", None),
        "final_p1": float(result.p1_trajectory[-1]),
        "final_p2": float(result.p2_trajectory[-1]),
    }

    # Add extra metadata if provided
    if extra_metadata:
        provenance["extra"] = extra_metadata

    # Save provenance to JSON file
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(provenance, f, indent=2)


def load_with_provenance(output_path: Path) -> Tuple[SimulationResult, Dict[str, Any]]:
    """
    Load simulation result with provenance metadata.

    Args:
        output_path: Path to .npz file

    Returns:
        (result, provenance): SimulationResult and provenance dict

    Raises:
        FileNotFoundError: If .npz or .json file missing
    """
    output_path = Path(output_path)

    # Load result from .npz
    result = SimulationResult.load(output_path)

    # Load provenance from .json
    json_path = output_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Provenance file not found: {json_path}")

    with open(json_path, "r") as f:
        provenance = json.load(f)

    return result, provenance


def load_sweep_results(sweep_dir: Path) -> list[Tuple[SimulationResult, Dict[str, Any]]]:
    """
    Load all results from a parameter sweep directory.

    Args:
        sweep_dir: Directory containing sweep results

    Returns:
        List of (result, provenance) tuples for each .npz file

    Example:
        results = load_sweep_results(Path("outputs/sweep_20250103"))
        for result, prov in results:
            gamma = result.config.gamma
            runtime = prov["runtime_seconds"]
            print(f"γ={gamma:.4f}: {runtime:.2f}s")
    """
    sweep_dir = Path(sweep_dir)

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    # Find all .npz files
    npz_files = sorted(sweep_dir.glob("*.npz"))

    if not npz_files:
        raise ValueError(f"No .npz files found in {sweep_dir}")

    # Load each result with provenance
    results = []
    for npz_path in npz_files:
        result, provenance = load_with_provenance(npz_path)
        results.append((result, provenance))

    return results
