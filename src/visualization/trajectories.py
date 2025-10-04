"""
Trajectory Visualization.

Plots market share trajectories over time.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
from src.simulation.result import SimulationResult
from src.analysis.equilibrium import EquilibriumFinder


def plot_trajectory(
    result: SimulationResult,
    save_path: Optional[Path] = None,
    show_equilibria: bool = True,
    figsize: tuple = (8, 5),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot market share trajectory p₁(t) over time.

    Args:
        result: SimulationResult with trajectory data
        save_path: Path to save figure (PDF format). If None, returns figure without saving
        show_equilibria: If True, show equilibrium lines
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object

    Output:
        Publication-quality figure showing:
        - p₁(t) trajectory over time
        - p₂(t) trajectory (complementary)
        - Equilibrium lines (if show_equilibria=True)
        - Grid, labels, legend
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Time axis
    T = len(result.p1_trajectory) - 1
    time = np.arange(T + 1)

    # Plot trajectories
    ax.plot(
        time,
        result.p1_trajectory,
        label="$p_1(t)$ (Brand 1)",
        color="#2E86AB",
        linewidth=2,
    )
    ax.plot(
        time,
        result.p2_trajectory,
        label="$p_2(t)$ (Brand 2)",
        color="#A23B72",
        linewidth=2,
        linestyle="--",
    )

    # Show equilibria if requested
    if show_equilibria:
        finder = EquilibriumFinder()
        equilibria = finder.find_equilibria(result.config.gamma)

        for eq in equilibria:
            linestyle = "-" if eq.stable else ":"
            alpha = 0.7 if eq.stable else 0.4
            label = f"Equilibrium ($p^*={eq.p_star:.3f}$, {'stable' if eq.stable else 'unstable'})"

            ax.axhline(
                y=eq.p_star,
                color="gray",
                linestyle=linestyle,
                alpha=alpha,
                linewidth=1.5,
                label=label,
            )

    # Styling
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Market Share", fontsize=12)
    ax.set_title(
        f"Market Share Dynamics ($\\gamma={result.config.gamma:.3f}$, $N={result.config.N}$)",
        fontsize=14,
        pad=15,
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.05)

    # Add text annotation with final values
    final_p1 = result.p1_trajectory[-1]
    ax.text(
        0.02,
        0.98,
        f"Final: $p_1={final_p1:.4f}$",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Trajectory plot saved to {save_path}")

    return fig


def plot_multiple_trajectories(
    results: List[SimulationResult],
    save_path: Optional[Path] = None,
    labels: Optional[List[str]] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot multiple trajectories on same axes (e.g., different seeds or gamma values).

    Args:
        results: List of SimulationResults
        save_path: Path to save figure
        labels: Optional labels for each trajectory
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color map for multiple trajectories
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))

    for i, result in enumerate(results):
        T = len(result.p1_trajectory) - 1
        time = np.arange(T + 1)

        label = labels[i] if labels else f"Run {i+1}"

        ax.plot(
            time, result.p1_trajectory, label=label, color=colors[i], linewidth=1.5, alpha=0.8
        )

    # Styling
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Market Share $p_1(t)$", fontsize=12)
    ax.set_title("Market Share Trajectories", fontsize=14, pad=15)
    ax.legend(loc="best", fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Multiple trajectories plot saved to {save_path}")

    return fig


def plot_trajectory_with_variance(
    results: List[SimulationResult],
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot mean trajectory with variance bands (from multiple runs with different seeds).

    Args:
        results: List of SimulationResults with same config but different seeds
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Stack all trajectories
    trajectories = np.array([r.p1_trajectory for r in results])

    # Compute mean and std
    mean_traj = np.mean(trajectories, axis=0)
    std_traj = np.std(trajectories, axis=0)

    T = len(mean_traj) - 1
    time = np.arange(T + 1)

    # Plot mean
    ax.plot(time, mean_traj, label="Mean $p_1(t)$", color="#2E86AB", linewidth=2.5)

    # Plot variance band
    ax.fill_between(
        time,
        mean_traj - std_traj,
        mean_traj + std_traj,
        alpha=0.3,
        color="#2E86AB",
        label="$\\pm 1$ std",
    )

    # Styling
    gamma = results[0].config.gamma
    n_runs = len(results)

    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Market Share $p_1(t)$", fontsize=12)
    ax.set_title(
        f"Mean Trajectory with Variance ($\\gamma={gamma:.3f}$, {n_runs} runs)",
        fontsize=14,
        pad=15,
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Trajectory with variance plot saved to {save_path}")

    return fig
