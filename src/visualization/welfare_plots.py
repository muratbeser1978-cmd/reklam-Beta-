"""
Welfare Analysis Visualization.

Plots welfare metrics and welfare loss.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
from src.simulation.result import SimulationResult
from src.analysis.welfare import WelfareCalculator
from src.distributions.constants import GAMMA_STAR


def plot_welfare_loss(
    gamma_grid: np.ndarray,
    welfare_losses: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot welfare loss ΔW as function of gamma.

    Args:
        gamma_grid: Array of gamma values
        welfare_losses: Array of welfare loss values ΔW(γ)
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure object

    Output:
        Shows welfare loss increasing as γ decreases below γ*
        ΔW = 0 for γ ≥ γ* (symmetric equilibrium optimal)
        ΔW > 0 for γ < γ* (asymmetric equilibria suboptimal)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot welfare loss
    ax.plot(gamma_grid, welfare_losses, color="#A23B72", linewidth=2.5, label="Welfare loss $\\Delta W$")

    # Mark critical point
    ax.axvline(
        x=GAMMA_STAR,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"$\\gamma^* = {GAMMA_STAR:.6f}$",
    )

    # Shade regions
    ax.axvspan(gamma_grid.min(), GAMMA_STAR, alpha=0.1, color="red", label="Welfare loss present")
    ax.axvspan(GAMMA_STAR, gamma_grid.max(), alpha=0.1, color="green", label="Optimal welfare")

    # Zero line
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel("Learning Weight $\\gamma$", fontsize=14)
    ax.set_ylabel("Welfare Loss $\\Delta W = W^{opt} - W^{eq}$", fontsize=14)
    ax.set_title("Social Welfare Loss vs Learning Weight", fontsize=16, pad=20)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(gamma_grid.min(), gamma_grid.max())

    # Set y-axis to start at 0 if all non-negative
    if np.all(welfare_losses >= -1e-10):
        ax.set_ylim(bottom=-0.01 * welfare_losses.max())

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Welfare loss plot saved to {save_path}")

    return fig


def plot_welfare_trajectory(
    result: SimulationResult,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot welfare W(t) over time for a single simulation.

    Args:
        result: SimulationResult with trajectory
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure object
    """
    # Compute welfare
    calculator = WelfareCalculator()

    # Get consumer preferences from final state (approximation)
    # Note: True theta would need to be stored in result
    # For visualization, we'll use a simplified version
    N = result.config.N
    theta = np.random.beta(result.config.alpha_0, result.config.beta_0, size=(N, 2))

    welfare_analysis = calculator.compute_welfare(
        result.p1_trajectory, theta, result.config.gamma
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    T = len(result.p1_trajectory) - 1
    time = np.arange(T + 1)

    # Plot welfare trajectory
    ax.plot(
        time,
        welfare_analysis.welfare_trajectory,
        color="#2E86AB",
        linewidth=2,
        label="Actual welfare $W(t)$",
    )

    # Plot optimal welfare
    ax.axhline(
        y=welfare_analysis.welfare_optimal,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Optimal $W^{{opt}}={welfare_analysis.welfare_optimal:.4f}$",
    )

    # Plot equilibrium welfare
    ax.axhline(
        y=welfare_analysis.welfare_equilibrium,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Equilibrium $W^{{eq}}={welfare_analysis.welfare_equilibrium:.4f}$",
    )

    # Styling
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Social Welfare $W(t)$", fontsize=12)
    ax.set_title(
        f"Welfare Trajectory ($\\gamma={result.config.gamma:.3f}$, $\\Delta W={welfare_analysis.welfare_loss:.4f}$)",
        fontsize=14,
        pad=15,
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Welfare trajectory plot saved to {save_path}")

    return fig


def plot_welfare_comparison(
    sweep_results: List[Tuple[float, float]],
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Compare optimal vs equilibrium welfare across gamma values.

    Args:
        sweep_results: List of (gamma, welfare_loss) tuples
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure object
    """
    # Sort by gamma
    sweep_results = sorted(sweep_results, key=lambda x: x[0])
    gammas = np.array([x[0] for x in sweep_results])
    welfare_losses = np.array([x[1] for x in sweep_results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Left: Welfare loss
    ax1.plot(gammas, welfare_losses, "o-", color="#A23B72", linewidth=2, markersize=5)
    ax1.axvline(x=GAMMA_STAR, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax1.set_xlabel("$\\gamma$", fontsize=12)
    ax1.set_ylabel("$\\Delta W$", fontsize=12)
    ax1.set_title("Welfare Loss", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Right: Relative welfare loss (as percentage)
    # Assuming W_opt ≈ 0.5 for Beta(2,2) preferences
    relative_loss = welfare_losses / 0.5 * 100  # Convert to percentage

    ax2.plot(gammas, relative_loss, "s-", color="#F18F01", linewidth=2, markersize=5)
    ax2.axvline(x=GAMMA_STAR, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax2.set_xlabel("$\\gamma$", fontsize=12)
    ax2.set_ylabel("Relative loss (% of $W^{opt}$)", fontsize=12)
    ax2.set_title("Relative Welfare Loss", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Welfare comparison plot saved to {save_path}")

    return fig


def plot_welfare_decomposition(
    gamma_grid: np.ndarray,
    welfare_optimal: np.ndarray,
    welfare_equilibrium: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot welfare decomposition: W_opt, W_eq, and gap ΔW.

    Args:
        gamma_grid: Array of gamma values
        welfare_optimal: W^opt for each gamma
        welfare_equilibrium: W^eq for each gamma
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot optimal and equilibrium welfare
    ax.plot(
        gamma_grid,
        welfare_optimal,
        color="green",
        linewidth=2.5,
        label="Optimal welfare $W^{opt}$",
    )
    ax.plot(
        gamma_grid,
        welfare_equilibrium,
        color="red",
        linewidth=2.5,
        label="Equilibrium welfare $W^{eq}$",
    )

    # Fill gap (welfare loss)
    ax.fill_between(
        gamma_grid,
        welfare_optimal,
        welfare_equilibrium,
        alpha=0.3,
        color="red",
        label="Welfare loss $\\Delta W$",
    )

    # Mark critical point
    ax.axvline(
        x=GAMMA_STAR,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"$\\gamma^* = {GAMMA_STAR:.6f}$",
    )

    # Styling
    ax.set_xlabel("Learning Weight $\\gamma$", fontsize=14)
    ax.set_ylabel("Social Welfare", fontsize=14)
    ax.set_title("Welfare Decomposition", fontsize=16, pad=20)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(gamma_grid.min(), gamma_grid.max())

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Welfare decomposition plot saved to {save_path}")

    return fig
