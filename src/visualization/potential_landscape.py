"""
Potential Landscape Visualization.

3D surface plots of potential function V(p, γ).
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, List
from src.analysis.potential import potential
from src.distributions.constants import GAMMA_STAR


def plot_potential_landscape(
    gamma_values: Optional[List[float]] = None,
    p_grid_resolution: int = 100,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
    dpi: int = 300,
    plot_3d: bool = True,
) -> plt.Figure:
    """
    Plot potential landscape V(p, γ) as 3D surface or contour plot.

    Args:
        gamma_values: List of gamma values to compute. If None, use range [0.5, 0.9]
        p_grid_resolution: Resolution for p axis
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
        plot_3d: If True, create 3D surface plot. If False, create contour plot

    Returns:
        matplotlib Figure object

    Output:
        Shows how potential shape changes with gamma:
        - γ > γ*: Single minimum at p=0.5
        - γ < γ*: Local maximum at p=0.5, two minima at asymmetric points
    """
    if gamma_values is None:
        gamma_values = np.linspace(0.5, 0.9, 30)

    # Create p grid
    p_grid = np.linspace(0.01, 0.99, p_grid_resolution)

    # Compute potential for all (p, γ) combinations
    P, GAMMA = np.meshgrid(p_grid, gamma_values)
    V = np.zeros_like(P)

    print(f"Computing potential landscape: {len(gamma_values)} × {p_grid_resolution} grid...")

    for i, gamma in enumerate(gamma_values):
        for j, p in enumerate(p_grid):
            V[i, j] = potential(p, gamma)

        if (i + 1) % 5 == 0:
            print(f"  Computed {i+1}/{len(gamma_values)} gamma values")

    print("Done computing potential landscape")

    if plot_3d:
        # 3D surface plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Plot surface
        surf = ax.plot_surface(
            P, GAMMA, V, cmap="viridis", alpha=0.9, edgecolor="none", antialiased=True
        )

        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Potential $V(p, \\gamma)$", fontsize=12)

        # Mark critical gamma*
        ax.plot(
            [0.5, 0.5],
            [GAMMA_STAR, GAMMA_STAR],
            [V.min(), V.max()],
            "r--",
            linewidth=2,
            label=f"$\\gamma^*={GAMMA_STAR:.3f}$",
        )

        # Styling
        ax.set_xlabel("Market Share $p$", fontsize=12, labelpad=10)
        ax.set_ylabel("Learning Weight $\\gamma$", fontsize=12, labelpad=10)
        ax.set_zlabel("Potential $V(p, \\gamma)$", fontsize=12, labelpad=10)
        ax.set_title("Potential Landscape", fontsize=16, pad=20)
        ax.view_init(elev=25, azim=45)

    else:
        # 2D contour plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Contour plot
        levels = 20
        contour = ax.contourf(P, GAMMA, V, levels=levels, cmap="viridis", alpha=0.9)
        contour_lines = ax.contour(P, GAMMA, V, levels=levels, colors="black", alpha=0.3, linewidths=0.5)

        # Colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Potential $V(p, \\gamma)$", fontsize=12)

        # Mark critical gamma*
        ax.axhline(
            y=GAMMA_STAR,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"$\\gamma^*={GAMMA_STAR:.3f}$",
        )

        # Mark symmetric equilibrium
        ax.axvline(x=0.5, color="white", linestyle=":", linewidth=1.5, alpha=0.7)

        # Styling
        ax.set_xlabel("Market Share $p$", fontsize=12)
        ax.set_ylabel("Learning Weight $\\gamma$", fontsize=12)
        ax.set_title("Potential Landscape (Contour View)", fontsize=16, pad=15)
        ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Potential landscape saved to {save_path}")

    return fig


def plot_potential_slices(
    gamma_values: List[float] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot potential V(p) for multiple fixed gamma values (slices of 3D landscape).

    Args:
        gamma_values: List of gamma values to plot
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure object

    Output:
        Shows transition from single-well (γ > γ*) to double-well (γ < γ*)
    """
    if gamma_values is None:
        # Default: sample around γ*
        gamma_values = [0.60, 0.65, GAMMA_STAR, 0.75, 0.80]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    p_grid = np.linspace(0.01, 0.99, 200)

    # Color gradient
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(gamma_values)))

    for i, gamma in enumerate(gamma_values):
        V = np.array([potential(p, gamma) for p in p_grid])

        label = f"$\\gamma={gamma:.3f}$"
        if abs(gamma - GAMMA_STAR) < 1e-6:
            label += " ($\\gamma^*$)"
            linewidth = 3
        else:
            linewidth = 2

        ax.plot(p_grid, V, label=label, color=colors[i], linewidth=linewidth)

    # Mark symmetric point
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel("Market Share $p$", fontsize=12)
    ax.set_ylabel("Potential $V(p, \\gamma)$", fontsize=12)
    ax.set_title("Potential Function for Different $\\gamma$ Values", fontsize=14, pad=15)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(
        0.5,
        0.98,
        "Single well ($\\gamma > \\gamma^*$) → Double well ($\\gamma < \\gamma^*$)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        fontsize=10,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Potential slices plot saved to {save_path}")

    return fig


def plot_potential_derivative(
    gamma_values: List[float] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot dV/dp (gradient) for multiple gamma values.

    Args:
        gamma_values: List of gamma values
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution

    Returns:
        matplotlib Figure object

    Output:
        Shows how gradient dV/dp = BR(p) - p changes with gamma
        Zero crossings indicate equilibria
    """
    if gamma_values is None:
        gamma_values = [0.60, 0.65, GAMMA_STAR, 0.75, 0.80]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    p_grid = np.linspace(0.05, 0.95, 150)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(gamma_values)))

    from src.analysis.potential import potential_derivative

    for i, gamma in enumerate(gamma_values):
        dV = np.array([potential_derivative(p, gamma) for p in p_grid])

        label = f"$\\gamma={gamma:.3f}$"
        if abs(gamma - GAMMA_STAR) < 1e-6:
            label += " ($\\gamma^*$)"
            linewidth = 3
        else:
            linewidth = 2

        ax.plot(p_grid, dV, label=label, color=colors[i], linewidth=linewidth)

    # Zero line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel("Market Share $p$", fontsize=12)
    ax.set_ylabel("Gradient $dV/dp = BR(p) - p$", fontsize=12)
    ax.set_title("Potential Gradient (Equilibria at Zero Crossings)", fontsize=14, pad=15)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Potential derivative plot saved to {save_path}")

    return fig
