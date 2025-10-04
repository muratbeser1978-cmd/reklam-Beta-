"""
Bifurcation Diagram Visualization.

Plots equilibrium positions as function of gamma parameter.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from src.analysis.bifurcation import BifurcationAnalyzer, BifurcationData
from src.distributions.constants import GAMMA_STAR


def plot_bifurcation(
    bifurcation_data: Optional[BifurcationData] = None,
    gamma_min: float = 0.5,
    gamma_max: float = 0.9,
    resolution: int = 100,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 7),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot bifurcation diagram showing equilibria as function of gamma.

    Args:
        bifurcation_data: Precomputed bifurcation data. If None, compute fresh
        gamma_min: Minimum gamma value
        gamma_max: Maximum gamma value
        resolution: Number of gamma points
        save_path: Path to save figure (PDF format)
        figsize: Figure size in inches
        dpi: Resolution

    Returns:
        matplotlib Figure object

    Output:
        Bifurcation diagram showing:
        - Symmetric equilibrium (p*=0.5) always present
        - Asymmetric equilibria (p* ≠ 0.5) below γ*
        - Solid lines: stable equilibria
        - Dashed lines: unstable equilibria
        - Vertical line at γ* = 12/17
    """
    # Generate data if not provided
    if bifurcation_data is None:
        analyzer = BifurcationAnalyzer()
        bifurcation_data = analyzer.compute_bifurcation_diagram(
            gamma_min=gamma_min, gamma_max=gamma_max, resolution=resolution
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    gamma_grid = bifurcation_data.gamma_grid

    # Plot symmetric equilibrium
    stable_sym = bifurcation_data.stable_mask_symmetric
    unstable_sym = ~stable_sym
    ax.plot(
        gamma_grid[stable_sym],
        bifurcation_data.p_symmetric[stable_sym],
        color="#2E86AB",
        linewidth=2.5,
        label="Stable Symmetric",
        linestyle="-",
    )
    ax.plot(
        gamma_grid[unstable_sym],
        bifurcation_data.p_symmetric[unstable_sym],
        color="#2E86AB",
        linewidth=2.5,
        label="Unstable Symmetric",
        linestyle=":",
    )

    # Plot asymmetric equilibria
    stable_asym = bifurcation_data.stable_mask_asymmetric
    unstable_asym = ~stable_asym

    # High branch
    p_high = bifurcation_data.p_asymmetric_high
    valid_high = ~np.isnan(p_high)
    ax.plot(
        gamma_grid[valid_high & stable_asym],
        p_high[valid_high & stable_asym],
        color="#A23B72",
        linewidth=2.5,
        label="Stable Asymmetric",
        linestyle="-",
    )
    ax.plot(
        gamma_grid[valid_high & unstable_asym],
        p_high[valid_high & unstable_asym],
        color="#A23B72",
        linewidth=2.5,
        label="Unstable Asymmetric",
        linestyle=":",
    )

    # Low branch
    p_low = bifurcation_data.p_asymmetric_low
    valid_low = ~np.isnan(p_low)
    ax.plot(
        gamma_grid[valid_low & stable_asym],
        p_low[valid_low & stable_asym],
        color="#F18F01",
        linewidth=2.5,
        linestyle="-",
    )
    ax.plot(
        gamma_grid[valid_low & unstable_asym],
        p_low[valid_low & unstable_asym],
        color="#F18F01",
        linewidth=2.5,
        linestyle=":",
    )

    # Mark critical point γ*
    ax.axvline(
        x=GAMMA_STAR,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"$\\gamma^* = {GAMMA_STAR:.6f}$",
    )

    # Add shaded regions
    ax.axvspan(gamma_min, GAMMA_STAR, alpha=0.1, color="red", label="Multiple equilibria")
    ax.axvspan(GAMMA_STAR, gamma_max, alpha=0.1, color="green", label="Unique equilibrium")

    # Styling
    ax.set_xlabel("Learning Weight $\\gamma$", fontsize=14)
    ax.set_ylabel("Equilibrium Market Share $p^*$", fontsize=14)
    ax.set_title("Bifurcation Diagram: Supercritical Pitchfork", fontsize=16, pad=20)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(gamma_min, gamma_max)
    ax.set_ylim(0, 1)

    # Add annotations
    if np.any(valid_high):
        # Annotate bifurcation point
        ax.annotate(
            "Bifurcation point",
            xy=(GAMMA_STAR, 0.5),
            xytext=(GAMMA_STAR - 0.1, 0.7),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Bifurcation diagram saved to {save_path}")

    return fig


def plot_bifurcation_with_amplitude(
    bifurcation_data: Optional[BifurcationData] = None,
    gamma_min: float = 0.5,
    gamma_max: float = 0.9,
    resolution: int = 100,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot bifurcation diagram with amplitude subplot showing δ(γ) ~ √(γ* - γ).

    Args:
        bifurcation_data: Precomputed bifurcation data
        gamma_min: Minimum gamma
        gamma_max: Maximum gamma
        resolution: Number of points
        save_path: Save path
        figsize: Figure size
        dpi: Resolution

    Returns:
        Figure with two subplots: bifurcation diagram and amplitude
    """
    if bifurcation_data is None:
        analyzer = BifurcationAnalyzer()
        bifurcation_data = analyzer.compute_bifurcation_diagram(
            gamma_min=gamma_min, gamma_max=gamma_max, resolution=resolution
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    gamma_grid = bifurcation_data.gamma_grid

    # Left: Bifurcation diagram
    ax1.plot(
        gamma_grid,
        bifurcation_data.p_symmetric,
        color="#2E86AB",
        linewidth=2,
        label="Symmetric",
    )

    high_mask = bifurcation_data.p_asymmetric_high > 0
    if np.any(high_mask):
        ax1.plot(
            gamma_grid[high_mask],
            bifurcation_data.p_asymmetric_high[high_mask],
            color="#A23B72",
            linewidth=2,
            label="Asymmetric",
        )

    low_mask = bifurcation_data.p_asymmetric_low > 0
    if np.any(low_mask):
        ax1.plot(
            gamma_grid[low_mask],
            bifurcation_data.p_asymmetric_low[low_mask],
            color="#A23B72",
            linewidth=2,
        )

    ax1.axvline(x=GAMMA_STAR, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax1.set_xlabel("$\\gamma$", fontsize=12)
    ax1.set_ylabel("$p^*$", fontsize=12)
    ax1.set_title("Bifurcation Diagram", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Right: Amplitude δ(γ) = |p* - 0.5|
    below_gamma_star = gamma_grid < GAMMA_STAR
    if np.any(below_gamma_star):
        gamma_below = gamma_grid[below_gamma_star]
        high_below = bifurcation_data.p_asymmetric_high[below_gamma_star]

        # Compute amplitude
        amplitude = np.abs(high_below - 0.5)

        # Theoretical scaling: δ ~ √(γ* - γ)
        theoretical_amplitude = 0.2 * np.sqrt(GAMMA_STAR - gamma_below)

        ax2.plot(gamma_below, amplitude, "o", color="#A23B72", label="Observed $\\delta(\\gamma)$")
        ax2.plot(
            gamma_below,
            theoretical_amplitude,
            "--",
            color="black",
            linewidth=2,
            label="$\\delta \\propto \\sqrt{\\gamma^* - \\gamma}$",
        )

        ax2.set_xlabel("$\\gamma$ (below $\\gamma^*$)", fontsize=12)
        ax2.set_ylabel("Amplitude $\\delta = |p^* - 0.5|$", fontsize=12)
        ax2.set_title("Bifurcation Amplitude Scaling", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", dpi=dpi, bbox_inches="tight")
        print(f"Bifurcation with amplitude plot saved to {save_path}")

    return fig
