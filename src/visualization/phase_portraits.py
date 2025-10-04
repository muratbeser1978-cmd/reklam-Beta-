"""
Phase Portrait Visualization.

Best response curves, potential landscapes, equilibrium stability analysis.

Usage:
    from src.visualization.phase_portraits import PhasePortraitPlotter

    plotter = PhasePortraitPlotter()
    fig = plotter.plot_best_response(gamma=0.7)
    fig.savefig('phase_portrait.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from src.analysis.best_response import best_response as BR, br_derivative
from src.analysis.potential import potential, potential_derivative, potential_second_derivative
from src.visualization.utils import find_equilibria


class PhasePortraitPlotter:
    """
    Visualize phase space dynamics: BR curves, potential landscapes, stability.

    Methods:
        plot_best_response: BR(p,γ) curve with equilibria
        plot_potential_landscape: V(p,γ) potential function
        plot_phase_space: Combined BR + vector field
        plot_stability_analysis: Eigenvalues and basins of attraction
    """

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', figsize: Tuple[int, int] = (10, 6)):
        """Initialize plotter."""
        self.style = style
        self.figsize = figsize
        self.gamma_star = 12.0 / 17.0

    def plot_best_response(
        self,
        gamma: float,
        show_equilibria: bool = True,
        show_stability: bool = True,
        show_45_line: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot best response curve BR(p,γ).

        Args:
            gamma: Individual taste weight
            show_equilibria: Mark equilibrium points
            show_stability: Color-code stability
            show_45_line: Show 45° line
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> plotter = PhasePortraitPlotter()
            >>> fig = plotter.plot_best_response(gamma=0.65)
            >>> fig.savefig('br_curve.png', dpi=300)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Compute BR curve
        p_grid = np.linspace(0, 1, 500)
        br_values = np.array([BR(p, gamma) for p in p_grid])

        # Plot BR curve
        ax.plot(p_grid, br_values, linewidth=2.5, label='$BR(p, \\gamma)$',
               color='#2E86AB')

        # 45° line
        if show_45_line:
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6,
                   label='$BR(p) = p$ (equilibrium condition)')

        # Find and mark equilibria
        if show_equilibria:
            equilibria = find_equilibria(gamma)

            for eq in equilibria:
                # Compute stability (BR'(p*))
                br_prime = br_derivative(eq, gamma)
                stable = br_prime < 1.0

                # Color-code by stability
                if show_stability:
                    color = 'green' if stable else 'red'
                    marker = 'o' if stable else 'x'
                    size = 150 if stable else 200
                    label_suffix = ' (stable)' if stable else ' (unstable)'
                else:
                    color = 'red'
                    marker = 'o'
                    size = 150
                    label_suffix = ''

                ax.scatter(eq, eq, s=size, color=color, marker=marker,
                          edgecolors='black', linewidths=2, zorder=5,
                          label=f'$p^* = {eq:.3f}${label_suffix}')

        # Styling
        ax.set_xlabel('Market Share $p_1$', fontsize=13)
        ax.set_ylabel('Best Response $BR(p_1, \\gamma)$', fontsize=13)

        regime = "Below $\\gamma^*$" if gamma < self.gamma_star else "Above $\\gamma^*$"
        ax.set_title(f'Best Response Function ($\\gamma = {gamma:.3f}$, {regime})',
                    fontsize=15, fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95, fontsize=10)

        # Add annotation about critical threshold
        ax.text(0.05, 0.95, f'$\\gamma^* = {self.gamma_star:.4f}$',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top')

        fig.tight_layout()
        return fig

    def plot_potential_landscape(
        self,
        gamma: float,
        show_equilibria: bool = True,
        show_derivative: bool = False,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot potential function V(p,γ).

        Args:
            gamma: Individual taste weight
            show_equilibria: Mark equilibria
            show_derivative: Show dV/dp in subplot
            ax: Existing axes

        Returns:
            Figure

        Theory:
            V(p) = ∫[p - BR(p)]dp
            Equilibria: dV/dp = 0
            Stability: d²V/dp² > 0 (stable), < 0 (unstable)

        Example:
            >>> fig = plotter.plot_potential_landscape(gamma=0.60)
        """
        if ax is None:
            with plt.style.context(self.style):
                if show_derivative:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                else:
                    fig, ax1 = plt.subplots(figsize=self.figsize)
                    ax2 = None
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = None

        # Compute potential landscape
        p_grid = np.linspace(0, 1, 500)
        V_values = np.array([potential(p, gamma) for p in p_grid])

        # Plot V(p)
        ax1.plot(p_grid, V_values, linewidth=2.5, color='#6A4C93', label='$V(p)$')

        # Mark equilibria
        if show_equilibria:
            equilibria = find_equilibria(gamma)

            for eq in equilibria:
                V_eq = potential(eq, gamma)
                d2V = potential_second_derivative(eq, gamma)

                stable = d2V > 0  # Local minimum
                color = 'green' if stable else 'red'
                marker = 'o' if stable else 'x'

                ax1.scatter(eq, V_eq, s=200, color=color, marker=marker,
                          edgecolors='black', linewidths=2, zorder=5)

                # Annotate
                label = 'min' if stable else 'max'
                ax1.annotate(f'$p^* = {eq:.3f}$ ({label})',
                           xy=(eq, V_eq), xytext=(eq + 0.1, V_eq + 0.02),
                           fontsize=10, arrowprops=dict(arrowstyle='->', lw=1.5),
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

        ax1.set_ylabel('Potential $V(p, \\gamma)$', fontsize=13)
        ax1.set_title(f'Potential Landscape ($\\gamma = {gamma:.3f}$)',
                     fontsize=15, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Plot derivative (force field)
        if show_derivative and ax2 is not None:
            dV_values = np.array([potential_derivative(p, gamma) for p in p_grid])

            ax2.plot(p_grid, dV_values, linewidth=2.5, color='#F77F00', label='$dV/dp$')
            ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

            # Mark zeros (equilibria)
            if show_equilibria:
                for eq in equilibria:
                    ax2.axvline(eq, color='gray', linestyle=':', alpha=0.7)

            ax2.set_xlabel('Market Share $p_1$', fontsize=13)
            ax2.set_ylabel('Force $dV/dp$', fontsize=13)
            ax2.set_title('Potential Derivative (Restoring Force)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')
        else:
            ax1.set_xlabel('Market Share $p_1$', fontsize=13)

        fig.tight_layout()
        return fig

    def plot_phase_space(
        self,
        gamma: float,
        n_arrows: int = 20,
        trajectory: Optional[np.ndarray] = None,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot phase space with vector field and optional trajectory.

        Args:
            gamma: Individual taste weight
            n_arrows: Number of arrows in vector field
            trajectory: Optional p₁(t) trajectory to overlay
            ax: Existing axes

        Returns:
            Figure

        Dynamics:
            ṗ = BR(p) - p  (continuous time approximation)

        Example:
            >>> fig = plotter.plot_phase_space(gamma=0.65, trajectory=result.p1_trajectory)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # BR curve
        p_grid = np.linspace(0, 1, 300)
        br_values = np.array([BR(p, gamma) for p in p_grid])
        ax.plot(p_grid, br_values, linewidth=2.5, label='$BR(p)$', color='#2E86AB')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='$p_{t+1} = p_t$')

        # Vector field: arrows showing dynamics
        p_arrows = np.linspace(0.05, 0.95, n_arrows)
        for p in p_arrows:
            br_p = BR(p, gamma)
            dp = br_p - p  # Change in p

            # Arrow from (p, p) to (p, BR(p))
            ax.arrow(p, p, 0, dp * 0.8, head_width=0.02, head_length=0.02,
                    fc='gray', ec='gray', alpha=0.5, zorder=1)

        # Equilibria
        equilibria = find_equilibria(gamma)
        for eq in equilibria:
            br_prime = br_derivative(eq, gamma)
            stable = br_prime < 1.0
            color = 'green' if stable else 'red'
            marker = 'o' if stable else 'x'

            ax.scatter(eq, eq, s=200, color=color, marker=marker,
                      edgecolors='black', linewidths=2, zorder=5)

        # Overlay trajectory
        if trajectory is not None:
            # Create staircase: (p_t, p_t) -> (p_t, p_{t+1}) -> (p_{t+1}, p_{t+1})
            p_current = trajectory[:-1]
            p_next = trajectory[1:]

            ax.plot(p_current, p_next, 'o-', linewidth=1.5, markersize=4,
                   color='#A23B72', alpha=0.7, label='Trajectory', zorder=3)

            # Starting point
            ax.scatter(trajectory[0], trajectory[0], s=150, color='lime',
                      marker='s', edgecolors='black', linewidths=2,
                      label='Start', zorder=6)

            # Ending point
            ax.scatter(trajectory[-1], trajectory[-1], s=150, color='orange',
                      marker='*', edgecolors='black', linewidths=2,
                      label='End', zorder=6)

        ax.set_xlabel('Current State $p_t$', fontsize=13)
        ax.set_ylabel('Next State $p_{t+1} = BR(p_t)$', fontsize=13)
        ax.set_title(f'Phase Portrait ($\\gamma = {gamma:.3f}$)',
                    fontsize=15, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95)

        fig.tight_layout()
        return fig

    def plot_stability_analysis(
        self,
        gamma_values: np.ndarray,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot stability analysis: BR'(p*) vs γ.

        Args:
            gamma_values: Array of gamma values to analyze
            ax: Existing axes

        Returns:
            Figure

        Theory:
            Stable equilibrium: |BR'(p*)| < 1
            Bifurcation at γ*: BR'(0.5, γ*) = 1

        Example:
            >>> gammas = np.linspace(0.5, 0.9, 50)
            >>> fig = plotter.plot_stability_analysis(gammas)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # For each gamma, find equilibria and compute BR'
        stable_gammas = []
        stable_equilibria = []
        stable_derivatives = []

        unstable_gammas = []
        unstable_equilibria = []
        unstable_derivatives = []

        for gamma in gamma_values:
            equilibria = find_equilibria(gamma)

            for eq in equilibria:
                br_prime = br_derivative(eq, gamma)

                if abs(br_prime) < 1.0:
                    stable_gammas.append(gamma)
                    stable_equilibria.append(eq)
                    stable_derivatives.append(br_prime)
                else:
                    unstable_gammas.append(gamma)
                    unstable_equilibria.append(eq)
                    unstable_derivatives.append(br_prime)

        # Plot BR' vs gamma
        if stable_gammas:
            ax.scatter(stable_gammas, stable_derivatives, c='green', s=50,
                      alpha=0.7, label='Stable equilibria', marker='o')

        if unstable_gammas:
            ax.scatter(unstable_gammas, unstable_derivatives, c='red', s=50,
                      alpha=0.7, label='Unstable equilibria', marker='x')

        # Stability boundaries
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                  label='Stability boundary')
        ax.axhline(-1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

        # Critical threshold
        ax.axvline(self.gamma_star, color='purple', linestyle=':', linewidth=2,
                  alpha=0.7, label=f'$\\gamma^* = {self.gamma_star:.4f}$')

        ax.set_xlabel('Individual Taste Weight $\\gamma$', fontsize=13)
        ax.set_ylabel('Derivative $BR\'(p^*, \\gamma)$', fontsize=13)
        ax.set_title('Stability Analysis: BR\' at Equilibria', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95)

        # Shade stability regions
        ax.axhspan(-1, 1, alpha=0.1, color='green', zorder=0)
        ax.axhspan(1, ax.get_ylim()[1], alpha=0.1, color='red', zorder=0)
        ax.axhspan(ax.get_ylim()[0], -1, alpha=0.1, color='red', zorder=0)

        fig.tight_layout()
        return fig

    def plot_bifurcation_comparison(
        self,
        gamma_below: float = 0.65,
        gamma_above: float = 0.75,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Side-by-side comparison of BR curves below and above critical threshold.

        Args:
            gamma_below: γ < γ*
            gamma_above: γ > γ*
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> fig = plotter.plot_bifurcation_comparison()
        """
        with plt.style.context(self.style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Below critical
        self.plot_best_response(gamma_below, ax=ax1)
        ax1.set_title(f'Below Critical: $\\gamma = {gamma_below:.3f} < \\gamma^*$',
                     fontsize=14, fontweight='bold')

        # Above critical
        self.plot_best_response(gamma_above, ax=ax2)
        ax2.set_title(f'Above Critical: $\\gamma = {gamma_above:.3f} > \\gamma^*$',
                     fontsize=14, fontweight='bold')

        fig.suptitle('Bifurcation: Equilibrium Structure Changes at $\\gamma^*$',
                    fontsize=16, fontweight='bold', y=1.02)

        fig.tight_layout()
        return fig
