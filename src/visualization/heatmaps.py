"""
Heatmap Visualization.

2D parameter space exploration heatmaps for γ, β, variance, welfare, etc.

Usage:
    from src.visualization.heatmaps import HeatmapPlotter

    plotter = HeatmapPlotter()
    fig = plotter.plot_parameter_heatmap(gamma_grid, beta_grid, welfare_grid)
    fig.savefig('heatmap.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class HeatmapPlotter:
    """
    Visualize 2D parameter spaces with heatmaps.

    Methods:
        plot_parameter_heatmap: Generic 2D parameter heatmap
        plot_welfare_heatmap: Welfare loss vs (γ, β)
        plot_variance_heatmap: Variance vs parameters
        plot_bifurcation_region: Mark bifurcation boundaries
    """

    def __init__(self, style: str = 'seaborn-v0_8-white', figsize: Tuple[int, int] = (10, 8)):
        """Initialize plotter."""
        self.style = style
        self.figsize = figsize
        self.gamma_star = 12.0 / 17.0

    def plot_parameter_heatmap(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        z_grid: np.ndarray,
        xlabel: str = 'Parameter 1',
        ylabel: str = 'Parameter 2',
        zlabel: str = 'Metric Value',
        title: str = 'Parameter Space Heatmap',
        cmap: str = 'viridis',
        show_contours: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot generic 2D parameter space heatmap.

        Args:
            x_values: X-axis parameter values (n_x,)
            y_values: Y-axis parameter values (n_y,)
            z_grid: Metric values (n_y, n_x) grid
            xlabel, ylabel, zlabel: Axis labels
            title: Plot title
            cmap: Colormap name
            show_contours: Overlay contour lines
            vmin, vmax: Color scale limits
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> gammas = np.linspace(0.5, 0.9, 50)
            >>> betas = np.linspace(0.7, 0.95, 40)
            >>> welfare = compute_welfare_grid(gammas, betas)
            >>> fig = plotter.plot_parameter_heatmap(gammas, betas, welfare)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Create meshgrid
        X, Y = np.meshgrid(x_values, y_values)

        # Plot heatmap
        im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, shading='auto',
                          vmin=vmin, vmax=vmax)

        # Contour lines
        if show_contours:
            contours = ax.contour(X, Y, z_grid, colors='white', alpha=0.4,
                                 linewidths=1, levels=10)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(zlabel, fontsize=12)

        # Labels
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')

        fig.tight_layout()
        return fig

    def plot_welfare_heatmap(
        self,
        gamma_values: np.ndarray,
        beta_values: np.ndarray,
        welfare_loss_grid: np.ndarray,
        mark_critical: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot welfare loss heatmap: ΔW vs (γ, β).

        Args:
            gamma_values: γ values (n_gamma,)
            beta_values: β values (n_beta,)
            welfare_loss_grid: ΔW values (n_beta, n_gamma)
            mark_critical: Mark γ* line
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> gammas = np.linspace(0.5, 0.9, 40)
            >>> betas = np.linspace(0.7, 0.95, 30)
            >>> welfare_loss = compute_welfare_loss_grid(gammas, betas)
            >>> fig = plotter.plot_welfare_heatmap(gammas, betas, welfare_loss)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Heatmap
        X, Y = np.meshgrid(gamma_values, beta_values)
        im = ax.pcolormesh(X, Y, welfare_loss_grid, cmap='YlOrRd', shading='auto')

        # Contours
        contours = ax.contour(X, Y, welfare_loss_grid, colors='black', alpha=0.3,
                             linewidths=1.5, levels=8)
        ax.clabel(contours, inline=True, fontsize=9, fmt='%.3f')

        # Critical line
        if mark_critical:
            ax.axvline(self.gamma_star, color='blue', linestyle='--', linewidth=2.5,
                      alpha=0.8, label=f'$\\gamma^* = {self.gamma_star:.4f}$')
            ax.legend(loc='best', fontsize=11, framealpha=0.95)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Welfare Loss $\\Delta W$', fontsize=12)

        # Labels
        ax.set_xlabel('Individual Taste Weight $\\gamma$', fontsize=13)
        ax.set_ylabel('Retention Rate $\\beta$', fontsize=13)
        ax.set_title('Welfare Loss Heatmap: $\\Delta W(\\gamma, \\beta)$',
                    fontsize=15, fontweight='bold')

        fig.tight_layout()
        return fig

    def plot_variance_heatmap(
        self,
        gamma_values: np.ndarray,
        beta_values: np.ndarray,
        variance_grid: np.ndarray,
        log_scale: bool = True,
        mark_critical: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot variance heatmap: Var[p_∞] vs (γ, β).

        Args:
            gamma_values: γ values
            beta_values: β values
            variance_grid: Variance values
            log_scale: Use log scale for variance
            mark_critical: Mark γ*
            ax: Existing axes

        Returns:
            Figure

        Theory:
            Variance diverges near γ*, especially for high β (slow learning).

        Example:
            >>> fig = plotter.plot_variance_heatmap(gammas, betas, var_grid)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        X, Y = np.meshgrid(gamma_values, beta_values)

        # Apply log scale if requested
        if log_scale:
            z_plot = np.log10(variance_grid + 1e-10)  # Avoid log(0)
            cmap = 'plasma'
            cbar_label = 'Log₁₀ Variance'
        else:
            z_plot = variance_grid
            cmap = 'plasma'
            cbar_label = 'Variance $\\mathrm{Var}[p_\\infty]$'

        # Heatmap
        im = ax.pcolormesh(X, Y, z_plot, cmap=cmap, shading='auto')

        # Contours
        contours = ax.contour(X, Y, z_plot, colors='white', alpha=0.5,
                             linewidths=1, levels=10)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

        # Critical line
        if mark_critical:
            ax.axvline(self.gamma_star, color='cyan', linestyle='--', linewidth=2.5,
                      alpha=0.9, label=f'$\\gamma^* = {self.gamma_star:.4f}$')
            ax.legend(loc='best', fontsize=11, framealpha=0.95)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=12)

        # Labels
        ax.set_xlabel('Individual Taste Weight $\\gamma$', fontsize=13)
        ax.set_ylabel('Retention Rate $\\beta$', fontsize=13)

        title = 'Variance Heatmap (Log Scale)' if log_scale else 'Variance Heatmap'
        ax.set_title(title, fontsize=15, fontweight='bold')

        fig.tight_layout()
        return fig

    def plot_bifurcation_region(
        self,
        gamma_values: np.ndarray,
        beta_values: np.ndarray,
        n_equilibria_grid: np.ndarray,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot bifurcation region: number of equilibria vs (γ, β).

        Args:
            gamma_values: γ values
            beta_values: β values
            n_equilibria_grid: Number of equilibria at each (γ, β)
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> n_eq = compute_n_equilibria_grid(gammas, betas)
            >>> fig = plotter.plot_bifurcation_region(gammas, betas, n_eq)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        X, Y = np.meshgrid(gamma_values, beta_values)

        # Discrete colormap for number of equilibria
        cmap = plt.cm.get_cmap('Set3', int(n_equilibria_grid.max()) + 1)

        im = ax.pcolormesh(X, Y, n_equilibria_grid, cmap=cmap, shading='auto',
                          vmin=0, vmax=n_equilibria_grid.max())

        # Critical line
        ax.axvline(self.gamma_star, color='red', linestyle='--', linewidth=3,
                  alpha=0.8, label=f'$\\gamma^* = {self.gamma_star:.4f}$')

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.02, ticks=np.arange(0, n_equilibria_grid.max() + 1))
        cbar.set_label('Number of Equilibria', fontsize=12)

        # Labels
        ax.set_xlabel('Individual Taste Weight $\\gamma$', fontsize=13)
        ax.set_ylabel('Retention Rate $\\beta$', fontsize=13)
        ax.set_title('Bifurcation Region Map', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.95)

        # Annotations
        ax.text(0.6, 0.85, 'One equilibrium\n($\\gamma > \\gamma^*$)',
               fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.text(0.6, 0.75, 'Multiple equilibria\n($\\gamma < \\gamma^*$)',
               fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

        fig.tight_layout()
        return fig

    def plot_correlation_heatmap(
        self,
        variables: List[str],
        correlation_matrix: np.ndarray,
        annotate: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot correlation heatmap between simulation outcomes.

        Args:
            variables: List of variable names
            correlation_matrix: (n_vars, n_vars) correlation matrix
            annotate: Show correlation values
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> vars = ['Welfare Loss', 'Variance', 'Lock-in Fraction', 'Regret']
            >>> corr_matrix = np.corrcoef(data.T)
            >>> fig = plotter.plot_correlation_heatmap(vars, corr_matrix)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=(8, 7))
        else:
            fig = ax.figure

        # Heatmap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                      aspect='auto', interpolation='nearest')

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Correlation Coefficient', fontsize=12)

        # Ticks
        ax.set_xticks(np.arange(len(variables)))
        ax.set_yticks(np.arange(len(variables)))
        ax.set_xticklabels(variables, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(variables, fontsize=10)

        # Annotate values
        if annotate:
            for i in range(len(variables)):
                for j in range(len(variables)):
                    value = correlation_matrix[i, j]
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=text_color, fontsize=9, fontweight='bold')

        # Title
        ax.set_title('Correlation Matrix: Simulation Outcomes',
                    fontsize=15, fontweight='bold')

        fig.tight_layout()
        return fig
