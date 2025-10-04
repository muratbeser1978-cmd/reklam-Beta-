"""
Bifurcation Diagram Visualization.

Visualize how equilibrium structure changes with parameter γ across the critical threshold γ*.

Usage:
    from src.visualization.bifurcation_diagrams import BifurcationPlotter

    plotter = BifurcationPlotter()
    fig = plotter.plot_bifurcation_diagram(gamma_values, equilibria_data)
    fig.savefig('bifurcation.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from src.analysis.best_response import br_derivative, best_response as BR
from src.visualization.utils import find_equilibria


class BifurcationPlotter:
    """
    Visualize bifurcation phenomena and parameter sweeps.

    Methods:
        plot_bifurcation_diagram: Classic bifurcation diagram (γ vs p*)
        plot_variance_scaling: Critical scaling Var ~ (γ-γ*)^(-ν)
        plot_parameter_sweep: Multiple metrics vs γ
        plot_hysteresis_loop: Hysteresis in dynamic adjustment
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (10, 6)):
        """Initialize plotter."""
        self.style = style
        self.figsize = figsize
        self.gamma_star = 12.0 / 17.0

    def plot_bifurcation_diagram(
        self,
        gamma_values: Optional[np.ndarray] = None,
        equilibria_data: Optional[Dict[float, List[float]]] = None,
        show_stability: bool = True,
        show_critical_line: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot classic bifurcation diagram: γ vs p*.

        Args:
            gamma_values: Array of γ values (if None, compute automatically)
            equilibria_data: Dict mapping γ -> list of equilibria (if None, compute)
            show_stability: Color-code stable/unstable branches
            show_critical_line: Mark critical threshold γ*
            ax: Existing axes

        Returns:
            Figure

        Theory:
            - For γ > γ*: Single stable equilibrium at p* = 0.5
            - For γ < γ*: Pitchfork bifurcation → two stable equilibria at p* ≠ 0.5
            - At γ = γ*: BR'(0.5, γ*) = 1 (marginal stability)

        Example:
            >>> plotter = BifurcationPlotter()
            >>> fig = plotter.plot_bifurcation_diagram()
            >>> fig.savefig('bifurcation.png', dpi=300)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Generate data if not provided
        if gamma_values is None:
            gamma_values = np.linspace(0.5, 0.95, 200)

        if equilibria_data is None:
            equilibria_data = {}
            for gamma in gamma_values:
                equilibria_data[gamma] = find_equilibria(gamma)

        # Separate stable and unstable equilibria
        stable_gammas, stable_equilibria = [], []
        unstable_gammas, unstable_equilibria = [], []

        for gamma in gamma_values:
            if gamma not in equilibria_data:
                continue

            equilibria = equilibria_data[gamma]

            for eq in equilibria:
                if show_stability:
                    br_prime = br_derivative(eq, gamma)
                    is_stable = abs(br_prime) < 1.0

                    if is_stable:
                        stable_gammas.append(gamma)
                        stable_equilibria.append(eq)
                    else:
                        unstable_gammas.append(gamma)
                        unstable_equilibria.append(eq)
                else:
                    stable_gammas.append(gamma)
                    stable_equilibria.append(eq)

        # Plot equilibria branches
        if show_stability:
            if stable_gammas:
                ax.scatter(stable_gammas, stable_equilibria, c='#2E86AB', s=20,
                          alpha=0.8, label='Stable equilibria', marker='o', linewidths=0)
            if unstable_gammas:
                ax.scatter(unstable_gammas, unstable_equilibria, c='red', s=20,
                          alpha=0.6, label='Unstable equilibria', marker='x', linewidths=0.5)
        else:
            ax.scatter(stable_gammas, stable_equilibria, c='#2E86AB', s=20,
                      alpha=0.8, label='Equilibria', marker='o', linewidths=0)

        # Critical threshold
        if show_critical_line:
            ax.axvline(self.gamma_star, color='purple', linestyle='--', linewidth=2.5,
                      alpha=0.8, label=f'$\\gamma^* = {self.gamma_star:.4f}$', zorder=10)

        # Reference line p* = 0.5
        ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.5,
                  label='$p^* = 0.5$ (symmetric)')

        # Annotations
        ax.annotate('Supercritical\nPitchfork\nBifurcation',
                   xy=(self.gamma_star, 0.5), xytext=(self.gamma_star - 0.08, 0.75),
                   fontsize=11, color='purple', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                   bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

        ax.text(0.52, 0.15, 'Asymmetric\nEquilibria\n($\\gamma < \\gamma^*$)',
               fontsize=10, ha='left', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        ax.text(0.82, 0.55, 'Symmetric\nEquilibrium\n($\\gamma > \\gamma^*$)',
               fontsize=10, ha='left', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Styling
        ax.set_xlabel('Individual Taste Weight $\\gamma$', fontsize=13)
        ax.set_ylabel('Equilibrium Market Share $p^*$', fontsize=13)
        ax.set_title('Bifurcation Diagram: Equilibrium Structure vs $\\gamma$',
                    fontsize=15, fontweight='bold')
        ax.set_xlim(gamma_values.min(), gamma_values.max())
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95, fontsize=10)

        fig.tight_layout()
        return fig

    def plot_variance_scaling(
        self,
        gamma_values: np.ndarray,
        variances: np.ndarray,
        theoretical_exponent: float = 1.0,
        show_fit: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot critical scaling: Var[p] ~ (γ - γ*)^(-ν).

        Args:
            gamma_values: Array of γ values
            variances: Corresponding variance values
            theoretical_exponent: Theoretical ν (default: 1.0 mean-field)
            show_fit: Fit power law and show
            ax: Existing axes

        Returns:
            Figure

        Theory:
            Near critical point γ*, variance diverges:
            Var[p_∞] ~ |γ - γ*|^(-ν)

            Mean-field prediction: ν = 1

        Example:
            >>> gammas = np.linspace(0.71, 0.90, 30)
            >>> variances = [simulate(gamma).variance for gamma in gammas]
            >>> fig = plotter.plot_variance_scaling(gammas, variances)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Filter to γ > γ* (above critical)
        mask = gamma_values > self.gamma_star
        gamma_above = gamma_values[mask]
        var_above = variances[mask]

        if len(gamma_above) < 3:
            raise ValueError("Not enough data points above critical threshold")

        # Distance from critical point
        distance = gamma_above - self.gamma_star

        # Log-log plot
        log_distance = np.log(distance)
        log_var = np.log(var_above)

        # Plot data
        ax.scatter(distance, var_above, s=80, color='#2E86AB', alpha=0.7,
                  edgecolors='black', linewidths=1, label='Simulation data', zorder=5)

        # Power law fit
        if show_fit:
            # Linear fit in log-log space
            coeffs = np.polyfit(log_distance, log_var, 1)
            fitted_exponent = -coeffs[0]  # ν = -slope

            # Generate fitted curve
            dist_fit = np.linspace(distance.min(), distance.max(), 100)
            var_fit = np.exp(coeffs[1]) * dist_fit ** coeffs[0]

            ax.plot(dist_fit, var_fit, linewidth=2.5, color='red', linestyle='--',
                   label=f'Fitted: $\\nu = {fitted_exponent:.2f}$', zorder=4)

            # Theoretical prediction
            if theoretical_exponent is not None:
                # Normalize to match data at some point
                idx_mid = len(dist_fit) // 2
                scale = var_fit[idx_mid] / (dist_fit[idx_mid] ** (-theoretical_exponent))
                var_theory = scale * dist_fit ** (-theoretical_exponent)

                ax.plot(dist_fit, var_theory, linewidth=2.5, color='green', linestyle='-.',
                       alpha=0.7, label=f'Theory: $\\nu = {theoretical_exponent:.1f}$ (mean-field)',
                       zorder=3)

        # Log-log axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('Distance from Critical Point $|\\gamma - \\gamma^*|$', fontsize=13)
        ax.set_ylabel('Variance $\\mathrm{Var}[p_\\infty]$', fontsize=13)
        ax.set_title('Critical Scaling: Variance Divergence near Bifurcation',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', framealpha=0.95, fontsize=11)

        # Add annotation
        ax.text(0.05, 0.95, f'$\\gamma^* = {self.gamma_star:.4f}$\\n'
                            f'$\\mathrm{{Var}} \\sim |\\gamma - \\gamma^*|^{{-\\nu}}$',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top')

        fig.tight_layout()
        return fig

    def plot_parameter_sweep(
        self,
        gamma_values: np.ndarray,
        metrics_dict: Dict[str, np.ndarray],
        ylabel: str = 'Metric Value',
        title: str = 'Parameter Sweep: Metrics vs $\\gamma$',
        show_critical: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot multiple metrics vs γ.

        Args:
            gamma_values: Array of γ values
            metrics_dict: Dict mapping metric name -> values array
            ylabel: Y-axis label
            title: Plot title
            show_critical: Mark γ*
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> metrics = {
            ...     'Welfare Loss': welfare_losses,
            ...     'Variance': variances,
            ...     'Lock-in Fraction': lockin_fractions
            ... }
            >>> fig = plotter.plot_parameter_sweep(gammas, metrics)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))

        # Plot each metric
        for (metric_name, values), color in zip(metrics_dict.items(), colors):
            ax.plot(gamma_values, values, linewidth=2.5, marker='o', markersize=5,
                   label=metric_name, color=color, alpha=0.8)

        # Critical threshold
        if show_critical:
            ax.axvline(self.gamma_star, color='purple', linestyle='--', linewidth=2,
                      alpha=0.7, label=f'$\\gamma^* = {self.gamma_star:.4f}$', zorder=10)

        ax.set_xlabel('Individual Taste Weight $\\gamma$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95, fontsize=10, ncol=2 if len(metrics_dict) > 4 else 1)

        fig.tight_layout()
        return fig

    def plot_hysteresis_loop(
        self,
        gamma_up: np.ndarray,
        equilibria_up: np.ndarray,
        gamma_down: np.ndarray,
        equilibria_down: np.ndarray,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot hysteresis loop: equilibrium depends on path (increasing vs decreasing γ).

        Args:
            gamma_up: γ values for increasing sweep
            equilibria_up: Equilibria during upward sweep
            gamma_down: γ values for decreasing sweep
            equilibria_down: Equilibria during downward sweep
            ax: Existing axes

        Returns:
            Figure

        Theory:
            For γ < γ*, system can be in different equilibria depending on
            initial conditions. Hysteresis loop shows path dependence.

        Example:
            >>> # Sweep up from γ=0.5, starting at p=0.9
            >>> # Sweep down from γ=0.8, starting at p=0.5
            >>> fig = plotter.plot_hysteresis_loop(...)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Plot upward sweep
        ax.plot(gamma_up, equilibria_up, linewidth=2.5, color='#2E86AB',
               marker='o', markersize=5, label='Increasing $\\gamma$ →', alpha=0.8)

        # Plot downward sweep
        ax.plot(gamma_down, equilibria_down, linewidth=2.5, color='#F77F00',
               marker='s', markersize=5, label='← Decreasing $\\gamma$', alpha=0.8)

        # Hysteresis region
        gamma_overlap = np.intersect1d(gamma_up, gamma_down)
        if len(gamma_overlap) > 0:
            # Shade region where paths differ
            ax.fill_between(gamma_overlap,
                           np.interp(gamma_overlap, gamma_up, equilibria_up),
                           np.interp(gamma_overlap, gamma_down, equilibria_down),
                           alpha=0.3, color='gray', label='Hysteresis region')

        # Critical threshold
        ax.axvline(self.gamma_star, color='purple', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'$\\gamma^* = {self.gamma_star:.4f}$')

        # Reference lines
        ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

        ax.set_xlabel('Individual Taste Weight $\\gamma$', fontsize=13)
        ax.set_ylabel('Equilibrium Market Share $p^*$', fontsize=13)
        ax.set_title('Hysteresis Loop: Path-Dependent Equilibria',
                    fontsize=15, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.95, fontsize=11)

        # Annotations
        ax.annotate('Jump up', xy=(self.gamma_star + 0.01, 0.7),
                   xytext=(self.gamma_star + 0.05, 0.85),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'),
                   fontsize=10, color='#2E86AB', fontweight='bold')

        ax.annotate('Jump down', xy=(self.gamma_star - 0.01, 0.3),
                   xytext=(self.gamma_star - 0.05, 0.15),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#F77F00'),
                   fontsize=10, color='#F77F00', fontweight='bold')

        fig.tight_layout()
        return fig

    def plot_bifurcation_with_variance(
        self,
        gamma_values: np.ndarray,
        equilibria_data: Dict[float, List[float]],
        variances: np.ndarray,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Combined plot: Bifurcation diagram + variance.

        Args:
            gamma_values: Array of γ values
            equilibria_data: Dict γ -> equilibria list
            variances: Variance at each γ
            ax: Existing axes

        Returns:
            Figure with twin axes

        Example:
            >>> fig = plotter.plot_bifurcation_with_variance(gammas, eq_data, variances)
        """
        with plt.style.context(self.style):
            fig, ax1 = plt.subplots(figsize=self.figsize)

        # Plot bifurcation on primary axis
        stable_gammas, stable_equilibria = [], []
        for gamma in gamma_values:
            if gamma in equilibria_data:
                for eq in equilibria_data[gamma]:
                    stable_gammas.append(gamma)
                    stable_equilibria.append(eq)

        ax1.scatter(stable_gammas, stable_equilibria, c='#2E86AB', s=30,
                   alpha=0.7, label='Equilibria $p^*$', marker='o', linewidths=0)

        ax1.axvline(self.gamma_star, color='purple', linestyle='--', linewidth=2,
                   alpha=0.7, zorder=10)

        ax1.set_xlabel('Individual Taste Weight $\\gamma$', fontsize=13)
        ax1.set_ylabel('Equilibrium $p^*$', fontsize=13, color='#2E86AB')
        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Plot variance on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(gamma_values, variances, linewidth=2.5, color='#A23B72',
                marker='s', markersize=4, label='Variance $\\mathrm{Var}[p]$', alpha=0.8)

        ax2.set_ylabel('Variance $\\mathrm{Var}[p_\\infty]$', fontsize=13, color='#A23B72')
        ax2.tick_params(axis='y', labelcolor='#A23B72')

        # Title
        fig.suptitle('Bifurcation and Critical Slowing Down',
                    fontsize=15, fontweight='bold')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95)

        fig.tight_layout()
        return fig
