"""
Distribution Visualization.

Histograms, KDE plots, qq-plots for preferences, beliefs, and outcomes.

Usage:
    from src.visualization.distributions import DistributionPlotter

    plotter = DistributionPlotter()
    fig = plotter.plot_preference_distribution(theta)
    fig.savefig('preferences.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats as scipy_stats


class DistributionPlotter:
    """
    Visualize distributions of preferences, beliefs, and outcomes.

    Methods:
        plot_preference_distribution: True preferences θ
        plot_posterior_distribution: Posterior beliefs θ̂
        plot_equilibrium_distribution: Final equilibria across runs
        plot_qq_plot: Quantile-quantile plots
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (10, 6)):
        """Initialize plotter."""
        self.style = style
        self.figsize = figsize

    def plot_preference_distribution(
        self,
        theta: np.ndarray,
        show_kde: bool = True,
        show_theory: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot distribution of true preferences θ.

        Args:
            theta: (N, 2) preference matrix
            show_kde: Show kernel density estimate
            show_theory: Show theoretical Beta(2,2) PDF
            ax: Existing axes

        Returns:
            Figure

        Theory:
            θ_{i,a} ~ Beta(2, 2)

        Example:
            >>> plotter = DistributionPlotter()
            >>> fig = plotter.plot_preference_distribution(consumer_state.theta)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = None

        # Brand 1 preferences
        theta_1 = theta[:, 0]
        ax1.hist(theta_1, bins=30, density=True, alpha=0.6, color='#2E86AB',
                edgecolor='black', label='Observed')

        if show_kde:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(theta_1)
            x_grid = np.linspace(0, 1, 200)
            ax1.plot(x_grid, kde(x_grid), linewidth=2.5, color='#2E86AB',
                    label='KDE', alpha=0.8)

        if show_theory:
            x_grid = np.linspace(0, 1, 200)
            beta_pdf = scipy_stats.beta.pdf(x_grid, 2, 2)
            ax1.plot(x_grid, beta_pdf, linewidth=2.5, color='red', linestyle='--',
                    alpha=0.7, label='Beta(2,2) Theory')

        ax1.set_xlabel('$\\theta_{i,1}$ (Brand 1 Preference)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Brand 1 Preference Distribution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Brand 2 preferences
        if ax2 is not None:
            theta_2 = theta[:, 1]
            ax2.hist(theta_2, bins=30, density=True, alpha=0.6, color='#F77F00',
                    edgecolor='black', label='Observed')

            if show_kde:
                kde = gaussian_kde(theta_2)
                ax2.plot(x_grid, kde(x_grid), linewidth=2.5, color='#F77F00',
                        label='KDE', alpha=0.8)

            if show_theory:
                ax2.plot(x_grid, beta_pdf, linewidth=2.5, color='red', linestyle='--',
                        alpha=0.7, label='Beta(2,2) Theory')

            ax2.set_xlabel('$\\theta_{i,2}$ (Brand 2 Preference)', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.set_title('Brand 2 Preference Distribution', fontsize=14, fontweight='bold')
            ax2.legend(loc='best', framealpha=0.9)
            ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_posterior_distribution(
        self,
        consumer_state,
        brand: int = 0,
        show_convergence: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot posterior belief distribution θ̂.

        Args:
            consumer_state: ConsumerState object
            brand: Which brand to plot (0 or 1)
            show_convergence: Compare with true θ
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> fig = plotter.plot_posterior_distribution(consumer_state, brand=0)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        theta_true = consumer_state.theta[:, brand]
        theta_hat = consumer_state.theta_hat[:, brand]

        # Histograms
        ax.hist(theta_true, bins=30, density=True, alpha=0.5, color='#2E86AB',
               edgecolor='black', label='True $\\theta$')
        ax.hist(theta_hat, bins=30, density=True, alpha=0.5, color='#A23B72',
               edgecolor='black', label='Posterior $\\hat{\\theta}$')

        # KDE
        from scipy.stats import gaussian_kde
        x_grid = np.linspace(0, 1, 200)

        kde_true = gaussian_kde(theta_true)
        ax.plot(x_grid, kde_true(x_grid), linewidth=2.5, color='#2E86AB',
               alpha=0.8, label='True (KDE)')

        kde_hat = gaussian_kde(theta_hat)
        ax.plot(x_grid, kde_hat(x_grid), linewidth=2.5, color='#A23B72',
               alpha=0.8, label='Posterior (KDE)')

        # Convergence measure
        if show_convergence:
            mse = np.mean((theta_true - theta_hat) ** 2)
            mae = np.mean(np.abs(theta_true - theta_hat))

            ax.text(0.05, 0.95, f'MSE: {mse:.4f}\\nMAE: {mae:.4f}',
                   transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top')

        brand_name = f'Brand {brand + 1}'
        ax.set_xlabel(f'Preference for {brand_name}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Learning Convergence: True vs Posterior ({brand_name})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_equilibrium_distribution(
        self,
        equilibria: np.ndarray,
        gamma: float,
        theoretical_equilibria: Optional[List[float]] = None,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot distribution of final equilibria across multiple runs.

        Args:
            equilibria: (n_runs,) array of final p₁ values
            gamma: Parameter value
            theoretical_equilibria: Expected equilibria from theory
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> equilibria = [run.p1_trajectory[-1] for run in results]
            >>> fig = plotter.plot_equilibrium_distribution(np.array(equilibria), gamma=0.65)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Histogram
        ax.hist(equilibria, bins=40, density=True, alpha=0.7, color='#6A4C93',
               edgecolor='black', label='Observed')

        # KDE
        if len(equilibria) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(equilibria)
            x_grid = np.linspace(0, 1, 300)
            ax.plot(x_grid, kde(x_grid), linewidth=2.5, color='#6A4C93',
                   alpha=0.8, label='KDE')

        # Mark theoretical equilibria
        if theoretical_equilibria is not None:
            for eq in theoretical_equilibria:
                ax.axvline(eq, color='red', linestyle='--', linewidth=2,
                          alpha=0.7, label=f'Theory: $p^* = {eq:.3f}$')

        # Statistics
        mean_eq = np.mean(equilibria)
        std_eq = np.std(equilibria)

        ax.axvline(mean_eq, color='green', linestyle='-', linewidth=2,
                  alpha=0.7, label=f'Mean: {mean_eq:.3f}')
        ax.axvspan(mean_eq - std_eq, mean_eq + std_eq, alpha=0.2, color='green',
                  label=f'±1 Std ({std_eq:.4f})')

        ax.set_xlabel('Equilibrium Market Share $p^*$', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Equilibrium Distribution ($\\gamma = {gamma:.3f}$, n={len(equilibria)})',
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_qq_plot(
        self,
        data: np.ndarray,
        distribution: str = 'beta',
        params: Optional[Tuple] = None,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Quantile-quantile plot to test distributional assumptions.

        Args:
            data: Sample data
            distribution: Theoretical distribution ('beta', 'norm', etc.)
            params: Distribution parameters (if None, fit from data)
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> fig = plotter.plot_qq_plot(theta[:, 0], distribution='beta', params=(2, 2))
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Get distribution
        if distribution == 'beta':
            if params is None:
                params = scipy_stats.beta.fit(data)
            dist = scipy_stats.beta(*params)
        elif distribution == 'norm':
            if params is None:
                params = scipy_stats.norm.fit(data)
            dist = scipy_stats.norm(*params)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Theoretical quantiles
        sorted_data = np.sort(data)
        n = len(sorted_data)
        theoretical_quantiles = dist.ppf(np.linspace(0.01, 0.99, n))

        # Q-Q plot
        ax.scatter(theoretical_quantiles, sorted_data, s=30, alpha=0.6,
                  color='#2E86AB', edgecolors='black', linewidths=0.5)

        # 45° line
        limits = [max(theoretical_quantiles.min(), sorted_data.min()),
                 min(theoretical_quantiles.max(), sorted_data.max())]
        ax.plot(limits, limits, 'r--', linewidth=2, alpha=0.7, label='Perfect fit')

        # Labels
        ax.set_xlabel(f'Theoretical Quantiles ({distribution})', fontsize=12)
        ax.set_ylabel('Sample Quantiles', fontsize=12)
        ax.set_title(f'Q-Q Plot: {distribution.capitalize()} Distribution',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_belief_scatter(
        self,
        consumer_state,
        n_samples: int = 200,
        show_diagonal: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Scatter plot: true θ vs posterior θ̂ for both brands.

        Args:
            consumer_state: ConsumerState object
            n_samples: Number of consumers to plot
            show_diagonal: Show θ = θ̂ line
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> fig = plotter.plot_belief_scatter(consumer_state, n_samples=100)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = None

        N = consumer_state.theta.shape[0]
        indices = np.random.choice(N, size=min(n_samples, N), replace=False)

        # Brand 1
        theta_1 = consumer_state.theta[indices, 0]
        theta_hat_1 = consumer_state.theta_hat[indices, 0]

        ax1.scatter(theta_1, theta_hat_1, s=50, alpha=0.6, color='#2E86AB',
                   edgecolors='black', linewidths=0.5)

        if show_diagonal:
            ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7,
                    label='$\\hat{\\theta} = \\theta$ (perfect learning)')

        ax1.set_xlabel('True $\\theta_{i,1}$', fontsize=12)
        ax1.set_ylabel('Posterior $\\hat{\\theta}_{i,1}$', fontsize=12)
        ax1.set_title('Brand 1: Learning Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Correlation
        corr_1 = np.corrcoef(theta_1, theta_hat_1)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr_1:.3f}',
                transform=ax1.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')

        # Brand 2
        if ax2 is not None:
            theta_2 = consumer_state.theta[indices, 1]
            theta_hat_2 = consumer_state.theta_hat[indices, 1]

            ax2.scatter(theta_2, theta_hat_2, s=50, alpha=0.6, color='#F77F00',
                       edgecolors='black', linewidths=0.5)

            if show_diagonal:
                ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7,
                        label='$\\hat{\\theta} = \\theta$ (perfect learning)')

            ax2.set_xlabel('True $\\theta_{i,2}$', fontsize=12)
            ax2.set_ylabel('Posterior $\\hat{\\theta}_{i,2}$', fontsize=12)
            ax2.set_title('Brand 2: Learning Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

            corr_2 = np.corrcoef(theta_2, theta_hat_2)[0, 1]
            ax2.text(0.05, 0.95, f'Correlation: {corr_2:.3f}',
                    transform=ax2.transAxes, fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top')

        fig.tight_layout()
        return fig

    def plot_experience_distribution(
        self,
        consumer_state,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot distribution of effective experience n_eff.

        Args:
            consumer_state: ConsumerState object
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> fig = plotter.plot_experience_distribution(consumer_state)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = None

        n_eff_1 = consumer_state.n_eff[:, 0]
        n_eff_2 = consumer_state.n_eff[:, 1]

        # Brand 1
        ax1.hist(n_eff_1, bins=30, alpha=0.7, color='#2E86AB',
                edgecolor='black', label='Brand 1')

        ax1.axvline(n_eff_1.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {n_eff_1.mean():.1f}')

        ax1.set_xlabel('Effective Experience $n^{\\mathrm{eff}}_{i,1}$', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Brand 1 Experience Distribution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Brand 2
        if ax2 is not None:
            ax2.hist(n_eff_2, bins=30, alpha=0.7, color='#F77F00',
                    edgecolor='black', label='Brand 2')

            ax2.axvline(n_eff_2.mean(), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {n_eff_2.mean():.1f}')

            ax2.set_xlabel('Effective Experience $n^{\\mathrm{eff}}_{i,2}$', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Brand 2 Experience Distribution', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
