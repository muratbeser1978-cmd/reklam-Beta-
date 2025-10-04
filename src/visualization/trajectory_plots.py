"""
Trajectory Visualization.

Time series plots for market share, welfare, learning dynamics.

Usage:
    from src.visualization.trajectory_plots import TrajectoryPlotter

    plotter = TrajectoryPlotter()
    fig = plotter.plot_market_share_trajectory(result)
    fig.savefig('market_share.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class TrajectoryPlotter:
    """
    Visualize time series trajectories from simulation results.

    Methods:
        plot_market_share_trajectory: p₁(t) over time
        plot_welfare_trajectory: W(t) and components
        plot_learning_trajectory: Posterior convergence
        plot_multi_trajectory: Compare multiple runs
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize plotter.

        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize

    def plot_market_share_trajectory(
        self,
        p1_trajectory: np.ndarray,
        gamma: float,
        gamma_star: float = 12/17,
        burnin: int = 0,
        title: Optional[str] = None,
        show_equilibrium: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot market share p₁(t) over time.

        Args:
            p1_trajectory: (T+1,) market share trajectory
            gamma: Individual taste weight
            gamma_star: Critical threshold
            burnin: Burnin period to shade
            title: Custom title
            show_equilibrium: Whether to show equilibrium line
            ax: Existing axes (if None, create new figure)

        Returns:
            Matplotlib Figure

        Example:
            >>> fig = plotter.plot_market_share_trajectory(result.p1_trajectory, gamma=0.7)
            >>> fig.savefig('market_share.png', dpi=300)
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        T = len(p1_trajectory) - 1
        t = np.arange(T + 1)

        # Plot trajectory
        ax.plot(t, p1_trajectory, linewidth=2, label='$p_1(t)$', color='#2E86AB')

        # Shade burnin period
        if burnin > 0:
            ax.axvspan(0, burnin, alpha=0.2, color='gray', label='Burn-in')

        # Show equilibrium region
        if show_equilibrium:
            equilibrium_start = max(burnin, int(0.8 * T))
            equilibrium_value = p1_trajectory[equilibrium_start:].mean()
            ax.axhline(equilibrium_value, color='red', linestyle='--',
                      linewidth=1.5, alpha=0.7, label=f'Equilibrium ≈ {equilibrium_value:.3f}')

        # Reference lines
        ax.axhline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)

        # Labels and styling
        ax.set_xlabel('Time $t$', fontsize=12)
        ax.set_ylabel('Market Share $p_1(t)$', fontsize=12)

        if title is None:
            regime = "Below critical" if gamma < gamma_star else "Above critical"
            title = f'Market Share Trajectory ($\\gamma={gamma:.3f}$, {regime})'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)

        fig.tight_layout()
        return fig

    def plot_welfare_trajectory(
        self,
        welfare_analysis,
        burnin: int = 0,
        show_components: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot welfare W(t) and components over time.

        Args:
            welfare_analysis: WelfareAnalysisCorrected object
            burnin: Burnin period
            show_components: Show Q̄₁ and Q̄₂ components
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> welfare = welfare_calculator.compute_welfare(...)
            >>> fig = plotter.plot_welfare_trajectory(welfare)
        """
        if ax is None:
            with plt.style.context(self.style):
                if show_components:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                else:
                    fig, ax1 = plt.subplots(figsize=self.figsize)
                    ax2 = None
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = None

        T = len(welfare_analysis.welfare_trajectory) - 1
        t = np.arange(T + 1)

        # Main welfare trajectory
        ax1.plot(t, welfare_analysis.welfare_trajectory, linewidth=2,
                label='$W(t)$ (Eq 64)', color='#A23B72')
        ax1.plot(t, welfare_analysis.ex_ante_welfare, linewidth=2, linestyle='--',
                label='Ex-ante (for comparison)', color='#F18F01', alpha=0.7)

        # Optimal and equilibrium levels
        ax1.axhline(welfare_analysis.welfare_optimal, color='green', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'$W^{{opt}}$ = {welfare_analysis.welfare_optimal:.3f}')
        ax1.axhline(welfare_analysis.welfare_equilibrium, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'$W^{{eq}}$ = {welfare_analysis.welfare_equilibrium:.3f}')

        # Welfare loss annotation
        ax1.annotate(f'Welfare Loss: $\\Delta W = {welfare_analysis.welfare_loss:.4f}$',
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    ha='center', va='top')

        if burnin > 0:
            ax1.axvspan(0, burnin, alpha=0.2, color='gray')

        ax1.set_ylabel('Welfare $W(t)$', fontsize=12)
        ax1.set_title('Social Welfare Trajectory (Eq 64)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', framealpha=0.9)

        # Q̄ components
        if show_components and ax2 is not None:
            ax2.plot(t, welfare_analysis.Q_bar_trajectories[:, 0], linewidth=2,
                    label='$\\bar{Q}_1(t)$ (Brand 1 choosers)', color='#2E86AB')
            ax2.plot(t, welfare_analysis.Q_bar_trajectories[:, 1], linewidth=2,
                    label='$\\bar{Q}_2(t)$ (Brand 2 choosers)', color='#F77F00')

            ax2.set_xlabel('Time $t$', fontsize=12)
            ax2.set_ylabel('Average Utility $\\bar{Q}_a(t)$', fontsize=12)
            ax2.set_title('Utility Components (Eq 63)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', framealpha=0.9)

            if burnin > 0:
                ax2.axvspan(0, burnin, alpha=0.2, color='gray')
        else:
            ax1.set_xlabel('Time $t$', fontsize=12)

        fig.tight_layout()
        return fig

    def plot_learning_trajectory(
        self,
        consumer_state,
        consumer_ids: Optional[List[int]] = None,
        burnin: int = 0,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot posterior belief convergence for selected consumers.

        Args:
            consumer_state: ConsumerState object
            consumer_ids: List of consumer IDs to plot (default: first 5)
            burnin: Burnin period
            ax: Existing axes

        Returns:
            Figure

        Note:
            This shows final posterior means. For trajectory over time,
            need full history (not available in final state).
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = None

        if consumer_ids is None:
            consumer_ids = list(range(min(5, consumer_state.theta.shape[0])))

        N_plot = len(consumer_ids)
        colors = plt.cm.viridis(np.linspace(0, 1, N_plot))

        # Plot 1: True vs posterior means
        for idx, i in enumerate(consumer_ids):
            true_theta = consumer_state.theta[i]
            post_mean = consumer_state.theta_hat[i]

            ax1.scatter(true_theta[0], true_theta[1], marker='x', s=100,
                       color=colors[idx], label=f'True $\\theta_{i}$', alpha=0.7)
            ax1.scatter(post_mean[0], post_mean[1], marker='o', s=100,
                       color=colors[idx], label=f'Posterior $\\hat{{\\theta}}_{i}$')
            ax1.plot([true_theta[0], post_mean[0]], [true_theta[1], post_mean[1]],
                    color=colors[idx], linestyle='--', alpha=0.5)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='$\\theta_1 = \\theta_2$')
        ax1.set_xlabel('$\\theta_{i,1}$ (Brand 1)', fontsize=12)
        ax1.set_ylabel('$\\theta_{i,2}$ (Brand 2)', fontsize=12)
        ax1.set_title('Learning Convergence', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Plot 2: Posterior variance
        if ax2 is not None:
            alpha = consumer_state.alpha
            beta_param = consumer_state.beta_param

            # Var[Beta(α,β)] = αβ/((α+β)²(α+β+1))
            var_brand1 = (alpha[:, 0] * beta_param[:, 0]) / \
                        ((alpha[:, 0] + beta_param[:, 0])**2 * (alpha[:, 0] + beta_param[:, 0] + 1))
            var_brand2 = (alpha[:, 1] * beta_param[:, 1]) / \
                        ((alpha[:, 1] + beta_param[:, 1])**2 * (alpha[:, 1] + beta_param[:, 1] + 1))

            ax2.hist(var_brand1, bins=30, alpha=0.6, label='Brand 1', color='#2E86AB')
            ax2.hist(var_brand2, bins=30, alpha=0.6, label='Brand 2', color='#F77F00')

            ax2.set_xlabel('Posterior Variance', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Posterior Uncertainty Distribution', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_multi_trajectory(
        self,
        trajectories: List[np.ndarray],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title: str = 'Market Share Trajectories',
        show_mean: bool = True,
        show_std: bool = True,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot multiple trajectories for comparison.

        Args:
            trajectories: List of (T+1,) arrays
            labels: Labels for each trajectory
            colors: Colors for each trajectory
            title: Plot title
            show_mean: Show mean trajectory
            show_std: Show ±1 std band
            ax: Existing axes

        Returns:
            Figure

        Example:
            >>> trajectories = [run1.p1_trajectory, run2.p1_trajectory, ...]
            >>> fig = plotter.plot_multi_trajectory(trajectories, labels=['Run 1', 'Run 2'])
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        n_traj = len(trajectories)

        if labels is None:
            labels = [f'Run {i+1}' for i in range(n_traj)]

        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, n_traj))

        # Plot individual trajectories
        for traj, label, color in zip(trajectories, labels, colors):
            t = np.arange(len(traj))
            ax.plot(t, traj, linewidth=1.5, alpha=0.6, label=label, color=color)

        # Aggregate statistics
        if show_mean or show_std:
            # Ensure all trajectories same length
            min_len = min(len(t) for t in trajectories)
            traj_array = np.array([t[:min_len] for t in trajectories])
            t = np.arange(min_len)

            mean_traj = traj_array.mean(axis=0)
            std_traj = traj_array.std(axis=0)

            if show_mean:
                ax.plot(t, mean_traj, linewidth=3, color='black',
                       label=f'Mean (n={n_traj})', alpha=0.8)

            if show_std:
                ax.fill_between(t, mean_traj - std_traj, mean_traj + std_traj,
                               alpha=0.2, color='gray', label='±1 Std')

        # Reference lines
        ax.axhline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_xlabel('Time $t$', fontsize=12)
        ax.set_ylabel('Market Share $p_1(t)$', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9, ncol=2 if n_traj > 5 else 1)

        fig.tight_layout()
        return fig

    def plot_critical_slowing_down(
        self,
        trajectory: np.ndarray,
        metrics,
        window_size: int = 50,
        ax: Optional[Axes] = None,
    ) -> Figure:
        """
        Plot critical slowing down indicators.

        Args:
            trajectory: Market share trajectory
            metrics: CriticalSlowingMetrics object
            window_size: Rolling window size
            ax: Existing axes

        Returns:
            Figure
        """
        if ax is None:
            with plt.style.context(self.style):
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        else:
            fig = ax.figure
            ax1, ax2, ax3 = ax, None, None

        t = np.arange(len(trajectory))

        # Plot 1: Trajectory
        ax1.plot(t, trajectory, linewidth=2, color='#2E86AB')
        ax1.set_ylabel('$p_1(t)$', fontsize=12)
        ax1.set_title(f'Market Share (γ = {metrics.gamma:.3f}, Distance to critical: {metrics.distance_to_critical:.4f})',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0.5, color='black', linestyle=':', alpha=0.5)

        # Plot 2: Rolling variance
        if ax2 is not None:
            rolling_var = []
            for i in range(len(trajectory) - window_size + 1):
                window = trajectory[i:i+window_size]
                rolling_var.append(np.var(window))

            t_rolling = t[window_size//2:window_size//2+len(rolling_var)]
            ax2.plot(t_rolling, rolling_var, linewidth=2, color='#A23B72')
            ax2.axhline(metrics.variance, color='red', linestyle='--',
                       label=f'Overall Var = {metrics.variance:.6f}')
            ax2.set_ylabel('Rolling Variance', fontsize=12)
            ax2.set_title('Variance (Early Warning Signal)', fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Autocorrelation
        if ax3 is not None:
            max_lag = min(50, len(trajectory) // 4)
            lags = range(1, max_lag)
            autocorrs = [self._compute_autocorr(trajectory, lag) for lag in lags]

            ax3.bar(lags, autocorrs, color='#F77F00', alpha=0.7)
            ax3.axhline(metrics.autocorr_lag1, color='red', linestyle='--',
                       label=f'AC(1) = {metrics.autocorr_lag1:.3f}')
            ax3.set_xlabel('Lag', fontsize=12)
            ax3.set_ylabel('Autocorrelation', fontsize=12)
            ax3.set_title('Autocorrelation Function', fontsize=12)
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def _compute_autocorr(self, x: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag."""
        if len(x) <= lag or lag == 0:
            return 1.0 if lag == 0 else np.nan

        x1 = x[:-lag]
        x2 = x[lag:]

        if np.std(x1) == 0 or np.std(x2) == 0:
            return 1.0 if lag == 0 else 0.0

        return np.corrcoef(x1, x2)[0, 1]
