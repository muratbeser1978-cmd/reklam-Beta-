"""
Visualization tools for simulation results.

Includes both legacy functional API and new class-based plotters.
"""

# Legacy functional API
from src.visualization.trajectories import (
    plot_trajectory,
    plot_multiple_trajectories,
    plot_trajectory_with_variance,
)
from src.visualization.bifurcation_diagram import (
    plot_bifurcation,
    plot_bifurcation_with_amplitude,
)
from src.visualization.potential_landscape import (
    plot_potential_landscape,
    plot_potential_slices,
    plot_potential_derivative,
)
from src.visualization.welfare_plots import (
    plot_welfare_loss,
    plot_welfare_trajectory,
    plot_welfare_comparison,
    plot_welfare_decomposition,
)

# New class-based plotters (Phase 2)
from src.visualization.trajectory_plots import TrajectoryPlotter
from src.visualization.phase_portraits import PhasePortraitPlotter
from src.visualization.bifurcation_diagrams import BifurcationPlotter
from src.visualization.heatmaps import HeatmapPlotter
from src.visualization.distributions import DistributionPlotter

__all__ = [
    # Legacy functional API
    "plot_trajectory",
    "plot_multiple_trajectories",
    "plot_trajectory_with_variance",
    "plot_bifurcation",
    "plot_bifurcation_with_amplitude",
    "plot_potential_landscape",
    "plot_potential_slices",
    "plot_potential_derivative",
    "plot_welfare_loss",
    "plot_welfare_trajectory",
    "plot_welfare_comparison",
    "plot_welfare_decomposition",
    # New class-based plotters
    "TrajectoryPlotter",
    "PhasePortraitPlotter",
    "BifurcationPlotter",
    "HeatmapPlotter",
    "DistributionPlotter",
]
