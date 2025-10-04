"""
Quick visualization test.

Tests that all plotters can be instantiated and basic plots work.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.visualization import (
    TrajectoryPlotter,
    PhasePortraitPlotter,
    BifurcationPlotter,
    HeatmapPlotter,
    DistributionPlotter,
)

print("Testing visualization modules...")

# 1. Phase Portrait
print("✓ Testing PhasePortraitPlotter...")
phase_plotter = PhasePortraitPlotter()
fig = phase_plotter.plot_best_response(gamma=0.65)
plt.close(fig)
print("  - plot_best_response() works")

# 2. Bifurcation
print("✓ Testing BifurcationPlotter...")
bifurc_plotter = BifurcationPlotter()
gammas = np.linspace(0.6, 0.8, 20)
fig = bifurc_plotter.plot_bifurcation_diagram(gamma_values=gammas)
plt.close(fig)
print("  - plot_bifurcation_diagram() works")

# 3. Trajectory
print("✓ Testing TrajectoryPlotter...")
traj_plotter = TrajectoryPlotter()
p1_traj = 0.5 + 0.1 * np.random.randn(100)
p1_traj = np.clip(p1_traj, 0, 1)
fig = traj_plotter.plot_market_share_trajectory(p1_traj, gamma=0.7)
plt.close(fig)
print("  - plot_market_share_trajectory() works")

# 4. Heatmap
print("✓ Testing HeatmapPlotter...")
heatmap_plotter = HeatmapPlotter()
x = np.linspace(0.5, 0.9, 20)
y = np.linspace(0.7, 0.95, 15)
z = np.random.rand(15, 20)
fig = heatmap_plotter.plot_parameter_heatmap(x, y, z)
plt.close(fig)
print("  - plot_parameter_heatmap() works")

# 5. Distributions
print("✓ Testing DistributionPlotter...")
dist_plotter = DistributionPlotter()
theta = np.random.beta(2, 2, size=(100, 2))
fig = dist_plotter.plot_preference_distribution(theta)
plt.close(fig)
print("  - plot_preference_distribution() works")

print("\n" + "="*50)
print("✅ All visualization tests passed!")
print("="*50)
