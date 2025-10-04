"""
Analytical Constants for Consumer Learning Model.

This module defines exact mathematical constants derived from the Beta(2,2)
distribution and equilibrium analysis.

Equation References:
    - F_Z_AT_ZERO: Eq 5 (PDF of preference difference at zero)
    - GAMMA_STAR: Eq 8 (Critical bifurcation threshold)
    - HETEROGENEITY: Eq 6 (H parameter, preference heterogeneity measure)
"""

# Eq 5: f_Z(0) = 6/5 = 1.2
# PDF of Z = θ₁ - θ₂ evaluated at z=0, where θᵢ ~ Beta(2,2)
F_Z_AT_ZERO = 6.0 / 5.0

# Eq 8: γ* = 12/17 ≈ 0.7058823529411765
# Critical threshold for bifurcation: symmetric equilibrium stable iff γ > γ*
GAMMA_STAR = 12.0 / 17.0

# Eq 6: H = 23/60 ≈ 0.383333...
# Heterogeneity parameter measuring preference diversity
HETEROGENEITY = 23.0 / 60.0
