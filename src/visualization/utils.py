"""
Visualization utilities.

Helper functions for visualization modules.
"""

import numpy as np
from typing import List
from scipy.optimize import fsolve
from src.analysis.best_response import best_response


def find_equilibria(gamma: float, tol: float = 1e-6, grid_size: int = 500) -> List[float]:
    """
    Find equilibria p* where BR(p*, γ) = p*.

    Args:
        gamma: Individual taste weight
        tol: Tolerance for equilibrium detection
        grid_size: Number of grid points for initial search

    Returns:
        List of equilibrium values p*

    Method:
        1. Dense grid search to find sign changes in f(p) = BR(p) - p
        2. Refine each crossing with fsolve
        3. Deduplicate and sort

    Theory:
        - Equilibria occur where f(p) = BR(p) - p = 0
        - For γ > γ*: 1 equilibrium (p*=0.5)
        - For γ < γ*: 3 equilibria (p⁻ < 0.5 < p⁺)
    """
    equilibria = []

    # Dense grid for sign change detection
    p_grid = np.linspace(0.0, 1.0, grid_size)
    f_values = np.array([best_response(p, gamma) - p for p in p_grid])

    # Find sign changes (crossings of zero)
    # Use product of consecutive values < 0 to detect sign change
    sign_products = f_values[:-1] * f_values[1:]
    sign_changes = np.where(sign_products < 0)[0]

    # Refine each crossing with fsolve
    for idx in sign_changes:
        # Initial guess: midpoint of bracketing interval
        p_left = p_grid[idx]
        p_right = p_grid[idx + 1]
        p_init = (p_left + p_right) / 2.0

        def eq_condition(p_val):
            """Equilibrium condition: BR(p) - p = 0"""
            # p_val is a scalar from fsolve
            return best_response(p_val, gamma) - p_val

        try:
            # fsolve expects scalar input/output for 1D problems
            p_star = fsolve(eq_condition, p_init)[0]

            # Verify solution
            residual = eq_condition(p_star)

            # Check convergence and validity
            if abs(residual) < tol and 0.0 <= p_star <= 1.0:
                # Check if already found (deduplicate)
                is_duplicate = any(abs(p_star - eq) < tol for eq in equilibria)

                if not is_duplicate:
                    equilibria.append(float(p_star))
        except:
            pass

    # Additional check: p=0.5 is always an equilibrium
    # If not found yet, add it
    if len(equilibria) == 0 or not any(abs(0.5 - eq) < 0.01 for eq in equilibria):
        br_half = best_response(0.5, gamma)
        if abs(br_half - 0.5) < tol:
            equilibria.append(0.5)

    # Sort and return
    equilibria = sorted(equilibria)
    return equilibria
