"""
Beta Distribution Difference CDF and PDF.

Implements F_Z(z) and f_Z(z) for Z = θ₁ - θ₂ where θᵢ ~ Beta(α₀, β₀).

For computational efficiency, F_Z is precomputed on a grid and interpolated.
This follows research.md decision 2: numerical integration with cached results.

Equation References:
    - Eq 12: Z = θ₁ - θ₂ (preference difference)
    - Eq 5: f_Z(0) = 6/5 (analytical result for Beta(2,2))
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import beta as beta_dist

# Prior parameters (fixed per specification)
ALPHA_0 = 2.0
BETA_0 = 2.0


def _beta_pdf(x, alpha, beta):
    """PDF of Beta distribution."""
    return beta_dist.pdf(x, alpha, beta)


def _beta_cdf(x, alpha, beta):
    """CDF of Beta distribution."""
    return beta_dist.cdf(x, alpha, beta)


def _compute_F_Z_exact(z, alpha=ALPHA_0, beta=BETA_0):
    """
    Compute F_Z(z) = P[Z ≤ z] via numerical integration.

    Args:
        z: Value at which to evaluate CDF
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter

    Returns:
        F_Z(z): CDF value

    Equation:
        F_Z(z) = P[θ₁ - θ₂ ≤ z]
               = ∫ f_θ₁(x) · P[θ₂ ≥ x - z] dx
               = ∫ f_θ₁(x) · [1 - F_θ₂(x - z)] dx

    Note:
        P[θ₁ - θ₂ ≤ z] = P[θ₂ ≥ θ₁ - z]
        So we integrate f_θ₁(x) times the survival function of θ₂ at (x-z).
    """

    def integrand(x):
        # Probability density at x for θ₁
        pdf_theta1 = _beta_pdf(x, alpha, beta)

        # P[θ₂ ≥ x - z] = 1 - F_θ₂(x - z)
        if x - z <= 0:
            survival_theta2 = 1.0  # P[θ₂ ≥ negative] = 1
        elif x - z >= 1:
            survival_theta2 = 0.0  # P[θ₂ ≥ 1+] = 0
        else:
            survival_theta2 = 1.0 - _beta_cdf(x - z, alpha, beta)

        return pdf_theta1 * survival_theta2

    # Integration limits: x ∈ [0, 1] (support of θ₁)
    lower_limit = 0.0
    upper_limit = 1.0

    result, _ = quad(integrand, lower_limit, upper_limit, epsabs=1e-10, epsrel=1e-10)
    return result


def _compute_f_Z_exact(z, alpha=ALPHA_0, beta=BETA_0):
    """
    Compute f_Z(z) = P[Z=z] via numerical integration.

    Args:
        z: Value at which to evaluate PDF
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter

    Returns:
        f_Z(z): PDF value

    Equation:
        f_Z(z) = ∫ f_θ₁(x) · f_θ₂(x - z) dx  [Cross-correlation]
    """

    def integrand(x):
        # PDF at x for θ₁ and at (x - z) for θ₂
        return _beta_pdf(x, alpha, beta) * _beta_pdf(x - z, alpha, beta)

    lower_limit = max(0.0, z)
    upper_limit = min(1.0, 1.0 + z)

    if lower_limit >= upper_limit:
        return 0.0

    result, _ = quad(integrand, lower_limit, upper_limit, epsabs=1e-9, epsrel=1e-9)
    return result


def _precompute_F_Z_grid(z_min=-1.0, z_max=1.0, n_points=2001):
    """
    Precompute F_Z on a grid for fast interpolation.
    """
    z_grid = np.linspace(z_min, z_max, n_points)
    F_values = np.array([_compute_F_Z_exact(z) for z in z_grid])
    return interp1d(z_grid, F_values, kind="linear", bounds_error=False, fill_value=(0.0, 1.0))


def _precompute_f_Z_grid(z_min=-1.0, z_max=1.0, n_points=2001):
    """
    Precompute f_Z on a grid for fast interpolation.
    """
    z_grid = np.linspace(z_min, z_max, n_points)
    f_values = np.array([_compute_f_Z_exact(z) for z in z_grid])
    return interp1d(z_grid, f_values, kind="linear", bounds_error=False, fill_value=0.0)


# Module-level cache: precompute lookup tables
_F_Z_INTERPOLATOR = _precompute_F_Z_grid()
_f_Z_INTERPOLATOR = _precompute_f_Z_grid()


def F_Z(z):
    """
    CDF of preference difference Z = θ₁ - θ₂.
    """
    if np.isscalar(z):
        result = float(_F_Z_INTERPOLATOR(z))
        # Clip to [0, 1] to handle numerical precision
        return np.clip(result, 0.0, 1.0)
    else:
        result = _F_Z_INTERPOLATOR(np.asarray(z))
        # Clip to [0, 1] to handle numerical precision
        return np.clip(result, 0.0, 1.0)


def f_Z(z):
    """
    PDF of preference difference Z = θ₁ - θ₂.
    """
    if np.isscalar(z):
        return float(_f_Z_INTERPOLATOR(z))
    else:
        return _f_Z_INTERPOLATOR(np.asarray(z))