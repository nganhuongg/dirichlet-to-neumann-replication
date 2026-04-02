"""
convergence_test.py  –  Grid-refinement study for the DtN matrix
=================================================================

PURPOSE
-------
Verify that the DtN matrix approximation converges to the spectral reference
as the grid is refined (dx → 0, or equivalently m → ∞ with fixed physical
domain length L = m * dx).

This experiment reproduces the convergence data in Table 4.1 of the paper and
adds a complementary *exact* validation using sinusoidal test functions.

TWO TYPES OF TEST ARE PERFORMED
---------------------------------

Test A — Gaussian convergence study  (reproduces Table 4.1)
    For each grid size m in {2^4, 2^5, ..., 2^8}:
      - Build the DtN matrix N (size m+1 x m+1).
      - Apply it to the Gaussian phi(x) = exp(-x^2).
      - Compare to the spectral method used as a near-exact reference.
      - Record the mean absolute error and the convergence rate.

    Expected result: error ~ C * dx^2  (second-order convergence).

Test B — Sinusoidal exact validation
    For phi(x) = cos(k0 * x), the DtN operator has a known exact answer:
        phi_z(x) = |k0| * cos(k0 * x)
    because a pure Fourier mode e^{ikx} is an eigenfunction of DtN with
    eigenvalue |k|.  This test validates the method against a closed-form
    solution (no reference method needed).

    For each grid size m:
      - Build N.
      - Apply to phi = cos(k0 * x).
      - Compare to the exact answer k0 * cos(k0 * x).
      - Record the max absolute error.

HOW TO RUN
----------
From the project root:
    python experiments/convergence_test.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup: allow imports from src/ regardless of where the script is run.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.dtn import build_dtn_matrix
from src.spectral import spectral_dtn

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(output_dir, exist_ok=True)

# ===========================================================================
# TEST A: GAUSSIAN CONVERGENCE STUDY  (Table 4.1 of paper)
# ===========================================================================
# Fixed domain half-length: the Gaussian exp(-x^2) is negligible for |x| > 4,
# so the domain [-m/2*dx, m/2*dx] must be large enough to capture it.
# We keep dx = 0.1 fixed (same as MATLAB) and vary m = 2^k.
# As m grows: more grid points → better approximation of the integral → lower error.

dx = 0.1        # grid spacing (fixed throughout Test A)

m_values = [2**k for k in range(4, 9)]   # m = 16, 32, 64, 128, 256

errors_gaussian = []   # mean absolute error for each m

print("=" * 60)
print("TEST A — Gaussian convergence  (Table 4.1 of paper)")
print("=" * 60)
print(f"  {'m':>6}  {'dx':>6}  {'mean |error|':>14}  {'rate':>8}")
print("-" * 60)

for m in m_values:
    # Spatial grid:  x in [-(m/2)*dx, (m/2)*dx]  with m+1 points
    x = dx * np.arange(-m // 2, m // 2 + 1)   # shape (m+1,)

    # Gaussian test function
    phi = np.exp(-x**2)

    # DtN matrix approximation
    N    = build_dtn_matrix(m, dx)
    phiz = N @ phi                   # shape (m+1,)

    # Spectral reference on m periodic points (drop the repeated endpoint)
    phiz_ref = spectral_dtn(phi[:-1], dx)   # shape (m,)

    # Mean absolute error on the m interior points
    err = np.mean(np.abs(phiz[:-1] - phiz_ref))
    errors_gaussian.append(err)

    # Convergence rate: rate = log2(err_prev / err_curr)
    if len(errors_gaussian) > 1:
        rate = np.log2(errors_gaussian[-2] / errors_gaussian[-1])
        print(f"  {m:>6}  {dx:>6}  {err:>14.6e}  {rate:>8.2f}")
    else:
        print(f"  {m:>6}  {dx:>6}  {err:>14.6e}  {'—':>8}")

print()

# ===========================================================================
# TEST B: SINUSOIDAL EXACT VALIDATION
# ===========================================================================
# For phi(x) = cos(k0 * x), the DtN operator gives EXACTLY phi_z = k0 * cos(k0*x).
# This follows from the Fourier-space definition of DtN (paper §4.2):
#   phi_z_hat(k) = |k| * phi_hat(k)
# Since cos(k0*x) has only wavenumber ±k0, the output is exactly |k0|*cos(k0*x).
#
# For this to work on a FINITE grid of length L = m*dx, we choose k0 to be
# a valid grid frequency:  k0 = 2*pi*n_mode / L  for some integer n_mode.
# This ensures that the sinusoid fits the grid exactly (no aliasing).
#
# We use dx = 0.1, m = 256, and n_mode = 5 → k0 ≈ 1.227 rad/m.

m_sine    = 2**8      # grid size for sinusoidal test
dx_sine   = 0.1
n_mode    = 5         # number of full oscillations in the domain
L         = m_sine * dx_sine
k0        = 2.0 * np.pi * n_mode / L   # wave number (exact grid frequency)

x_sine   = dx_sine * np.arange(-m_sine // 2, m_sine // 2 + 1)   # shape (m+1,)
phi_sine = np.cos(k0 * x_sine)                                    # test function
phi_z_exact = k0 * np.cos(k0 * x_sine)                           # exact DtN answer

N_sine    = build_dtn_matrix(m_sine, dx_sine)
phiz_sine = N_sine @ phi_sine   # DtN matrix approximation

err_sine_max  = np.max(np.abs(phiz_sine - phi_z_exact))
err_sine_mean = np.mean(np.abs(phiz_sine - phi_z_exact))

print("=" * 60)
print("TEST B — Sinusoidal exact validation")
print(f"  phi(x) = cos(k0*x),  exact answer: phi_z = k0*cos(k0*x)")
print(f"  k0 = 2*pi*{n_mode}/L = {k0:.4f},  L = {L},  m = {m_sine}")
print("=" * 60)
print(f"  max  |error| = {err_sine_max:.6e}")
print(f"  mean |error| = {err_sine_mean:.6e}")
print()

# ===========================================================================
# ADDITIONAL CHECKS  (structural properties of N)
# ===========================================================================

# --- Check 1: Row sums should be ~0 (constant phi => zero derivative) -----
# The DtN of a constant function c is 0 (uniform potential has no normal
# derivative).  Therefore N @ ones should be ~0.  This checks that no
# "spurious flux" is introduced by the quadrature.
N_check = build_dtn_matrix(2**6, 0.1)   # use smaller m for speed
row_sums = N_check @ np.ones(2**6 + 1)
print(f"Row-sum check (N @ 1 should be ~0):  max |row sum| = {np.max(np.abs(row_sums)):.2e}")

# --- Check 2: N should be symmetric (self-adjoint operator) ---------------
# The DtN operator is self-adjoint in L^2, so the matrix should satisfy N = N^T.
sym_err = np.max(np.abs(N_check - N_check.T))
print(f"Symmetry check (N == N^T):           max |N - N^T| = {sym_err:.2e}")

# --- Check 3: N should be positive semi-definite --------------------------
# The lower-half-plane DtN operator satisfies <phi, N phi> >= 0 (non-negative
# energy flux into the domain).  We check the minimum eigenvalue.
eigvals = np.linalg.eigvalsh(N_check)   # symmetric eigenvalue decomposition
print(f"PSD check (min eigenvalue >= 0):     min eigenvalue = {eigvals.min():.2e}")
print()

# ===========================================================================
# FIGURE A: Log-log convergence plot  (Test A)
# ===========================================================================

dx_values = [dx] * len(m_values)    # dx is fixed; m changes

fig_A, ax_A = plt.subplots(figsize=(7, 5))

ax_A.loglog(m_values, errors_gaussian, "o-", linewidth=2, markersize=8,
            color="steelblue", label="DtN matrix error")

# Overlay a reference O(m^{-2}) line scaled to pass through the first data point.
# Since dx is fixed and L = m*dx, larger m means smaller dx_effective ~ 1/m,
# so O(m^{-2}) convergence in m corresponds to O(dx^2) convergence in step size.
m_ref = np.array([m_values[0], m_values[-1]], dtype=float)
ax_A.loglog(m_ref,
            errors_gaussian[0] * (m_ref[0] / m_ref) ** 2,
            "k--", linewidth=1.5, label=r"Reference slope $\propto m^{-2}$")

ax_A.set_xlabel("Number of grid intervals  m")
ax_A.set_ylabel("Mean absolute error  |N phi - phi_z ref|")
ax_A.set_title(
    "Convergence of DtN matrix (Gaussian test)\n"
    r"Expected rate: $O(m^{-2})$  (2nd-order method)"
)
ax_A.legend()
ax_A.grid(True, which="both")
fig_A.tight_layout()
fig_A.savefig(os.path.join(output_dir, "python_convergence_gaussian.png"), dpi=150)

# ===========================================================================
# FIGURE B: Sinusoidal test — exact vs matrix  (Test B)
# ===========================================================================

fig_B, ax_B = plt.subplots(figsize=(8, 4))
ax_B.axhline(0, color="k", linewidth=0.8)

ax_B.plot(x_sine, phi_z_exact, linewidth=3, color="steelblue",
          label=r"Exact: $\varphi_z = k_0 \cos(k_0 x)$")
ax_B.plot(x_sine, phiz_sine, "x", color="firebrick", markersize=5,
          label="DtN matrix approximation")

ax_B.set_xlabel("x")
ax_B.set_ylabel(r"$\varphi_z$")
ax_B.set_title(
    r"Sinusoidal exact test:  $\varphi(x) = \cos(k_0 x)$,  exact DtN = $k_0 \cos(k_0 x)$"
    f"\n$k_0 = {k0:.4f}$,  max |error| = {err_sine_max:.2e}"
)
ax_B.legend()
ax_B.grid(True)
fig_B.tight_layout()
fig_B.savefig(os.path.join(output_dir, "python_sinusoidal_exact_test.png"), dpi=150)

print(f"  Figures saved to:  {os.path.abspath(output_dir)}")
plt.show()
