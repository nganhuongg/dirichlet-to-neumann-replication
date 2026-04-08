"""
gaussian_test.py  –  Validation of the DtN matrix against the spectral method
===============================================================================

PURPOSE
-------
Reproduce the validation experiment from paper §4.2 and Figures 4.2–4.3:
apply the DtN matrix to a Gaussian test function and compare the result to
the spectral (FFT) reference solution.

WHY A GAUSSIAN?
---------------
The Gaussian  phi(x) = exp(-x^2)  is the standard test function in the paper
for three reasons:
  1. It is smooth (infinitely differentiable), so the DtN integral converges
     well and the matrix approximation should be accurate.
  2. It decays rapidly to zero, so a finite domain is a good approximation of
     the infinite real line.
  3. Its Fourier transform is also a Gaussian (phi_hat(k) = sqrt(pi)*exp(-k^2/4)),
     which makes the spectral method's output easy to interpret.

WHAT THIS SCRIPT PRODUCES
--------------------------
  Figure 1  —  The Gaussian input function phi(x)  (reproduces Figure 4.2a)
  Figure 2  —  Comparison: spectral vs DtN matrix for phi_z  (Figure 4.2b)
  Figure 3  —  Several rows of matrix N, showing the Toeplitz shift pattern
               (reproduces Figure 4.3)
  Console   —  Mean absolute error between the two methods (Table 4.1 entry)

EXPECTED ERROR (Table 4.1 of the paper)
-----------------------------------------
  m = 256,  dx = 0.1  =>  mean absolute error ~ 0.001412

HOW TO RUN
----------
From the project root:
    python experiments/gaussian_test.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup: allow imports from src/ regardless of where the script is run.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dtn import build_dtn_matrix        # matrix DtN (our approximation)
from src.spectral import spectral_dtn       # FFT-based DtN (reference)

# ---------------------------------------------------------------------------
# Output directory: save figures alongside existing MATLAB output images.
# ---------------------------------------------------------------------------
output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(output_dir, exist_ok=True)

# ===========================================================================
# PARAMETERS  (matching Appendix A / DtNmatrix.m)
# ===========================================================================
dx = 0.1    # grid spacing
m  = 2**8   # 256 intervals → 257 grid points

# ===========================================================================
# GRID AND TEST FUNCTION
# ===========================================================================

# Uniform grid from -(m/2)*dx to +(m/2)*dx.
# In MATLAB:  x = dx * (-m/2 : m/2)'   (same thing, column vector)
x = dx * np.arange(-m // 2, m // 2 + 1)   # shape (m+1,) = (257,)

# Gaussian test function:  phi(x) = exp(-x^2)
# Chosen because it is smooth, compactly supported in practice, and its
# Fourier transform is also a Gaussian — easy to reason about analytically.
phi = np.exp(-x**2)    # shape (257,)

# ===========================================================================
# DtN MATRIX APPROXIMATION  (paper §4.1)
# ===========================================================================

# Build the (257 x 257) DtN matrix.  This encodes the singular integral
#   phi_z(x0) = (1/pi) * PV integral of [phi(x0)-phi(x)] / (x-x0)^2 dx
# as a matrix-vector product.  Building it is the expensive step (~seconds
# for m=256).  Once built, applying it is just N @ phi.
N    = build_dtn_matrix(m, dx)   # shape (257, 257)
phiz = N @ phi                   # matrix-vector product -> phi_z approximation (257,)

# ===========================================================================
# SPECTRAL REFERENCE SOLUTION  (paper §4.2)
# ===========================================================================

# The FFT method requires m (not m+1) equally-spaced points on a periodic
# domain — the last point is NOT repeated.  Drop the endpoint to match.
# MATLAB equivalent:  phiSpec = phi(1:end-1)
phi_periodic  = phi[:-1]                          # shape (256,)
phiz_spectral = spectral_dtn(phi_periodic, dx)    # shape (256,)

# ===========================================================================
# VALIDATION ERROR  (reproduces Table 4.1 of the paper)
# ===========================================================================

# Compare on the m=256 points where both methods have values.
# phiz has 257 points; phiz_spectral has 256 points.
# Use the first 256 points of phiz (dropping the last endpoint).
err = np.mean(np.abs(phiz[:-1] - phiz_spectral))
print(f"  m = {m:4d}  |  dx = {dx}  |  mean |error| = {err:.6e}")
print(f"  Expected from Table 4.1: ~1.41e-03")

# ===========================================================================
# FIGURE 1: Input function  (paper Figure 4.2a)
# ===========================================================================

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.axhline(0, color="k", linewidth=0.8)
ax1.plot(x[:-1], phi_periodic, linewidth=3, color="steelblue")
ax1.set_xlabel("x")
ax1.set_ylabel(r"$\varphi$")
ax1.set_title(
    r"Figure 4.2a — Gaussian test function $\varphi(x) = e^{-x^2}$"
    "\n(input to the DtN operator)"
)
ax1.grid(True)
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, "python_dtn_input_gaussian.png"), dpi=150)
plt.close(fig1)

# ===========================================================================
# FIGURE 2: Comparison — spectral vs matrix DtN  (paper Figure 4.2b)
# ===========================================================================

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.axhline(0, color="k", linewidth=0.8)

line_spec,  = ax2.plot(x[:-1], phiz_spectral,
                        linewidth=3, color="steelblue",
                        label="Spectral method  (reference)")
line_matrix, = ax2.plot(x, phiz,
                         "x", color="black", markersize=5, linewidth=1.5,
                         label="DtN matrix approximation")

ax2.legend()
ax2.set_xlabel("x")
ax2.set_ylabel(r"$\varphi_z$")
ax2.set_title(
    r"Figure 4.2b — $\varphi_z$: spectral method vs DtN matrix"
    f"\n(m = {m}, dx = {dx},  mean |error| = {err:.2e})"
)
ax2.grid(True)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, "python_dtn_comparison.png"), dpi=150)
plt.close(fig2)

# ===========================================================================
# FIGURE 3: Row structure of N  (paper Figure 4.3)
# ===========================================================================
# Because the DtN operator is translation-invariant, each row of N is a
# shifted copy of the same weight pattern.  Plotting several rows confirms
# this Toeplitz structure: the curves look identical but slid left/right.

row_indices = [
    10,
    int(np.ceil((m + 1) / 4)),
    int(np.ceil((m + 1) / 2)),
    m - 10,          # near the right boundary (= end-10 in MATLAB)
]

fig3, ax3 = plt.subplots(figsize=(8, 4))
for idx in row_indices:
    ax3.plot(N[idx, :], linewidth=1.8, label=f"row {idx}")

ax3.set_xlabel("Column index j")
ax3.set_ylabel("N[i, j]")
ax3.set_title(
    "Figure 4.3 — Selected rows of the DtN matrix N\n"
    "(each row is a shifted copy of the same weight template)"
)
ax3.legend()
ax3.grid(True)
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, "python_dtn_matrix_rows.png"), dpi=150)
plt.close(fig3)

print(f"\n  Figures saved to:  {os.path.abspath(output_dir)}")
