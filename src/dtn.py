"""
Numerical Dirichlet-to-Neumann (DtN) operator
==========================================================
Python translation of DtNmatrix.m  

PHYSICAL BACKGROUND  
--------------------------------------
We model a 2-D ideal, irrotational, incompressible fluid occupying the
lower half-plane z ≤ 0.  The velocity potential φ satisfies Laplace's
equation inside the fluid:

    Δφ = 0,   z ≤ 0                                        (eq. 2.28a)

On the free surface z = 0 the linearised governing equations are:

    η_t  =  φ_z                                             (eq. 2.28b)
    φ_t  =  -(1/Fr) η  +  (1/We) η_xx  -  p_s             (eq. 2.28c)

Both equations live only on the 1-D surface z = 0, but eq. (2.28b)
requires φ_z — a quantity defined *inside* the domain.  To stay on the
surface we need an operator that maps the Dirichlet data φ(x,0) to the
Neumann data φ_z(x,0).  That operator is the Dirichlet-to-Neumann (DtN)
operator N, so φ_z = N(φ).

DERIVATION OF THE DtN INTEGRAL FORMULA 
------------------------------------------------------
Using Green's Second Identity on a shrinking half-annulus (Fig. 3.1) and
taking the limits r₁→0 (inner radius) and r₂→∞ (outer radius), the paper
derives (eq. 3.28):

    φ_z(x₀, 0) = (1/π) · lim_{ε→0} ∫_{ℝ \ (x₀-ε, x₀+ε)}
                              [φ(x₀) - φ(x)] / (x - x₀)²  dx

This is a *singular* Cauchy-principal-value integral: the integrand blows
up as x → x₀, but the numerator φ(x₀)-φ(x) → 0 at the same rate, so the
limit is finite (convergence proved in §3.1.1 via Taylor expansion).

Because the operator is linear in φ, on a uniform grid with m+1 points
spaced dx apart it can be represented as a dense matrix N of size
(m+1)×(m+1):

    φ_z  ≈  N · φ                                          (matrix–vector product)

TRANSLATIONAL INVARIANCE AND THE TOEPLITZ STRUCTURE
-----------------------------------------------------
The weight that point x contributes to φ_z(x₀) depends only on the
*distance* |x - x₀|, not on the absolute position.  Therefore every row
of N is just a shifted copy of the same weight vector (the paper calls it
"line"); see Figure 4.3.  Exploiting this, we only need to compute one
template row and then assemble N by shifting it.

NUMERICAL APPROXIMATION  
------------------------------------------------
The singular integral is split into three parts, with cut-off M = 2·dx:

    φ_z(x₀) = L  +  P₁  +  P₂                            (eq. 4.2)

  P₁  Exact integration of the bare 1/(x-x₀)² kernel over |x-x₀| > M
      (eq. 4.8).  The result depends only on the value at x₀:

          P₁ = φ(x₀) / (π · dx)                           (since M = 2·dx)

  P₂  Numerical integration of -φ(x)/(x-x₀)² over |x-x₀| > M using
      Simpson's rule on panels of width 2·dx (§4.1.2, eq. 4.32).

  L   Near-singularity integral over |x-x₀| < M.  φ is approximated
      by a 4th-order polynomial fitted to the 5 nearest nodes, then
      integrated analytically (§4.1.3, eq. 4.39–4.45).

FILE STRUCTURE
--------------
  build_line(m, dx)        → template weight vector  (length m+1)
  build_dtn_matrix(m, dx)  → full (m+1)×(m+1) DtN matrix N
  dtn_spectral(phi, dx)    → exact φ_z via FFT (for validation)
  __main__                 → reproduces Figures 4.2, 4.3 and Table 4.1
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1.  Build the template weight vector  ("line" in the MATLAB code)
# ---------------------------------------------------------------------------

def build_line(m: int, dx: float) -> np.ndarray:
    """
    Compute the weight vector `line` of length m+1.

    `line[k]` (after the final normalisation) is the DtN weight at grid
    distance k·dx from the evaluation point x₀:

        φ_z(x₀) ≈  Σ_{k=0}^{m}  line[k] · φ(x₀ + k·dx)
                  + Σ_{k=1}^{m}  line[k] · φ(x₀ - k·dx)   ← by symmetry

    (The matrix assembly step mirrors this vector to both sides.)

    Before normalisation the values stored are  line[k] * (π·dx)  so that
    the global division by (π·dx) at the end gives the correct weights.

    Parameters
    ----------
    m  : int    number of grid *intervals*; grid has m+1 points.
    dx : float  uniform grid spacing.

    Returns
    -------
    line : ndarray, shape (m+1,)
        Normalised DtN weights.
    """

    line = np.zeros(m + 1)   # all zeros to start; accumulate contributions below

    # ------------------------------------------------------------------
    # (A)  NEAR-SINGULARITY CONTRIBUTION: L + P₁  (§4.1.1 and §4.1.3)
    # ------------------------------------------------------------------
    #
    # L approximates the integral over |x-x₀| < M = 2·dx using a
    # 4th-order polynomial Q fitted to the 5 nearby nodes
    #   m = -2, -1, 0, 1, 2  (i.e. x₀ ± dx, x₀ ± 2dx).
    # After integrating analytically only the even-power coefficients b₂
    # and b₄ survive (odd powers are anti-symmetric and cancel).
    # The result (eq. 4.45) gives weights on the 5 nodes:
    #
    #   node at x₀       (index 0):  +22/6  =  11/3
    #   nodes at x₀±dx   (index 1):  -(32/3)/6  = -16/9
    #   nodes at x₀±2dx  (index 2):  -(1/3)/6   = -1/18
    #
    # P₁ (eq. 4.8) adds 1 to the centre node (index 0):
    #   P₁ = φ(x₀)/(π·dx)  → un-normalised coefficient = +1
    #
    # Combined (un-normalised, i.e. multiplied by π·dx):
    line[0] = 1.0 + 11.0/3.0    # center:  P₁  +  L centre   (eq. 4.8 + 4.45)
    line[1] = -16.0/9.0          # ±1-neighbour: L term        (eq. 4.45)
    line[2] = -1.0/18.0          # ±2-neighbour: L term        (eq. 4.45)

    # ------------------------------------------------------------------
    # (B)  FAR-FIELD CONTRIBUTION: P₂  (§4.1.2, eq. 4.32)
    # ------------------------------------------------------------------
    #
    # P₂ covers the region |x-x₀| > M = 2·dx.  The positive half [M,∞)
    # is handled by summing over Simpson panels of width 2·dx each.
    # Each panel is centred at n·dx (with n running over ODD integers ≥ 3)
    # so that the panels tile [2·dx, ∞) without gaps:
    #
    #   n = 3  →  panel [2dx, 4dx]   (nodes at 2dx, 3dx, 4dx)
    #   n = 5  →  panel [4dx, 6dx]   (nodes at 4dx, 5dx, 6dx)
    #   ...
    #
    # Adjacent panels SHARE their endpoint (e.g. 4dx belongs to both
    # n=3 and n=5), so the weights from each panel are ADDED (+=).
    #
    # Fitting a 2nd-order polynomial to the three nodes in each panel
    # and integrating analytically gives (eq. 4.32):
    #
    #   contribution to node at (n-1)·dx:  an = f_{-1}(n)
    #   contribution to node at  n   ·dx:  bn = f_0(n)
    #   contribution to node at (n+1)·dx:  cn = f_1(n)
    #
    # where  (eq. 4.32 and Appendix A code):
    #   an = f_{-1}(n) = -n/(n-1) + (n+½)·ln((n+1)/(n-1)) - 1
    #   bn = f_0(n)    = -2n·ln((n+1)/(n-1)) + 4
    #   cn = f_1(n)    = -n/(n+1) + (n-½)·ln((n+1)/(n-1)) - 1
    #
    # Index mapping (MATLAB 1-based → Python 0-based):
    #   MATLAB line(nn)   → Python line[nn-1]   (distance (nn-1)·dx)
    #   MATLAB line(nn+1) → Python line[nn]     (distance  nn   ·dx)
    #   MATLAB line(nn+2) → Python line[nn+1]   (distance (nn+1)·dx)

    for nn in range(3, m + 1, 2):           # nn = 3, 5, 7, …, m-1 (odd)
        log_ratio = np.log((nn + 1.0) / (nn - 1.0))   # ln((n+1)/(n-1))

        an = -nn / (nn - 1.0) + (nn + 0.5) * log_ratio - 1.0   # f_{-1}(n)
        bn = -2.0 * nn * log_ratio + 4.0                         # f_0(n)
        cn = -nn / (nn + 1.0) + (nn - 0.5) * log_ratio - 1.0   # f_1(n)

        # Accumulate into the template vector.
        # Note: line[2] (nn=3, left endpoint) already contains -1/18 from
        # the L term; the two contributions are physically distinct and
        # are simply summed here.
        line[nn - 1] += an    # left  endpoint of this panel
        line[nn]     += bn    # centre of this panel
        line[nn + 1] += cn    # right endpoint of this panel

    # ------------------------------------------------------------------
    # (C)  GLOBAL NORMALISATION  (prefactor 1/(π·dx) in eq. 4.2)
    # ------------------------------------------------------------------
    # All weights were accumulated as (weight × π·dx).  Dividing by π·dx
    # recovers the true DtN weights.
    line /= (np.pi * dx)

    return line


# ---------------------------------------------------------------------------
# 2.  Assemble the full DtN matrix N
# ---------------------------------------------------------------------------

def build_dtn_matrix(m: int, dx: float) -> np.ndarray:
    """
    Assemble the (m+1)×(m+1) Dirichlet-to-Neumann matrix N.

    By translational invariance the DtN weight from grid point j to
    evaluation point i depends only on |i-j|, so

        N[i, j]  ≈  line[ |i - j| ]

    which gives a symmetric Toeplitz structure.  Every row is a shifted
    copy of the template vector `line` (illustrated in Figure 4.3 of the
    paper).

    Row i is assembled as:

        N[i, :]  =  [ line[i],   line[i-1], …, line[1],   ← left  side
                      line[0],   line[1],   …, line[m-i] ] ← right side

    i.e. we concatenate:
        left_part  = line[1 : i+1] reversed    (i elements)
        right_part = line[0 : m-i+1]           (m-i+1 elements)
    Total length = i + (m-i+1) = m+1  ✓

    MATLAB equivalent (1-based indexing, where ii = i+1):
        N(ii,:) = [fliplr(line(2:ii)),  line(1:m-ii+2)]

    Parameters
    ----------
    m  : int   – number of grid intervals (m+1 grid points).
    dx : float – grid spacing.

    Returns
    -------
    N : ndarray, shape (m+1, m+1)
    """
    line = build_line(m, dx)

    N = np.zeros((m + 1, m + 1))

    for i in range(m + 1):
        # Left half: the i nodes to the LEFT of evaluation point i.
        # line[1] is the nearest left neighbour, line[i] is the farthest.
        # We reverse so the row reads "farthest … nearest" from the left.
        left_part = line[1 : i + 1][::-1]       # shape (i,)

        # Right half: evaluation point itself (line[0]) and all nodes to
        # the RIGHT, up to the boundary.
        right_part = line[0 : m - i + 1]        # shape (m-i+1,)

        N[i, :] = np.concatenate([left_part, right_part])

    return N


# ---------------------------------------------------------------------------
# 3.  Spectral (FFT-based) reference implementation  (§4.2)
# ---------------------------------------------------------------------------

def dtn_spectral(phi: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute φ_z via the Fourier / spectral method (paper §4.2).

    In Fourier space, the DtN operator for the lower half-plane z ≤ 0 is
    simply multiplication by |k|  (the absolute wavenumber):

        φ̂_z(k)  =  |k| · φ̂(k)

    This follows from the fact that the general solution to Laplace's
    equation in z ≤ 0 that decays as z → -∞ is
        φ(x,z) = ∫ φ̂(k) e^{ikx + |k|z} dk,
    so differentiation with respect to z at z=0 multiplies by |k|.

    Steps:
        1. FFT(φ)           transform to frequency domain
        2. multiply by |k|  apply DtN in Fourier space
        3. IFFT             return to physical space
        4. take real part   discard tiny numerical imaginary noise

    This gives the *exact* answer for a periodic domain and is used as a
    benchmark against the matrix approximation (Table 4.1, Fig. 4.2b).

    Parameters
    ----------
    phi : ndarray, length m  (an EVEN number; no endpoint repetition —
          the last grid point is NOT the same as the first).
    dx  : float grid spacing.

    Returns
    -------
    phiz_spec : ndarray, length m  (real-valued).
    """
    m = len(phi)                               # number of sample points

    # Wavenumber spacing in Fourier space
    dkx = 2.0 * np.pi / (m * dx)

    # Build the frequency vector in FFT order:
    #   [0, 1, 2, …, m/2,  -m/2+1, …, -1] × dkx
    # (This is the standard NumPy FFT ordering.)
    pos_freqs = np.arange(0, m // 2 + 1)
    neg_freqs = np.arange(-m // 2 + 1, 0)
    kx = np.concatenate([pos_freqs, neg_freqs]) * dkx

    # Apply DtN in Fourier space and transform back
    phiz_spec = np.real(np.fft.ifft(np.abs(kx) * np.fft.fft(phi)))
    return phiz_spec


# ---------------------------------------------------------------------------
# 4.  Main script – reproduces Figures 4.2a, 4.2b, 4.3 and Table 4.1
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Parameters (matching Appendix A / DtNmatrix.m) ----
    dx = 0.1
    m  = 2**8    # 256 grid intervals → 257 grid points

    # ---- Spatial grid ----
    # x runs from -(m/2)·dx  to  +(m/2)·dx  in steps of dx.
    # MATLAB: x = dx * (-m/2 : m/2)'   (column vector)
    x = dx * np.arange(-m // 2, m // 2 + 1)   # shape (m+1,) = (257,)

    # ---- Test function: Gaussian  φ(x) = exp(-x²)  ----
    phi = np.exp(-x**2)      # shape (257,)

    # ---- DtN matrix approximation ----
    N    = build_dtn_matrix(m, dx)          # (257 × 257) matrix
    phiz = N @ phi                          # matrix–vector product → φ_z approximation

    # ---- Spectral reference ----
    # The FFT method requires m (not m+1) equi-spaced points on a
    # periodic domain (no repeated endpoint), so drop the last element.
    # MATLAB: phiSpec = phi(1:end-1)
    phiSpec  = phi[:-1]                     # shape (256,)
    phizSpec = dtn_spectral(phiSpec, dx)    # shape (256,)  – exact reference

    # ---- Figure 1: Gaussian input  (Figure 4.2a) ----
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.axhline(0, color='k')
    ax1.plot(x[:-1], phiSpec, linewidth=4)
    ax1.set_xlabel("x")
    ax1.set_ylabel("φ")
    ax1.set_title("Figure 4.2a Gaussian input  φ(x) = exp(−x²)")
    ax1.grid(True)
    fig1.tight_layout()

    # ---- Figure 2: DtN comparison  (Figure 4.2b) ----
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.axhline(0, color='k')
    a, = ax2.plot(x[:-1], phizSpec, linewidth=4, label="Spectral Method")
    b, = ax2.plot(x,      phiz,     'xk', markersize=6, linewidth=2,
                  label="DtN Approximation")
    ax2.legend()
    ax2.set_xlabel("x")
    ax2.set_ylabel("φ_z")
    ax2.set_title("Figure 4.2b φ_z: spectral vs DtN matrix approximation")
    ax2.grid(True)
    fig2.tight_layout()

    # ---- Figure 3: Row structure of N  (Figure 4.3) ----
    # Different rows should look like shifted copies of the same pattern,
    # confirming the translational invariance of the DtN operator.
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(N[10, :],                         linewidth=2, label="row 10")
    ax3.plot(N[int(np.ceil((m+1)/4)), :],      linewidth=2,
             label=f"row {int(np.ceil((m+1)/4))}")
    ax3.plot(N[int(np.ceil((m+1)/2)), :],      linewidth=2,
             label=f"row {int(np.ceil((m+1)/2))}")
    ax3.plot(N[-11, :],                        linewidth=2, label="row end-10")
    ax3.set_xlabel("Column index")
    ax3.set_ylabel("Element value")
    ax3.set_title("Figure 4.3 – Various rows of N (shifted weight pattern)")
    ax3.legend()
    ax3.grid(True)
    fig3.tight_layout()

    # ---- Validation error  (Table 4.1) ----
    # Compare phiz (matrix, m+1 points) with phizSpec (spectral, m points)
    # on the m shared points.
    err = np.mean(np.abs(phiz[:-1] - phizSpec))
    print(f"m = {m:4d}  |  average absolute error = {err:.15f}")
    # Expected for m=256:  ~0.001412258158996  (Table 4.1 of the paper)

    plt.show()
