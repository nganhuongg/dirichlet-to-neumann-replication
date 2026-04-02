"""
spectral.py  –  Spectral (FFT-based) Dirichlet-to-Neumann operator
===================================================================

PURPOSE
-------
This module implements the *spectral* (Fourier) version of the
Dirichlet-to-Neumann (DtN) operator.  It is used as a high-accuracy
reference solution to validate the matrix approximation built in dtn.py.

MATHEMATICAL BACKGROUND  (paper §4.2)
--------------------------------------
We want φ_z(x, 0), the normal derivative of the velocity potential on the
free surface, given only the surface values φ(x, 0).

On a *periodic* domain of length L = m · dx the velocity potential that
satisfies Laplace's equation (Δφ = 0) in the lower half-plane (z ≤ 0) and
decays as z → −∞ can be written as a Fourier series:

    φ(x, z) = Σ_k  φ̂(k) · e^{i k x + |k| z}

where the e^{|k|z} factor ensures decay for z < 0.  Differentiating with
respect to z and evaluating at z = 0 gives:

    φ_z(x, 0) = Σ_k  |k| · φ̂(k) · e^{i k x}

In other words, the DtN operator is *multiplication by |k| in Fourier space*:

    φ̂_z(k)  =  |k| · φ̂(k)                          (paper eq. 4.33)

Algorithm (three steps):
    1. FFT(φ)          — transform surface values to frequency domain
    2. Multiply by |k| — apply DtN in Fourier space (exact for periodic data)
    3. IFFT            — return to physical space

This is exact up to floating-point precision for band-limited periodic
functions, and provides an excellent reference for non-periodic functions
(e.g. a Gaussian) as long as the domain is large enough that boundary
effects are negligible.

RELATION TO dtn.py
------------------
dtn.py builds a dense matrix N so that N @ phi ≈ phi_z.  The matrix approach
works on a finite non-periodic domain and captures boundary behaviour.  The
spectral method here assumes periodicity and is used purely for validation.

USAGE
-----
    from src.spectral import spectral_dtn
    phi_z = spectral_dtn(phi, dx)   # phi has length m (no repeated endpoint)
"""

import numpy as np


def spectral_dtn(phi: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute the Dirichlet-to-Neumann derivative phi_z via the spectral method.

    Given the surface values phi(x, 0) on a uniform periodic grid, returns
    phi_z(x, 0) = N[phi] by multiplying the Fourier transform by |k| and
    transforming back.

    Parameters
    ----------
    phi : ndarray, shape (m,)
        Surface values phi at m equally-spaced points on a periodic domain.
        The last point must NOT be a repetition of the first (open interval).
        Typically obtained by dropping the repeated endpoint:
            phi = phi_full[:-1]   # if phi_full has m+1 points
    dx : float
        Grid spacing between adjacent sample points.

    Returns
    -------
    phi_z : ndarray, shape (m,)
        Approximate normal derivative phi_z(x, 0), real-valued.

    Notes
    -----
    The wavenumber vector is built in NumPy's standard FFT ordering:
        k = [0, 1, 2, ..., m/2,  -m/2+1, ..., -1] * dkx
    where dkx = 2*pi / (m * dx) is the fundamental frequency spacing.
    This matches the convention used in the original MATLAB code and
    in paper §4.2.

    The result of IFFT can have tiny imaginary parts (~1e-15) due to
    floating-point arithmetic; taking the real part discards this noise.
    """
    m = len(phi)   # number of sample points on the periodic domain

    # ------------------------------------------------------------------
    # Step 1: Wavenumber vector  (paper §4.2, eq. 4.33)
    # ------------------------------------------------------------------
    # dkx = 2*pi / L  where L = m * dx is the total domain length.
    # kx covers [0, ..., m/2, -m/2+1, ..., -1] * dkx in FFT order.
    dkx = 2.0 * np.pi / (m * dx)

    pos_freqs = np.arange(0, m // 2 + 1)      # 0, 1, ..., m/2
    neg_freqs = np.arange(-m // 2 + 1, 0)     # -m/2+1, ..., -1
    kx = np.concatenate([pos_freqs, neg_freqs]) * dkx

    # ------------------------------------------------------------------
    # Step 2-3: Apply DtN in Fourier space and invert
    # ------------------------------------------------------------------
    # FFT -> multiply by |k| -> IFFT
    # The real() call removes floating-point imaginary noise (~1e-15).
    phi_z = np.real(np.fft.ifft(np.abs(kx) * np.fft.fft(phi)))

    return phi_z
