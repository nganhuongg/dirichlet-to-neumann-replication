# Cylinder Impact Replication

Python replication of the numerical experiments in *Impact of an Infinite Cylinder*, focused on the Dirichlet-to-Neumann (DtN) operator for a free surface and the linearized time-dependent impact simulation driven by a Gaussian pressure field.

The repository contains:

- a Python implementation of the DtN matrix construction
- an FFT-based spectral DtN reference for validation
- a Crank-Nicholson free-surface simulation that mirrors the MATLAB appendix code
- the original MATLAB scripts used as the reference source
- generated figures in `results/figures` for quick visual comparison

## What This Project Reproduces

The underlying model treats the fluid as inviscid, incompressible, and irrotational in the lower half-plane. The main numerical ingredients are:

- a dense DtN matrix that maps surface potential `phi(x, 0)` to the normal derivative `phi_z(x, 0)`
- a spectral DtN operator used as a high-accuracy benchmark
- a block Crank-Nicholson time integrator for the coupled free-surface variables `eta` and `phi`
- a Gaussian pressure loading of the form
  `p_s(x, t) = exp(-x^2) * (0.5 - 0.5 cos(2 pi t))`

The Python implementation is written to closely follow the reconstructed MATLAB appendix listings in [`original/`](./original).

## Repository Layout

```text
cylinderimpact_replication/
|-- src/
|   |-- dtn.py            # DtN matrix construction and spectral reference helper
|   |-- spectral.py       # FFT-based DtN operator
|   `-- simulation.py     # Time-dependent free-surface simulation
|-- experiments/
|   |-- gaussian_test.py      # DtN vs spectral validation on a Gaussian
|   `-- convergence_test.py   # Refinement and exact sinusoidal tests
|-- original/
|   |-- DtNmatrix.m
|   `-- simulation.m
|-- results/
|   `-- figures/          # Saved MATLAB/Python output figures
|-- requirements.txt
`-- README.md
```

## Installation

Create a virtual environment if you want an isolated setup, then install the requirements:

```bash
pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `matplotlib`
- `scipy`
- `jupyter`

## How To Run

Run commands from the repository root.

### 1. Validate the DtN matrix against the spectral method

```bash
python experiments/gaussian_test.py
```

This script:

- applies the DtN matrix to a Gaussian test function
- compares the result with the FFT-based spectral operator
- saves validation figures into `results/figures`

### 2. Run the convergence study

```bash
python experiments/convergence_test.py
```

This script:

- measures Gaussian-test error over multiple grid sizes
- checks a sinusoidal case with a known exact DtN response
- reports structural diagnostics such as symmetry and row-sum behavior

### 3. Run the free-surface simulation

```bash
python -m src.simulation
```

This script:

- constructs the DtN matrix and second-difference operator
- advances the coupled system with Crank-Nicholson time stepping
- saves the Gaussian pressure surface and several free-surface snapshots

## Key Results

### DtN validation: matrix vs spectral method

The core validation result is that the matrix approximation closely follows the spectral DtN reference for a Gaussian input.

![DtN comparison](results/figures/matlab_dtn_comparison.png)

This is the most important verification figure in the repository because it shows the numerical DtN construction reproducing the expected surface-normal derivative.

### DtN matrix structure

The DtN operator is assembled as a symmetric Toeplitz-like matrix whose rows are shifted copies of the same weight pattern.

![DtN matrix structure](results/figures/matlab_dtn_matrix_structure.png)

This plot helps confirm the translational invariance built into the discretization.

### Pressure forcing used in the simulation

The simulation is driven by a localized Gaussian pressure distribution in space with compact time variation over the forcing interval.

![Pressure surface](results/figures/python_simulation_pressure_surface.png)

### Representative free-surface response

One representative snapshot from the Python simulation is shown below. Additional times are available in `results/figures`.

![Free-surface snapshot](results/figures/python_simulation_snapshot_t95.png)

## Source Notes

- [`src/dtn.py`](./src/dtn.py) contains the main DtN matrix construction and an inline script that reproduces the paper-style validation figures.
- [`src/spectral.py`](./src/spectral.py) provides the FFT-based DtN operator for periodic data.
- [`src/simulation.py`](./src/simulation.py) implements the time-dependent free-surface model using the same nondimensional groups and forcing structure as the MATLAB reconstruction.
- [`original/DtNmatrix.m`](./original/DtNmatrix.m) and [`original/simulation.m`](./original/simulation.m) preserve the MATLAB reference implementation used for comparison.

## Current Outputs In `results/figures`

Saved figures currently include:

- MATLAB DtN validation and matrix-structure plots
- MATLAB simulation pressure surface and time snapshots
- Python simulation pressure surface and time snapshots

If you rerun the experiment scripts, the figures in `results/figures` will be updated.

## Possible Next Steps

- add a short section summarizing expected numerical error values from the paper
- include side-by-side Python vs MATLAB comparisons for the simulation snapshots
- add automated tests that verify key error tolerances in CI
