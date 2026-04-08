"""
Crank-Nicholson simulation of a pressured free surface.

This module follows Chapter 5 and Appendix A of
"Impact of an Infinite Cylinder" and mirrors the MATLAB
Simulation.m listing included in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from src.dtn import build_dtn_matrix
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dtn import build_dtn_matrix


@dataclass(frozen=True)
class SimulationParameters:
    lx_cgs: float = 160.0
    l_cgs: float = 1.0
    m: int = 2**10
    tau_cgs: float = 0.5
    tf_cgs: float = 0.5
    nt: int = 150
    rho_cgs: float = 1.0
    sigma_cgs: float = 70.0
    g_cgs: float = 980.0


def nondimensional_groups(params: SimulationParameters) -> tuple[float, float]:
    """Return the paper's Froude and Weber numbers."""
    length_unit = params.l_cgs
    time_unit = params.tau_cgs
    fr = length_unit / (time_unit**2 * params.g_cgs)
    we = params.rho_cgs * length_unit**3 / (time_unit**2 * params.sigma_cgs)
    return fr, we


def simulation_grid(params: SimulationParameters) -> tuple[float, float, np.ndarray, float, np.ndarray]:
    """Build the nondimensional spatial and temporal grids."""
    length_unit = params.l_cgs
    time_unit = params.tau_cgs

    lx = params.lx_cgs / length_unit
    tf = params.tf_cgs / time_unit

    dx = lx / params.m
    x_vec = np.arange(-lx / 2.0, lx / 2.0 + dx, dx)

    dt = tf / params.nt
    t_vec = np.arange(0.0, tf + dt, dt)

    return lx, tf, x_vec, dt, t_vec


def second_difference_matrix(m: int) -> np.ndarray:
    """
    Build the centered second-difference matrix used in Appendix A.

    This intentionally matches the paper's MATLAB listing.
    """
    dxx = np.diag(-2.0 * np.ones(m + 1))
    dxx += np.diag(np.ones(m), k=-1)
    dxx += np.diag(np.ones(m), k=1)
    return dxx


def gaussian_pressure(x_vec: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    """
    Pressure from equation (5.13) / Appendix A:

        p_s(x, t) = exp(-x^2) * (0.5 - 0.5 cos(2 pi t))
    """
    spatial = np.exp(-(x_vec**2))[:, None]
    temporal = (0.5 - 0.5 * np.cos(2.0 * np.pi * t_vec))[None, :]
    return spatial * temporal


def crank_nicholson_blocks(
    n_matrix: np.ndarray,
    dxx: np.ndarray,
    dt: float,
    fr: float,
    we: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the block matrices A and B from equation (5.8)."""
    size = n_matrix.shape[0]
    eye = np.eye(size)

    a_mat = np.block(
        [
            [eye, -dt * n_matrix / 2.0],
            [dt * eye / (2.0 * fr) - dt * dxx / (2.0 * we), eye],
        ]
    )

    b_mat = np.block(
        [
            [eye, dt * n_matrix / 2.0],
            [-dt * eye / (2.0 * fr) + dt * dxx / (2.0 * we), eye],
        ]
    )

    return a_mat, b_mat


def run_simulation(
    params: SimulationParameters | None = None,
) -> dict[str, np.ndarray | float]:
    """Run the paper's free-surface simulation and return the computed fields."""
    params = params or SimulationParameters()

    fr, we = nondimensional_groups(params)
    lx, tf, x_vec, dt, t_vec = simulation_grid(params)
    n_matrix = build_dtn_matrix(params.m, lx / params.m)
    dxx = second_difference_matrix(params.m)
    a_mat, b_mat = crank_nicholson_blocks(n_matrix, dxx, dt, fr, we)
    ps = gaussian_pressure(x_vec, t_vec)

    vec_old = np.zeros(2 * (params.m + 1))
    eta_history = np.zeros((params.m + 1, params.nt + 1))
    phi_history = np.zeros((params.m + 1, params.nt + 1))

    eta_history[:, 0] = vec_old[: params.m + 1]
    phi_history[:, 0] = vec_old[params.m + 1 :]

    zero_block = np.zeros(params.m + 1)

    for kk in range(params.nt):
        forcing = np.concatenate([zero_block, -dt * (ps[:, kk] + ps[:, kk + 1])])
        rhs = b_mat @ vec_old + forcing
        vec_new = np.linalg.solve(a_mat, rhs)

        eta_history[:, kk + 1] = vec_new[: params.m + 1]
        phi_history[:, kk + 1] = vec_new[params.m + 1 :]
        vec_old = vec_new

    return {
        "Fr": fr,
        "We": we,
        "Lx": lx,
        "Tf": tf,
        "dt": dt,
        "x": x_vec,
        "t": t_vec,
        "pressure": ps,
        "eta": eta_history,
        "phi": phi_history,
    }


def save_outputs(results: dict[str, np.ndarray | float], output_dir: Path) -> None:
    """Save the pressure surface and the four paper snapshots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x_vec = np.asarray(results["x"])
    t_vec = np.asarray(results["t"])
    pressure = np.asarray(results["pressure"])
    eta = np.asarray(results["eta"])
    lx = float(results["Lx"])

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    tt, xx = np.meshgrid(t_vec, x_vec)

    # Match MATLAB's smooth surf rendering more closely by disabling the
    # default downsampling in Matplotlib's 3D surface plot.
    ax.plot_surface(
        tt,
        xx,
        pressure,
        cmap="viridis",
        linewidth=0,
        antialiased=False,
        rcount=pressure.shape[0],
        ccount=pressure.shape[1],
    )
    ax.set_xlabel("t / tau")
    ax.set_ylabel("x / l")
    ax.set_zlabel("p_s")
    ax.set_title("Gaussian pressure distribution")
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    fig.savefig(output_dir / "python_simulation_pressure_surface.png", dpi=180)
    plt.close(fig)

    snapshots = {
        15: "python_simulation_snapshot_t15.png",
        55: "python_simulation_snapshot_t55.png",
        95: "python_simulation_snapshot_t95.png",
        135: "python_simulation_snapshot_t135.png",
    }

    for step, filename in snapshots.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_vec, eta[:, step], linewidth=2)
        ax.set_xlim([-lx / 2.0, lx / 2.0])
        ax.set_ylim([-0.02, 0.02])
        ax.grid(True)
        ax.set_xlabel("x / l")
        ax.set_ylabel("eta / l")
        ax.set_title(f"t / tau = {t_vec[step]:.4f}")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    results = run_simulation()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results" / "figures"
    save_outputs(results, output_dir)


if __name__ == "__main__":
    main()
