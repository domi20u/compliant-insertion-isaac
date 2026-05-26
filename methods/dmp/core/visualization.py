"""Plotting helpers for offline visualization of PI² runs and NN test results.

All functions here take results as numpy arrays and produce matplotlib figures.
None of them are called from inside the optimization loop — they're only used
by the top-level scripts after a run completes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_pi2_run(final_traj: np.ndarray,
                 weights: np.ndarray,
                 costs: np.ndarray,
                 z_targets: np.ndarray,
                 max_val: float,
                 out_dir: Path,
                 run_id: int) -> None:
    """Save the canonical PI² diagnostic figures for one optimization run.

    Args:
        final_traj: [T, 3] final policy rollout
        weights:    [num_basis, num_dof] final DMP weights
        costs:      [n_iter, 5] cost-trace per iteration
        z_targets:  [2] obstacle y-locations
        max_val:    target obstacle height
        out_dir:    directory to write figures into
        run_id:     index used in filenames
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / "graphics" / "img_train"
    pdf_dir = out_dir / "graphics" / "pdf_train"
    img_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Trajectory + obstacles
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    ax1.plot(final_traj[:, 1], final_traj[:, 2], 'b-', linewidth=1.5,
             label='final policy')
    for z in z_targets:
        ax1.plot(z, max_val, 'r*', markersize=12)
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.set_xlabel(r"$\mathbf{e}_1$ [m]")
    ax1.set_ylabel(r"$\mathbf{e}_2$ [m]")
    ax1.set_xlim([-0.02, 0.18])
    ax1.set_ylim([-0.01, max_val + 0.02])
    ax1.set_title(f"Final DMP trajectory (run {run_id})")
    ax1.legend()
    fig1.savefig(img_dir / f"img_{run_id}.png", dpi=120, format="png")
    fig1.savefig(pdf_dir / f"fig_{run_id}.pdf", format="pdf")
    plt.close(fig1)

    # Weights + cost trace
    fig2, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(12, 4))
    n_basis = weights.shape[0]
    width = 0.35
    idx = np.arange(n_basis)
    ax_w.bar(idx - width/2, weights[:, 1], width, label=r"$\theta_{e_1}$")
    ax_w.bar(idx + width/2, weights[:, 2], width, label=r"$\theta_{e_2}$")
    ax_w.set_xlabel("basis index")
    ax_w.set_ylabel("weight")
    ax_w.set_title("Final DMP weights")
    ax_w.legend()

    ax_c.plot(costs[:, 0], 'k-', label='total')
    ax_c.plot(costs[:, 1], 'r--', label='height (-)')
    ax_c.plot(costs[:, 2], 'g--', label='table')
    ax_c.plot(costs[:, 3], 'b--', label='bounds')
    ax_c.plot(costs[:, 4], 'm--', label='smooth')
    ax_c.set_xlabel("iteration")
    ax_c.set_ylabel("cost")
    ax_c.set_title("Cost trace")
    ax_c.legend()
    ax_c.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(img_dir / f"param_img_{run_id}.png", dpi=120)
    fig2.savefig(pdf_dir / f"param_fig_{run_id}.pdf", format="pdf")
    plt.close(fig2)


def plot_nn_test_old(trajectories: list[np.ndarray],
                 colors: list[np.ndarray],
                 obstacle_features: np.ndarray,
                 L_demo: float,
                 out_path: Path) -> None:
    """Plot all NN-policy rollouts for a single test batch.

    Args:
        trajectories: list of [T, 3] arrays
        colors:       list of [3] color arrays (red=fail, green=success)
        obstacle_features: [n, 4] each row [height, y0, y1, _]
        L_demo: demo length scaling
        out_path: output figure file
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 11))
    ax.grid(False)
    ax.plot([-0.0045, -0.0045], [0, 0.25], "r--", linewidth=1)
    ax.plot([0.1545, 0.1545], [0, 0.25], "r--", linewidth=1)
    ax.plot([-0.05, 0.25], [0, 0], "k", linewidth=0.5)
    ax.set_xlim([-0.02, 0.17])
    ax.set_ylim([-0.01, 0.32])
    ax.set_xlabel("X (m)")

    for traj, col, ins in zip(trajectories, colors, obstacle_features):
        ax.plot(traj[:, 1], traj[:, 2], color=col, linewidth=0.8)
        ax.plot(ins[1] * L_demo, ins[0] * L_demo, '*', color=col)
        ax.plot(ins[2] * L_demo, ins[0] * L_demo, '*', color=col)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def plot_nn_test(trajectories: list[np.ndarray],
                 colors: list[np.ndarray],
                 obstacle_features: np.ndarray,
                 L_demo: float,
                 out_path: Path,
                 max_val: float | None = None,
                 n_inputs: int = 3) -> None:
    """Plot all NN-policy rollouts for a single test batch.

    Args:
        trajectories: list of [T, 3] arrays
        colors:       list of [3] color arrays (red=fail, green=success)
        obstacle_features: [n, 4] each row [height, y0, y1, _]
        L_demo: demo length scaling for star positions (cost is normalised
            by L_demo, so star coords need to be un-normalised by it).
        out_path: output figure file
        max_val: trajectory-axis extent. Defaults to L_demo for backward
            compat with the original avoidance layout (L_demo=0.25).
        n_inputs: NN input dim. Stars per sample = ``n_inputs - 1``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if max_val is None:
        max_val = L_demo

    # Workspace walls live in absolute coordinates — independent of max_val.
    wall_x_lo, wall_x_hi = -0.0045, 0.1545

    # Y-axis (trajectory direction) scales with max_val. The original used
    # L_demo=0.25 and a y-range of [-0.01, 0.32], i.e. ~28% headroom above
    # max_val and a small dip below zero. Preserve those ratios.
    y_lo = -0.04 * max_val
    y_hi = 1.28 * max_val
    ground_y = 0.0

    fig, ax = plt.subplots(figsize=(8, 11))
    ax.grid(False)

    # Vertical workspace walls, extending to top of plot.
    ax.plot([wall_x_lo, wall_x_lo], [0, y_hi], "r--", linewidth=1)
    ax.plot([wall_x_hi, wall_x_hi], [0, y_hi], "r--", linewidth=1)
    # Ground line.
    ax.plot([-0.05, 0.25], [ground_y, ground_y], "k", linewidth=0.5)

    ax.set_xlim([-0.02, 0.17])
    ax.set_ylim([y_lo, y_hi])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    n_stars = max(0, n_inputs - 1)            # avoidance:2, insertion:1

    for traj, col, ins in zip(trajectories, colors, obstacle_features):
        ax.plot(traj[:, 1], traj[:, 2], color=col, linewidth=0.8)
        # ins layout: [height, y0, y1, _]. Stars are (y_k, height) for k=0..n_stars-1.
        for k in range(n_stars):
            ax.plot(ins[1 + k] * L_demo, ins[0] * L_demo, '*', color=col)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_imitation(t_demo: np.ndarray, demo: np.ndarray,
                   t_pred: np.ndarray, pred: np.ndarray,
                   out_path: Path) -> None:
    """Sanity plot for the ridge-regressed DMP fit."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_dof = demo.shape[1]
    fig, axes = plt.subplots(n_dof, 1, figsize=(8, 2.2 * n_dof), sharex=True)
    if n_dof == 1:
        axes = [axes]
    for d in range(n_dof):
        axes[d].plot(t_demo, demo[:, d], 'k--', label='demo')
        axes[d].plot(t_pred, pred[:, d], 'C0', label='dmp')
        axes[d].set_ylabel(f"dof {d}")
        axes[d].legend(loc='best')
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
