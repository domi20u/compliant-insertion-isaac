"""Run PI² for many random obstacle configurations and save training data.

Replaces ``obst_avoid_box_acc_data_gen.py``. Usage:

    python scripts/generate_data.py --output-dir results/run0 --n-runs 10

All knobs live in ``configs/configs.py::DataGenConfig``; they can be overridden
from the command line via ``tyro``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make sibling packages importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import tyro

from configs.configs import DataGenConfig
from core.dmp_wrapper import DMPWrapper
from core.pi2_plotting import PI2OptimizerPlotting
from core.visualization import plot_pi2_run


def main(cfg: DataGenConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    print(f"[generate_data] device: {device}")
    print(f"[generate_data] output_dir: {cfg.output_dir}")
    print(f"[generate_data] n_runs: {cfg.n_runs}, n_samples/iter: {cfg.pi2.n_samples}")

    # ---- Build DMP and imitate the demonstration once
    dmp = DMPWrapper(cfg.dmp, device=device)
    dmp.imitate_path(cfg.dmp.trajectory_file)
    print(f"[generate_data] imitated demo, learned_L = {dmp.learned_L:.4f}, bounds = {cfg.cost.bounds}, max_val = {cfg.cost.max_val}, goal_penalty = {cfg.cost.goal_penalty}, basis_count = {cfg.dmp.n_bfs}")

    # ---- Build optimizer (reused across runs)
    opt = PI2OptimizerPlotting(
        dmp, cfg.pi2, cfg.cost,
        plot_every=5,            # snapshot every 5 iters
        plot_dir="results/run0",
        plot_name="obstacle_3",
        live_plot=True,          # open the window
        live_pause=0.1,          # ~10 updates/sec feels natural
        show_samples=True,
        n_show_samples=8,
    )


    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    for run_id in range(cfg.n_runs):
        # Sample two random y-locations for the obstacles (sorted)
        z_heights = np.sort(cfg.z_height_low + cfg.z_height_range
                            * np.random.rand(2))
        z_heights = np.array([0.1, 0.15-0.00075])  # fixed heights for testing --- IGNORE ---
        z_t = torch.tensor(z_heights, dtype=dmp.dtype, device=dmp.device)

        print(f"[generate_data] run {run_id+1}/{cfg.n_runs} | "
              f"z_heights={z_heights.round(4)}")

        #print(z_t)

        result = opt.optimize(z_t)

        run_dir = out_dir / f"train{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save per-run training data (matches old format for compatibility)
        np.savez(
            run_dir / f"train_{run_id}.npz",
            training_data=result.train_data,
            labels=result.label_z,
            n_episodes=result.n_iter,
            costs=result.avg_costs,
            t_pi2=result.duration,
            z_heights=z_heights,
            max_val=cfg.cost.max_val,
            y_locations=result.y_locations,
            final_x_traj=result.final_x_track,
        )

        # Final-state plots
        plot_pi2_run(
            final_traj=result.final_x_track,
            weights=result.train_data[-1],
            costs=result.avg_costs,
            z_targets=z_heights,
            max_val=cfg.cost.max_val,
            out_dir=out_dir,
            run_id=run_id,
        )

        elapsed = (time.time() - t_total) / 60
        print(f"[generate_data] run {run_id} done in {result.duration:.2f}s "
              f"(iters={result.n_iter}, finished={result.finished}); "
              f"elapsed total: {elapsed:.1f} min")


if __name__ == "__main__":
    cfg = tyro.cli(DataGenConfig)
    main(cfg)
