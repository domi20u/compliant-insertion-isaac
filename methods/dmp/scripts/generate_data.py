"""Run PI² for many random task configurations and save training data.

Usage:

    # Box-avoidance (default — backward compatible)
    python scripts/generate_data.py --output-dir results/avoidance_run0 --n-runs 10

    # Insertion
    python scripts/generate_data.py --scenario insertion \
        --output-dir results/insertion_run0 --n-runs 10

Per-scenario defaults (basis count, sigmas, cost) live in
``configs/configs.py``; everything else (n_runs, output_dir, seed) is shared
across scenarios and can be overridden from the CLI via ``tyro``.
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

from configs.configs import DataGenConfig, InsertionDataGenConfig
from core.dmp_wrapper import DMPWrapper
from core.pi2_optimizer import PI2Optimizer
from core.scenarios import get_scenario
from core.visualization import plot_pi2_run

from core.pi2_plotting import PI2OptimizerPlotting


# Map scenario name -> top-level DataGen config dataclass.
_DATAGEN_CFG = {
    "avoidance": DataGenConfig,
    "insertion": InsertionDataGenConfig,
}


def _pop_scenario_flag(argv: list[str]) -> str:
    """Extract --scenario before tyro sees the rest of argv.

    Supports both ``--scenario insertion`` and ``--scenario=insertion``.
    Defaults to ``avoidance`` for backward compatibility.
    """
    name = "avoidance"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--scenario" and i + 1 < len(argv):
            name = argv[i + 1]
            del argv[i:i + 2]
            continue
        if a.startswith("--scenario="):
            name = a.split("=", 1)[1]
            del argv[i]
            continue
        i += 1
    if name not in _DATAGEN_CFG:
        raise SystemExit(
            f"Unknown --scenario {name!r}. Choices: {sorted(_DATAGEN_CFG)}")
    return name


def main(cfg, scenario_name: str) -> None:
    scenario = get_scenario(scenario_name)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    print(f"[generate_data] scenario: {scenario_name}")
    print(f"[generate_data] device: {device}")
    print(f"[generate_data] output_dir: {cfg.output_dir}")
    print(f"[generate_data] n_runs: {cfg.n_runs}, n_samples/iter: {cfg.pi2.n_samples}")

    # ---- Build DMP and imitate the demonstration once
    dmp = DMPWrapper(cfg.dmp, device=device)
    dmp.imitate_path(cfg.dmp.trajectory_file)
    print(f"[generate_data] imitated demo, learned_L = {dmp.learned_L:.4f}, bounds = {cfg.cost.bounds}," 
          f" max_val = {cfg.cost.max_val}, goal_penalty = {cfg.cost.goal_penalty}, basis_count = {cfg.dmp.n_bfs}, "
          f"sigma_range = ({cfg.pi2.sigma_min}, {cfg.pi2.sigma_max}), sigma_steepness = {cfg.pi2.sigma_steepness}, basis_bandwidth_factor = {cfg.dmp.basis_bandwidth_factor}")

    L_demo = float(dmp.learned_L)

    # ---- Build optimizer (reused across runs), wired to the scenario's cost.
    pi2 = PI2Optimizer(dmp, cfg.pi2, cfg.cost, cost_fn=scenario.cost_fn)
    #pi2 = PI2OptimizerPlotting(
    #    dmp, cfg.pi2, cfg.cost,
    #    plot_every=5,            # snapshot every 5 iters
    #    plot_dir="results/run0",
    #    plot_name="obstacle_3",
    #    live_plot=True,          # open the window
    #    live_pause=0.1,          # ~10 updates/sec feels natural
    #    show_samples=True,
    #    n_show_samples=8,
    #)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Insertion needs goal_proximity to know where to pin the slot bottom;
    # avoidance ignores it. We pass it unconditionally — BoxAvoidScenario
    # accepts and discards via **_ignored, so callers don't have to branch.
    z_target_kwargs = {"goal_proximity": getattr(cfg.cost, "goal_proximity", 0.0)}

    t_total = time.time()
    for run_id in range(cfg.n_runs):
        # Sample the scenario's FREE task params (1 or 2 y-locations).
        task_params = scenario.sample_task_params(
            rng, cfg.z_height_low, cfg.z_height_range)            # [n_task_params]
        # Expand to [2] for the cost (insertion pins the second border).
        z_heights = scenario.z_targets_from_task(
            task_params, L_demo, **z_target_kwargs)
        z_t = torch.tensor(z_heights, dtype=dmp.dtype, device=dmp.device)

        print(f"[generate_data] run {run_id+1}/{cfg.n_runs} | "
              f"task_params={task_params.round(4)} -> z_targets={z_heights.round(4)}")

        result = pi2.optimize(z_t)

        run_dir = out_dir / f"train{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save per-run training data. Layout matches the avoidance format so
        # existing analysis tooling keeps working; train_policy decodes the
        # NN input via the scenario at load time.
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
            scenario=scenario_name,
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
    argv = sys.argv[1:]
    scenario_name = _pop_scenario_flag(argv)
    sys.argv = [sys.argv[0]] + argv

    cfg_cls = _DATAGEN_CFG[scenario_name]
    cfg = tyro.cli(cfg_cls)
    main(cfg, scenario_name)
