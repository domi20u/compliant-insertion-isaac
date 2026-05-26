"""Evaluate a trained NN policy by GPU-batched DMP rollouts.

Usage:

    # Box-avoidance (default — backward compatible)
    python scripts/test_policy.py --data-dir results/avoidance_run0 \
        --model-name model

    # Insertion
    python scripts/test_policy.py --scenario insertion \
        --data-dir results/insertion_run0 --model-name model \
        --n-basis 20
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import tyro

from configs.configs import (
    BoxAvoidCostConfig,
    DMPConfig,
    Insert2pCostConfig,
    TestConfig,
)
from core.cost_functions import acc_from_pos
from core.dmp_wrapper import DMPWrapper
from core.scenarios import Scenario, get_scenario, list_scenarios
from core.visualization import plot_nn_test


def _pop_scenario_flag(argv: list[str]) -> str:
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
    if name not in list_scenarios():
        raise SystemExit(
            f"Unknown --scenario {name!r}. Choices: {list_scenarios()}")
    return name


def _default_cost_cfg(scenario: Scenario, L_demo: float):
    """Build a scenario-appropriate cost config for diagnostic evaluation.

    Test-time cost values are only used for logging / success classification,
    so we instantiate the scenario's default config and just override
    ``max_val`` to ``L_demo`` (matches original avoidance behaviour).
    """
    cost_cfg = scenario.cost_cfg_type()
    #cost_cfg.max_val = L_demo
    return cost_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: TestConfig, scenario: Scenario) -> None:
    # Scenario owns n_inputs and n_dims — see train_policy.py for rationale.
    expected_n_inputs = scenario.n_task_params + 1
    if cfg.n_inputs != expected_n_inputs:
        print(f"[test_policy] overriding n_inputs {cfg.n_inputs} -> "
              f"{expected_n_inputs} (from scenario {scenario.name!r})")
        cfg.n_inputs = expected_n_inputs
    if cfg.n_dims != scenario.n_dims:
        print(f"[test_policy] overriding n_dims {cfg.n_dims} -> "
              f"{scenario.n_dims} (from scenario {scenario.name!r})")
        cfg.n_dims = scenario.n_dims

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    data_dir = Path(cfg.data_dir)

    print(f"[test_policy] scenario: {scenario.name}")
    print(f"[test_policy] device: {device}")
    print(f"[test_policy] data_dir: {data_dir}, model: {cfg.model_name}.pth")

    # ---- Load NN policy
    model_path = data_dir / f"{cfg.model_name}.pth"
    nn_policy = torch.jit.load(str(model_path), map_location=device)
    nn_policy.eval()

    # ---- Build DMP and imitate the demo
    dmp_cfg = DMPConfig(trajectory_file=cfg.trajectory_file,
                        n_bfs=cfg.n_basis - 1)
    cost_cfg = _default_cost_cfg(scenario, cfg.L_demo)
    print(f"[test_policy] cost config - max_val: {cost_cfg.max_val}")
    dmp = DMPWrapper(dmp_cfg, device=str(device))
    dmp.imitate_path(cfg.trajectory_file)
    init_weights = dmp.weights_init.clone()                  # [D, K]

    # Override goal y to match L_demo (mirrors original behavior)
    dmp.x_goal = dmp.x_goal.clone()
    dmp.x_goal[1] = cfg.L_demo

    # Insertion needs goal_proximity; avoidance ignores it via **_ignored.
    z_target_kwargs = {"goal_proximity": getattr(cost_cfg, "goal_proximity", 0.0)}

    # ---- Sample test obstacle/task configurations.
    # Output is ALWAYS [N, 3] — the third column is a dummy for insertion so
    # the per-sample loop / plotting stays scenario-agnostic.
    n_total = cfg.n_tests * cfg.n_tests_up

    if cfg.extrapolate_max_in1 is not None:
        x_low, x_high = cfg.max_in1, cfg.extrapolate_max_in1
    else:
        x_low, x_high = 0.0, cfg.max_in1

    obstacles = scenario.sample_test_obstacles(
        rng, n_total, x_low, x_high, cfg.min_in2, cfg.max_in3)   # [N, 3]
    print(f"[test_policy] sampled {n_total} test configurations "
          f"(extrapolate={cfg.extrapolate_max_in1 is not None})")

    # NN inputs come from the scenario (handles per-column offsets + slicing).
    nn_inputs = scenario.encode_test_obstacles(obstacles, cfg.ins_offset)  # [N, n_inputs]
    assert nn_inputs.shape == (n_total, cfg.n_inputs), \
        f"scenario produced {nn_inputs.shape}, expected {(n_total, cfg.n_inputs)}"
    nn_inputs_t = torch.tensor(nn_inputs, dtype=torch.float32, device=device)

    # ---- Single-pass NN inference (the whole test set in one batch)
    t0 = time.time()
    with torch.no_grad():
        new_means_flat = nn_policy.predict(nn_inputs_t)        # [N, K*n_dims]
    nn_ms = (time.time() - t0) * 1000 / n_total
    print(f"[test_policy] NN inference: {nn_ms:.3f} ms/sample")

    new_means = new_means_flat.reshape(n_total, cfg.n_basis, cfg.n_dims)  # [N, K, n_dims]

    # ---- Build full weight matrices: dof 0 stays init, dofs 1+2 from NN
    weights_batch = init_weights.unsqueeze(0).expand(n_total, -1, -1).clone()  # [N, D, K]
    weights_batch[:, 1, :] = new_means[:, :, 0]
    weights_batch[:, 2, :] = new_means[:, :, 1]

    # ---- Single batched rollout for the whole test set
    t0 = time.time()
    roll = dmp.rollout_batch(weights_batch)
    pos = roll["pos"]                                        # [N, T, 3]
    times = roll["times"]
    acc = acc_from_pos(pos, times)
    rollout_ms = (time.time() - t0) * 1000 / n_total
    print(f"[test_policy] DMP rollout: {rollout_ms:.3f} ms/sample (batched)")

    # ---- Evaluate per-sample (z_targets differs per sample).
    height_errors: list[float] = []
    bad_ids: list[int] = []
    counter = 0

    obstacles_t = torch.tensor(obstacles, dtype=dmp.dtype, device=device)
    L_demo_t = cfg.L_demo

    out_dir = data_dir / "tests" / f"list_{cfg.n_tests_up}_{cfg.n_tests}_off{cfg.ins_offset}"
    img_dir = out_dir / "graphics" / "img_new"
    cost_dir = out_dir / "costs"
    img_dir.mkdir(parents=True, exist_ok=True)
    cost_dir.mkdir(parents=True, exist_ok=True)

    pos_cpu = pos.detach().cpu().numpy()                     # [N, T, 3]

    for i in range(cfg.n_tests):
        all_costs = np.zeros([cfg.n_tests_up, 5])
        trajectories: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        ins_for_plot: list[np.ndarray] = []

        for j in range(cfg.n_tests_up):
            idx = i * cfg.n_tests_up + j
            ins = obstacles[idx]                             # [3]

            # Per-scenario z_targets: avoidance uses both y's; insertion
            # pins the second to L_demo - goal_proximity.
            z_targets = scenario.z_targets_from_obstacle_row(
                obstacles_t[idx], L_demo_t, **z_target_kwargs)    # [2]

            res = scenario.cost_fn(
                pos[idx:idx+1], acc[idx:idx+1], z_targets, cost_cfg,
                x_0_y=float(dmp.x_0[1].item()),
                x_goal_y=float(dmp.x_goal[1].item()),
                goal_pos=dmp.x_goal,
            )
            all_costs[j] = res.costs[0].detach().cpu().numpy()

            height_error = -all_costs[j, 1] / L_demo_t - ins[0]
            height_errors.append(height_error)
            if height_error < 0:
                bad_ids.append(counter)
                colors.append(np.array([1.0, 0.0, 0.0]))
            else:
                colors.append(np.array([0.0, 1.0, 0.0]))

            trajectories.append(pos_cpu[idx])
            ins_for_plot.append(np.array([ins[0], ins[1], ins[2], 0.0]))
            counter += 1

        np.savetxt(cost_dir / f"{i}_logging.txt", all_costs, fmt="%.6f", delimiter=",")
        plot_nn_test(trajectories, colors, np.asarray(ins_for_plot),
             L_demo=L_demo_t,
             out_path=img_dir / f"img_{i}.png",
             max_val=cost_cfg.max_val,
             n_inputs=cfg.n_inputs)

    # ---- Summary
    height_errors_arr = np.asarray(height_errors)
    n_bad = int((height_errors_arr < 0).sum())
    success_rate = 100.0 * (1.0 - n_bad / counter)
    bad_errs = -height_errors_arr[height_errors_arr < 0]
    good_errs = height_errors_arr[height_errors_arr >= 0]

    print(f"[test_policy] Completed {counter} tests.")
    print(f"[test_policy] Success rate: {success_rate:.2f}%")
    if bad_errs.size:
        print(f"[test_policy] Bad-error mean: {bad_errs.mean():.4f}, max: {bad_errs.max():.4f}")
    if good_errs.size:
        print(f"[test_policy] Good-error mean: {good_errs.mean():.4f}, max: {good_errs.max():.4f}")
    print(f"[test_policy] Bad IDs: {bad_ids}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    scenario_name = _pop_scenario_flag(argv)
    sys.argv = [sys.argv[0]] + argv

    cfg = tyro.cli(TestConfig)
    scenario = get_scenario(scenario_name)
    main(cfg, scenario)
