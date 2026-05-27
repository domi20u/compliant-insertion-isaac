"""Generate a single insertion DMP trajectory from the trained NN policy.

All configuration lives in a YAML file. The CLI only exposes the path to
that config and the two task parameters required by the insertion model.

Usage::

    python scripts/sim/gen_insertion_trajectory.py \
        --config configs/insertion_traj.yaml \
        --task-params 0.05 0.20

The script:
  1. Loads YAML config (paths, start/goal, n_basis, ins_offset, ...).
  2. Adds the path to ``methods/dmp`` to ``sys.path`` so ``core.dmp_wrapper``
     and ``configs.configs`` are importable.
  3. Loads the TorchScript NN policy.
  4. Builds the DMP, imitates the demonstration, overrides x_0 and x_goal.
  5. Computes the NN input as
         [task_param_1 + ins_offset[0],
          task_param_2 - ins_offset[1]]
     and predicts the y/z forcing-term weights.
  6. Rolls out once and saves [T, 3] to .npy (+ .npz with metadata).
  7. Optionally drops a 3D plot of the rollout for debugging (config flag
     ``debug_plot`` or CLI ``--plot``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ─── Config loading ─────────────────────────────────────────────────────────
REQUIRED_KEYS = (
    "methods_dmp", "model_path", "demo_path", "out_path",
    "start", "goal", "n_basis", "ins_offset",
)


def load_config(path: Path) -> dict:
    """Load YAML, validate required keys, resolve paths relative to repo root.

    Repo root is taken to be two levels up from this script (scripts/sim/foo.py
    => repo). Adjust if your layout differs.
    """
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"[gen_traj] config {path} is missing keys: {missing}")

    repo_root = Path(__file__).resolve().parents[2]
    for key in ("methods_dmp", "model_path", "demo_path", "out_path"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else (repo_root / p).resolve()

    # Defaults for optional fields.
    cfg.setdefault("device", "cuda")
    cfg.setdefault("seed", 0)
    cfg.setdefault("rescale", None)          # None | "rotodilatation" | "rotodilatation_xy"
    cfg.setdefault("debug_plot", False)
    cfg.setdefault("plot_path", None)        # if None and debug_plot, shows interactively

    # Sanity check on the offset shape.
    offs = cfg["ins_offset"]
    if not (isinstance(offs, (list, tuple)) and len(offs) == 2):
        raise SystemExit(
            f"[gen_traj] ins_offset must be a 2-element list, got {offs!r}")

    if cfg["rescale"] not in (None, "rotodilatation", "rotodilatation_xy"):
        raise SystemExit(
            f"[gen_traj] rescale must be one of None, 'rotodilatation', "
            f"'rotodilatation_xy' (got {cfg['rescale']!r}).")

    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate one insertion DMP rollout.")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to YAML config (see configs/insertion_traj.yaml).")
    p.add_argument("--task-params", type=float, nargs=2, required=True,
                   metavar=("P1", "P2"),
                   help="The two insertion task parameters.")
    p.add_argument("--plot", action="store_true",
                   help="Show/save a 3D plot of the rollout (overrides config).")
    p.add_argument("--plot-path", type=Path, default=None,
                   help="If set, save the plot to this path instead of showing it.")
    return p.parse_args()


def _bootstrap_paths(methods_dmp: Path) -> None:
    methods_dmp = methods_dmp.resolve()
    if not (methods_dmp / "core" / "dmp_wrapper.py").exists():
        raise SystemExit(
            f"[gen_traj] {methods_dmp} doesn't look like the dmp methods dir "
            f"(no core/dmp_wrapper.py). Check `methods_dmp` in the config.")
    sys.path.insert(0, str(methods_dmp))


def plot_rollout_3d(pos: np.ndarray,
                    start: np.ndarray,
                    goal: np.ndarray,
                    demo_x0: np.ndarray,
                    demo_xg: np.ndarray,
                    task_params: tuple[float, float],
                    rescale: str | None,
                    save_path: Path | None = None) -> None:
    """3D plot of the generated trajectory plus runtime/learned endpoints.

    Shows three things at a glance:
      - the rolled-out trajectory (line, color-graded by time);
      - runtime start/goal (large markers) — what the controller will track;
      - learned (demo) start/goal (small markers) — the rotodilatation
        source frame, drawn only when ``rescale`` is on so a side-by-side
        comparison is meaningful.
    """
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")        # headless if we're writing to disk
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory: scatter with a colormap gives a clear time-arrow without
    # needing arrowheads, which read badly in 3d. Light line underneath
    # connects the dots for continuity.
    t_norm = np.linspace(0, 1, len(pos))
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            color="0.7", lw=0.8, alpha=0.6, zorder=1)
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                    c=t_norm, cmap="viridis", s=8, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label("normalized time")

    def _fmt(v):
        return "[" + ", ".join(f"{float(x):.3f}" for x in v) + "]"

    # Runtime endpoints (what we actually executed).
    ax.scatter(*start, color="tab:green", s=120, marker="o",
               edgecolors="k", linewidths=0.8,
               label=f"runtime start {_fmt(start)}",
               zorder=5)
    ax.scatter(*goal, color="tab:red", s=120, marker="*",
               edgecolors="k", linewidths=0.8,
               label=f"runtime goal  {_fmt(goal)}",
               zorder=5)

    # Learned (demo) endpoints — only meaningful when rotodilatation is on,
    # otherwise they're identical to the runtime ones up to whatever the
    # user overrode them to.
    if rescale is not None and not (
            np.allclose(start, demo_x0) and np.allclose(goal, demo_xg)):
        ax.scatter(*demo_x0, color="tab:green", s=40, marker="o",
                   alpha=0.5,
                   label=f"demo start {_fmt(demo_x0)}",
                   zorder=4)
        ax.scatter(*demo_xg, color="tab:red", s=60, marker="*",
                   alpha=0.5,
                   label=f"demo goal  {_fmt(demo_xg)}",
                   zorder=4)
        # Faint reference line showing the demo's displacement, to make
        # the rotation visually obvious.
        ax.plot([demo_x0[0], demo_xg[0]],
                [demo_x0[1], demo_xg[1]],
                [demo_x0[2], demo_xg[2]],
                color="0.4", lw=0.6, ls="--", alpha=0.4,
                label="demo displacement")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    tp_str = "[" + ", ".join(f"{float(p):.4f}" for p in task_params) + "]"
    title = (f"DMP rollout  •  task_params={tp_str}  •  "
             f"rescale={rescale!r}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # Equal aspect ratio in 3d isn't supported pre-mpl 3.3; do it manually
    # so rotations don't look squashed (the whole point of this plot).
    all_pts = np.vstack([pos, start[None], goal[None],
                         demo_x0[None], demo_xg[None]])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    if radius == 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[gen_traj] saved plot -> {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _bootstrap_paths(cfg["methods_dmp"])

    # CLI overrides config for plot flags.
    do_plot = args.plot or cfg["debug_plot"]
    plot_path = args.plot_path if args.plot_path is not None else cfg["plot_path"]
    if plot_path is not None:
        plot_path = Path(plot_path)

    # Imports deferred until after sys.path is set up.
    from configs.configs import DMPConfig  # noqa: E402
    from core.dmp_wrapper import DMPWrapper  # noqa: E402

    torch.manual_seed(cfg["seed"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    print(f"[gen_traj] config={args.config}")
    print(f"[gen_traj] device={device}, seed={cfg['seed']}, rescale={cfg['rescale']!r}")

    # ----- Load NN policy.
    nn_policy = torch.jit.load(str(cfg["model_path"]), map_location=device)
    nn_policy.eval()
    print(f"[gen_traj] loaded NN policy from {cfg['model_path']}")

    # ----- Build DMP and imitate the demo.
    dmp_cfg = DMPConfig(trajectory_file=str(cfg["demo_path"]),
                        n_bfs=cfg["n_basis"] - 1, rescale=cfg["rescale"])
    dmp = DMPWrapper(dmp_cfg, device=str(device))
    dmp.imitate_path(str(cfg["demo_path"]))
    init_weights = dmp.weights_init.clone()                      # [D, K]
    # Capture the demo's endpoints from the wrapper's dedicated demo slots.
    # x_0 / x_goal also hold these values immediately after imitate_path,
    # but they get overwritten below — x_0_demo / x_goal_demo do not.
    demo_x0 = dmp.x_0_demo.detach().cpu().numpy().copy()
    demo_xg = dmp.x_goal_demo.detach().cpu().numpy().copy()
    print(f"[gen_traj] imitated demo: weights_init shape={tuple(init_weights.shape)} "
          f"demo_x0={demo_x0.tolist()} demo_xg={demo_xg.tolist()}")

    # ----- Override start and goal with the config-supplied points.
    start = torch.tensor(cfg["start"], dtype=dmp.dtype, device=device)
    goal = torch.tensor(cfg["goal"],   dtype=dmp.dtype, device=device)
    dmp.x_0 = start.clone()
    dmp.x_goal = goal.clone()
    print(f"[gen_traj] start={cfg['start']} goal={cfg['goal']}")

    # ----- Build the NN input: two task params, each shifted by ins_offset.
    # encoding: [p1 + offset[0], p2 - offset[1]]
    p1, p2 = args.task_params
    offs = cfg["ins_offset"]
    nn_input = np.array([[p1 + offs[0], p2 - offs[1]]], dtype=np.float32)
    print(f"[gen_traj] task_params={args.task_params} ins_offset={offs} "
          f"-> nn_input={nn_input.tolist()}")
    nn_input_t = torch.tensor(nn_input, device=device)

    n_dims = 2   # insertion: NN predicts forcing terms for dofs 1 and 2 (y, z)

    # ----- Predict the y, z forcing-term weights.
    with torch.no_grad():
        new_means_flat = nn_policy.predict(nn_input_t)           # [1, K*n_dims]

    expected = cfg["n_basis"] * n_dims
    if new_means_flat.shape[-1] != expected:
        raise RuntimeError(
            f"NN policy output {new_means_flat.shape[-1]} values; expected "
            f"n_basis*n_dims = {cfg['n_basis']}*{n_dims} = {expected}. Check "
            f"that n_basis in the config matches the trained model.")

    new_means = new_means_flat.reshape(1, cfg["n_basis"], n_dims)  # [1, K, n_dims]

    # ----- Assemble [N=1, D, K] weight tensor. dof 0 (x) stays at init.
    weights_batch = init_weights.unsqueeze(0).clone()            # [1, D, K]
    weights_batch[:, 1, :] = new_means[:, :, 0]
    weights_batch[:, 2, :] = new_means[:, :, 1]

    # ----- Rollout at the wrapper's native resolution. To play faster in
    # sim, use --playback-speed in run_dmp_in_sim.py, not a smaller
    # n_timesteps here. Down-sampling at generation gives the controller
    # a sparse waypoint stream and produces staircase tracking. -----
    roll = dmp.rollout_batch(weights_batch)
    pos = roll["pos"][0].detach().cpu().numpy()                  # [T, 3]
    times = roll["times"]
    if torch.is_tensor(times):
        times = times.detach().cpu().numpy()
    else:
        times = np.asarray(times)
    times = np.squeeze(times)
    if times.ndim > 1:
        times = times[0]
    times = np.ascontiguousarray(times, dtype=np.float32)
    if times.shape[0] != pos.shape[0]:
        raise RuntimeError(f"times has {times.shape[0]} samples but pos has "
                           f"{pos.shape[0]} — wrapper API mismatch.")
    print(f"[gen_traj] rollout: T={pos.shape[0]} steps, "
          f"first={pos[0].tolist()} last={pos[-1].tolist()}")

    # ----- Save.
    out_path = Path(cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path.with_suffix(".npz"),
             pos=pos, times=times,
             start=np.asarray(cfg["start"]),
             goal=np.asarray(cfg["goal"]),
             demo_x0=demo_x0, demo_xg=demo_xg,
             task_params=np.asarray(args.task_params),
             ins_offset=np.asarray(offs),
             rescale=np.asarray(str(cfg["rescale"])))
    np.save(out_path, pos.astype(np.float32))
    print(f"[gen_traj] saved {pos.shape} -> {out_path} (+ .npz metadata)")

    # ----- Optional debug plot.
    if do_plot:
        plot_rollout_3d(
            pos=pos,
            start=np.asarray(cfg["start"], dtype=float),
            goal=np.asarray(cfg["goal"], dtype=float),
            demo_x0=demo_x0,
            demo_xg=demo_xg,
            task_params=tuple(args.task_params),
            rescale=cfg["rescale"],
            save_path=plot_path,
        )


if __name__ == "__main__":
    main()
