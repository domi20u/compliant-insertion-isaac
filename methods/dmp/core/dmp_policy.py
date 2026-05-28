"""Shared construction of a policy-primed insertion DMP.

Both the offline generator (``gen_insertion_trajectory.py``) and the
end-to-end simulator (``run_dmp_in_sim.py``) need the same front-half:

    load NN policy  ->  build DMP  ->  imitate demo  ->  query policy for
    the y/z forcing weights  ->  assemble the [1, D, K] weight tensor.

Keeping that in one place means the two entry points can't drift apart in
how they encode the NN input, which DOFs the policy drives, or how the
weights get spliced onto the imitation baseline. The offline generator then
calls ``rollout_batch``; the simulator calls ``reset_step`` / ``step``.

This module deliberately knows nothing about Isaac Sim or YAML I/O — it
takes already-resolved values so it can be imported in either context.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class PrimedDMP:
    """A DMP wrapper that has imitated a demo and had policy weights applied.

    Attributes:
        dmp:          the DMPWrapper instance (already imitated + weighted).
        weights:      [1, D, K] weight tensor handed to rollout/step.
        demo_x0:      [D] demo start (rotodilatation source).
        demo_xg:      [D] demo goal.
        nn_input:     [1, 2] the encoded policy input actually used.
        tau:          DMP native duration in seconds.
    """
    dmp: object
    weights: torch.Tensor
    demo_x0: np.ndarray
    demo_xg: np.ndarray
    nn_input: np.ndarray
    tau: float


def build_primed_dmp(
    *,
    DMPWrapper,
    DMPConfig,
    nn_policy,
    demo_path: str,
    n_basis: int,
    task_params: tuple[float, float],
    ins_offset: tuple[float, float],
    start: Optional[np.ndarray],
    goal: Optional[np.ndarray],
    rescale: Optional[str],
    device: torch.device,
    n_dims: int = 2,
) -> PrimedDMP:
    """Build a DMP, imitate the demo, and apply the policy forcing weights.

    Args:
        DMPWrapper:  the class (injected so this module needs no sys.path
                     bootstrapping of its own).
        DMPConfig:   the config class.
        nn_policy:   a loaded TorchScript module exposing ``predict``.
        demo_path:   CSV demonstration path.
        n_basis:     number of DMP basis functions (== K).
        task_params: the two insertion task parameters (p1, p2).
        ins_offset:  the two-element offset applied when encoding nn_input.
        start:       runtime start [D] or None to keep the demo start.
        goal:        runtime goal  [D] or None to keep the demo goal.
        rescale:     None | "rotodilatation" | "rotodilatation_xy".
        device:      torch device.
        n_dims:      number of DOFs the policy drives (default 2 -> y, z).

    Returns:
        PrimedDMP with everything wired and ready to roll out or step.
    """
    dmp_cfg = DMPConfig(trajectory_file=str(demo_path),
                        n_bfs=n_basis - 1, rescale=rescale)
    dmp = DMPWrapper(dmp_cfg, device=str(device))
    dmp.imitate_path(str(demo_path))

    init_weights = dmp.weights_init.clone()                      # [D, K]
    demo_x0 = dmp.x_0_demo.detach().cpu().numpy().copy()
    demo_xg = dmp.x_goal_demo.detach().cpu().numpy().copy()

    # Override the runtime endpoints (rotodilatation source stays pinned to
    # the demo via x_0_demo / x_goal_demo inside the wrapper).
    if start is not None:
        dmp.x_0 = torch.as_tensor(start, dtype=dmp.dtype, device=device).clone()
    if goal is not None:
        dmp.x_goal = torch.as_tensor(goal, dtype=dmp.dtype, device=device).clone()

    # Encode the NN input: [p1 + offset[0], p2 - offset[1]].
    p1, p2 = task_params
    nn_input = np.array([[p1 + ins_offset[0], p2 - ins_offset[1]]],
                        dtype=np.float32)
    nn_input_t = torch.tensor(nn_input, device=device)

    with torch.no_grad():
        new_means_flat = nn_policy.predict(nn_input_t)           # [1, K*n_dims]

    expected = n_basis * n_dims
    if new_means_flat.shape[-1] != expected:
        raise RuntimeError(
            f"NN policy output {new_means_flat.shape[-1]} values; expected "
            f"n_basis*n_dims = {n_basis}*{n_dims} = {expected}. Check that "
            f"n_basis matches the trained model.")

    new_means = new_means_flat.reshape(1, n_basis, n_dims)        # [1, K, n_dims]

    # Splice policy weights onto the imitation baseline. DOF 0 (x) keeps its
    # imitated weights; DOFs 1..n_dims (y, z) get the policy's.
    weights = init_weights.unsqueeze(0).clone()                  # [1, D, K]
    for d in range(n_dims):
        weights[:, 1 + d, :] = new_means[:, :, d]

    return PrimedDMP(
        dmp=dmp,
        weights=weights,
        demo_x0=demo_x0,
        demo_xg=demo_xg,
        nn_input=nn_input,
        tau=float(dmp.tau),
    )
