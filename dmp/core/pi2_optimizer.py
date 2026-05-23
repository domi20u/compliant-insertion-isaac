"""GPU-batched PI² (Path Integral Policy Improvement) optimizer.

Replaces the original ``PI2.optimize_continuous_box_acc`` method. Key
differences:

* **No plotting.** Returns trajectories + diagnostics for offline plotting.
* **Single-call rollout** of all ``n_samples`` candidates per iteration.
* **Vectorized cost evaluation** end-to-end on the device.
* **Pure**: holds no state across runs apart from ``current_weights``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from configs.configs import BoxAvoidCostConfig, PI2Config
from core.cost_functions import (
    BoxAvoidResult,
    acc_from_pos,
    costs_to_weights,
    evaluate_box_avoid,
)
from core.dmp_wrapper import DMPWrapper


@dataclass
class PI2Result:
    """All artifacts from one PI² run, kept on CPU for serialization."""

    train_data: np.ndarray            # [n_iter, num_basis, num_dof]
    label_z: np.ndarray               # [n_iter, n_obstacles]
    avg_costs: np.ndarray             # [n_iter, 5]
    y_locations: np.ndarray           # [n_iter, n_obstacles]
    final_x_track: np.ndarray         # [T, num_dof]
    n_iter: int
    duration: float
    finished: bool


class PI2Optimizer:
    """PI² optimizer parameterised by a vectorised cost function.

    Task-specific behaviour lives entirely in ``cost_fn`` + ``cost_cfg``,
    which conform to the signature shared by ``evaluate_box_avoid`` and
    ``evaluate_insert_2p`` (both return ``BoxAvoidResult``). Defaults to the
    box-avoid cost so existing call sites keep working.
    """

    def __init__(self, dmp: DMPWrapper, cfg: PI2Config,
                 cost_cfg,
                 cost_fn=None):
        self.dmp = dmp
        self.cfg = cfg
        self.cost_cfg = cost_cfg
        self.cost_fn = cost_fn if cost_fn is not None else evaluate_box_avoid
        self.device = dmp.device
        self.dtype = dmp.dtype

        self.n_basis = dmp.num_basis
        self.n_dof = dmp.num_dof

        self.covar = self._init_covariance()  # [num_basis, num_basis, num_dof]
        self.current_weights: torch.Tensor | None = None     # [num_dof, num_basis]

    # ------------------------------------------------------------ initial cov
    def _init_covariance(self) -> torch.Tensor:
        """Per-DOF diagonal covariance with exponentially-spaced sigmas."""
        s_min, s_max = self.cfg.sigma_min, self.cfg.sigma_max
        steepness = self.cfg.sigma_steepness
        n = self.n_basis

        exponents = torch.linspace(0, 1, n, dtype=self.dtype, device=self.device)
        exponents = exponents.pow(steepness)
        sigmas = s_min + (s_max - s_min) * exponents
        diag = (torch.exp(sigmas) - 1.0).pow(2)               # [num_basis]

        cov_per_dof = torch.diag(diag)                        # [num_basis, num_basis]
        # Replicate per DOF — shape [num_basis, num_basis, num_dof]
        return cov_per_dof.unsqueeze(-1).expand(-1, -1, self.n_dof).contiguous()

    # ---------------------------------------------------------------- explore
    def explore(self, mean_w: torch.Tensor) -> torch.Tensor:
        """Sample n_samples candidate weight matrices around the current mean.

        Returns:
            samples: [N, num_dof, num_basis]
        """
        N = self.cfg.n_samples
        D, K = self.n_dof, self.n_basis
        out = mean_w.unsqueeze(0).expand(N, D, K).clone()    # [N, D, K]

        for d in range(D):
            if self.cfg.param_perturb[d]:
                cov_d = self.covar[..., d]                    # [K, K]
                dist = torch.distributions.MultivariateNormal(
                    loc=mean_w[d], covariance_matrix=cov_d)
                out[:, d, :] = dist.sample((N,))
        return out

    # ----------------------------------------------------------- main routine
    def optimize(self, z_targets: torch.Tensor) -> PI2Result:
        """Run PI² for the configured task.

        Args:
            z_targets: [2] y-locations consumed by the cost. For avoidance,
              both are free; for insertion, the second is the scenario-pinned
              slot bottom.

        Returns:
            PI2Result with all per-iteration data on CPU/numpy.
        """
        import time

        cfg = self.cfg
        cost_cfg = self.cost_cfg
        dmp = self.dmp

        # Initialize from the imitated weights every run
        self.current_weights = dmp.weights_init.clone()
        self.covar = self._init_covariance()

        max_iters = cfg.max_iters
        n_obstacles = z_targets.numel()

        train_data = np.zeros((max_iters, self.n_basis, self.n_dof))
        label_z = np.zeros((max_iters, n_obstacles))
        y_locations = np.zeros((max_iters, n_obstacles))
        avg_costs = np.zeros((max_iters, 5))

        x_0_y = float(dmp.x_0[1].item())
        x_goal_y = float(dmp.x_goal[1].item())

        z_targets = z_targets.to(self.device, self.dtype)

        t_start = time.time()
        finished = False
        ii = 0

        for ii in range(max_iters):
            # ---- 1) explore: sample N candidates around mean
            samples = self.explore(self.current_weights)        # [N, D, K]

            # ---- 2) batched rollout of all candidates
            roll = dmp.rollout_batch(samples)
            pos = roll["pos"]                                   # [N, T, D]
            times = roll["times"]
            acc = acc_from_pos(pos, times)

            # ---- 3) vectorized cost evaluation
            res = self.cost_fn(pos, acc, z_targets, cost_cfg,
                               x_0_y, x_goal_y,
                               goal_pos=dmp.x_goal)
            costs_total = res.costs[:, 0]                       # [N]
            #print(f"#{ii}: costs: {costs_total.min().item():.4f} .. {costs_total.max().item():.4f} ")

            # ---- 4) PI² weight update on the cost-weighted samples
            w = costs_to_weights(costs_total, h=cfg.h)          # [N]
            mean_new = (w.view(-1, 1, 1) * samples).sum(dim=0)  # [D, K]

            self.current_weights = mean_new
            self.covar = (cfg.covar_decay ** 2) * self.covar

            # ---- 5) evaluate the policy mean (single-batch rollout)
            roll_mean = dmp.rollout_batch(self.current_weights.unsqueeze(0))
            pos_m = roll_mean["pos"]
            acc_m = acc_from_pos(pos_m, roll_mean["times"])
            res_m = self.cost_fn(pos_m, acc_m, z_targets, cost_cfg,
                                 x_0_y, x_goal_y,
                                 goal_pos=dmp.x_goal)

            train_data[ii] = mean_new.detach().cpu().numpy().T   # [num_basis, num_dof]
            avg_costs[ii] = res_m.costs[0].detach().cpu().numpy()
            label_z[ii] = res_m.height_at_borders[0].detach().cpu().numpy()
            y_locations[ii] = res_m.y_borders[0].detach().cpu().numpy()

            if res_m.finished[0].item():
                finished = True
                break

        duration = time.time() - t_start

        # Trim arrays to the iterations that actually ran
        n_iter = ii + 1
        final_pos = pos_m[0].detach().cpu().numpy()             # [T, num_dof]

        return PI2Result(
            train_data=train_data[:n_iter],
            label_z=label_z[:n_iter],
            avg_costs=avg_costs[:n_iter],
            y_locations=y_locations[:n_iter],
            final_x_track=final_pos,
            n_iter=n_iter,
            duration=duration,
            finished=finished,
        )
