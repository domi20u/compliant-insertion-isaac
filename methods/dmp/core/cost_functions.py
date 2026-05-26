"""Fully-vectorized cost functions for PI² obstacle-avoidance.

Every function takes a batched rollout tensor ``pos`` of shape ``[N, T, D]``
(positions) and returns scalar costs of shape ``[N, n_terms]``. The first
column is always the total cost; the remainder are diagnostic terms that
match the layout used by the original ``evaluate_rollout_*`` functions.

The cost functions are pure: no side effects, no plotting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from configs.configs import BoxAvoidCostConfig, Insert2pCostConfig


class BoxAvoidResult(NamedTuple):
    """Per-sample diagnostic info from the box-avoid cost evaluation."""

    costs: torch.Tensor       # [N, 5] total + 4 components
    height_at_borders: torch.Tensor  # [N, 2] z at the two y-borders
    y_borders: torch.Tensor   # [N, 2]
    finished: torch.Tensor    # [N] bool — True if all samples crossed max_val
    bbox: torch.Tensor        # [N, 4] (y_min, y_max, z_min, z_max)


def _argmin_abs(diff: torch.Tensor) -> torch.Tensor:
    """Index of minimum |diff| along dim=-1, returns [N]."""
    return torch.argmin(torch.abs(diff), dim=-1)


def evaluate_box_avoid(pos: torch.Tensor,
                       acc_demo: torch.Tensor,
                       z_targets: torch.Tensor,
                       cfg: BoxAvoidCostConfig,
                       x_0_y: float,
                       x_goal_y: float,
                       goal_pos: torch.Tensor | None = None) -> BoxAvoidResult:
    """Vectorized version of evaluate_rollout_continuous_box_acc (len(z)==2 path).

    Args:
        pos:         [N, T, 3] rolled-out positions (x, y, z)
        acc_demo:    [N, T, 3] rolled-out accelerations (used for smoothness)
        z_targets:   [2] obstacle y-locations (called 'z' in the original PI2)
        cfg:         cost weighting config
        x_0_y:       demo initial y position
        x_goal_y:    demo goal y position
        goal_pos:    [3] goal position; if provided AND cfg.goal_penalty > 0,
                     a goal-convergence term is added to the cost.

    Returns:
        BoxAvoidResult
    """
    N, T, _ = pos.shape
    device = pos.device
    assert z_targets.numel() == 2, "Box-avoid cost only supports 2 obstacles."

    y_traj = pos[..., 1]                                     # [N, T]
    z_traj = pos[..., 2]                                     # [N, T]

    # Indices closest to the two y-targets (per sample)
    z0, z1 = z_targets[0], z_targets[1]
    id0 = _argmin_abs(y_traj - z0)                           # [N]
    id1 = _argmin_abs(y_traj - z1)                           # [N]
    # Avoid empty slice between borders
    id0 = torch.where(id0 == id1, id0 - 1, id0).clamp(min=0)

    n_idx = torch.arange(N, device=device)
    y_b0 = y_traj[n_idx, id0]
    y_b1 = y_traj[n_idx, id1]
    z_b0 = z_traj[n_idx, id0]
    z_b1 = z_traj[n_idx, id1]

    # cost[1] = -min_z over the segment [id0, id1] per sample
    # We mask z values outside [id0, id1] to +inf before reducing.
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(N, T)  # [N, T]
    in_box = (t_idx >= id0.unsqueeze(1)) & (t_idx <= id1.unsqueeze(1))
    z_masked = torch.where(in_box, z_traj, torch.full_like(z_traj, float("inf")))
    min_z_in_box = z_masked.min(dim=1).values                 # [N]
    cost_height = -min_z_in_box

    # cost[3] = bounds violation on y (must stay within [x_0_y - bL, x_goal_y + bR])
    # Original formulation: -sum(min(0, bL + y - x_0_y)) - sum(min(0, bR + x_goal_y - y))
    bL, bR = cfg.bounds
    lower_violation = -torch.clamp(bL + y_traj - x_0_y, max=0.0).sum(dim=1)
    upper_violation = -torch.clamp(bR + x_goal_y - y_traj, max=0.0).sum(dim=1)
    cost_bounds = lower_violation + upper_violation

    # cost[2] = table penalty: -10 * sum of negative z's
    z_below = torch.clamp(z_traj, max=0.0)
    cost_table = -cfg.table_penalty * z_below.sum(dim=1)

    # cost[4] = smoothness: initial-acc magnitude + jerk norm
    init_acc_mag = acc_demo[:, 0, :].abs().sum(dim=1)
    jerk = acc_demo[:, 1:, :] - acc_demo[:, :-1, :]
    jerk_norm = torch.linalg.norm(jerk.reshape(N, -1), dim=1)
    cost_smooth = cfg.smoothness_initial_acc * init_acc_mag + cfg.smoothness_jerk * jerk_norm

    # Optional goal-convergence cost: penalize end-effector deviation from goal
    # Set BoxAvoidCostConfig.goal_penalty > 0 to enable.
    if cfg.goal_penalty > 0.0 and goal_pos is not None:
        goal_err = torch.linalg.norm(pos[:, -1, :] - goal_pos.unsqueeze(0), dim=1)
        cost_smooth = cost_smooth + cfg.goal_penalty * goal_err

    # Total
    cost_total = cost_height + cost_bounds + cost_table + cost_smooth

    costs = torch.stack([cost_total, cost_height, cost_table, cost_bounds, cost_smooth],
                        dim=1)                                # [N, 5]

    finished = cost_height < -cfg.max_val                     # [N] bool

    bbox = torch.stack([y_traj.min(dim=1).values,
                        y_traj.max(dim=1).values,
                        z_traj.min(dim=1).values,
                        z_traj.max(dim=1).values], dim=1)     # [N, 4]

    height_at_borders = torch.stack([z_b0, z_b1], dim=1)      # [N, 2]
    y_borders = torch.stack([y_b0, y_b1], dim=1)              # [N, 2]

    return BoxAvoidResult(costs=costs,
                          height_at_borders=height_at_borders,
                          y_borders=y_borders,
                          finished=finished,
                          bbox=bbox)

def evaluate_insert_2p(pos: torch.Tensor,
                       acc_demo: torch.Tensor,
                       z_targets: torch.Tensor,
                       cfg: Insert2pCostConfig,
                       x_0_y: float,
                       x_goal_y: float,
                       goal_pos: torch.Tensor | None = None,
                       goal_proximity: float = 0.00075) -> BoxAvoidResult:
    """Vectorized cost for the two-parameter insertion task.
 
    Mirrors ``evaluate_box_avoid`` term-for-term so the rest of the framework
    (PI² loop, logging, ``BoxAvoidResult`` consumers) stays unchanged. The
    geometric reinterpretation vs. box-avoidance:
 
      * ``z_targets[0]`` is the slot opening — the trajectory must clear it
        (same role as the first obstacle in box-avoidance).
      * ``z_targets[1]`` is the slot bottom, pinned by the scenario to
        ``L_demo - goal_proximity``. The trajectory must reach it (and ideally
        approach it vertically because of the asymmetric ``cfg.bounds``).
 
    The ``cost_height`` term is identical to box-avoidance: minimise z over
    the in-slot segment [id0, id1]. Insertion-specific shaping (an optional
    verticality penalty on the final descent) is gated by
    ``cfg.verticality_penalty``; set it to 0 (the default) to reproduce the
    old ``optimize_continuous_box_acc`` behaviour exactly.
 
    Args:
        pos:             [N, T, 3] rolled-out positions (x, y, z)
        acc_demo:        [N, T, 3] rolled-out accelerations (smoothness)
        z_targets:       [2] = (slot opening y, slot bottom y)
        cfg:             insertion cost weighting config
        x_0_y:           demo initial y position
        x_goal_y:        demo goal y position
        goal_pos:        [3] goal position; if given and cfg.goal_penalty > 0
                         a goal-convergence term is added.
        goal_proximity:  passed by the scenario for sanity-checking only —
                         the actual bound envelope lives in cfg.bounds.
 
    Returns:
        BoxAvoidResult — same shape/semantics as box-avoidance so existing
        callers (npz writer, plot_pi2_run, test_policy summary) work as-is.
    """
    N, T, _ = pos.shape
    device = pos.device
    assert z_targets.numel() == 2, "Insertion cost expects (slot_opening, slot_bottom)."
 
    y_traj = pos[..., 1]                                     # [N, T]
    z_traj = pos[..., 2]                                     # [N, T]
 
    # Indices closest to the slot opening (z0) and slot bottom (z1) per sample.
    z0, z1 = z_targets[0], z_targets[1]
    id0 = _argmin_abs(y_traj - z0)                           # [N]
    id1 = _argmin_abs(y_traj - z1)                           # [N]
    # Avoid an empty slice if the trajectory grazed both at the same step.
    id0 = torch.where(id0 == id1, id0 - 1, id0).clamp(min=0)
 
    n_idx = torch.arange(N, device=device)
    y_b0 = y_traj[n_idx, id0]
    y_b1 = y_traj[n_idx, id1]
    z_b0 = z_traj[n_idx, id0]
    z_b1 = z_traj[n_idx, id1]
 
    # cost_height = -min_z over [id0, id1] per sample.
    # Geometric meaning differs from box-avoidance (here we want the trajectory
    # to *reach* the slot bottom, not clear it), but mathematically identical:
    # the lower min_z is, the further the tool has descended into the slot.
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(N, T)
    in_slot = (t_idx >= id0.unsqueeze(1)) & (t_idx <= id1.unsqueeze(1))
    z_masked = torch.where(in_slot, z_traj, torch.full_like(z_traj, float("inf")))
    min_z_in_slot = z_masked.min(dim=1).values
    cost_height = -min_z_in_slot
 
    # cost_bounds: y must stay within [x_0_y - bL, x_goal_y + bR].
    # For insertion cfg.bounds is typically [0.1, goal_proximity] — a wide
    # left envelope during approach, a tight right envelope during descent.
    bL, bR = cfg.bounds
    lower_violation = -torch.clamp(bL + y_traj - x_0_y, max=0.0).sum(dim=1)
    upper_violation = -torch.clamp(bR + x_goal_y - y_traj, max=0.0).sum(dim=1)
    cost_bounds = lower_violation + upper_violation
 
    # cost_table: penalise negative z (below the table). Same as box-avoid.
    z_below = torch.clamp(z_traj, max=0.0)
    cost_table = -cfg.table_penalty * z_below.sum(dim=1)
 
    # cost_smooth: initial-acc magnitude + jerk norm. Same as box-avoid.
    init_acc_mag = acc_demo[:, 0, :].abs().sum(dim=1)
    jerk = acc_demo[:, 1:, :] - acc_demo[:, :-1, :]
    jerk_norm = torch.linalg.norm(jerk.reshape(N, -1), dim=1)
    cost_smooth = cfg.smoothness_initial_acc * init_acc_mag + cfg.smoothness_jerk * jerk_norm
 
    # Optional insertion-specific verticality term: penalise lateral (x and y)
    # motion during the final descent segment [id1 .. T-1]. Off by default so
    # this function is a drop-in replacement for the reused box-avoid cost.
    if getattr(cfg, "verticality_penalty", 0.0) > 0.0:
        in_descent = t_idx >= id1.unsqueeze(1)               # [N, T] bool
        # Lateral displacement step-to-step in x and y.
        dx = pos[:, 1:, 0] - pos[:, :-1, 0]
        dy = pos[:, 1:, 1] - pos[:, :-1, 1]
        lat_step = torch.sqrt(dx * dx + dy * dy + 1e-12)     # [N, T-1]
        # Mask to descent region (align with the [:, 1:] step axis).
        descent_mask = in_descent[:, 1:]
        lat_in_descent = torch.where(descent_mask, lat_step, torch.zeros_like(lat_step))
        cost_smooth = cost_smooth + cfg.verticality_penalty * lat_in_descent.sum(dim=1)
 
    # Optional goal-convergence cost.
    if cfg.goal_penalty > 0.0 and goal_pos is not None:
        goal_err = torch.linalg.norm(pos[:, -1, :] - goal_pos.unsqueeze(0), dim=1)
        cost_smooth = cost_smooth + cfg.goal_penalty * goal_err
 
    cost_total = cost_height + cost_bounds + cost_table + cost_smooth
 
    costs = torch.stack([cost_total, cost_height, cost_table, cost_bounds, cost_smooth],
                        dim=1)                                # [N, 5]
 
    # "Finished" semantics differ from box-avoidance: insertion is done when
    # the tool has descended to the slot bottom (z below -cfg.max_val), but
    # since cost_height encodes -min_z_in_slot the same predicate works.
    finished = cost_height < -cfg.max_val
 
    bbox = torch.stack([y_traj.min(dim=1).values,
                        y_traj.max(dim=1).values,
                        z_traj.min(dim=1).values,
                        z_traj.max(dim=1).values], dim=1)     # [N, 4]
 
    height_at_borders = torch.stack([z_b0, z_b1], dim=1)      # [N, 2]
    y_borders = torch.stack([y_b0, y_b1], dim=1)              # [N, 2]
 
    return BoxAvoidResult(costs=costs,
                          height_at_borders=height_at_borders,
                          y_borders=y_borders,
                          finished=finished,
                          bbox=bbox)



def costs_to_weights(costs: torch.Tensor, h: float) -> torch.Tensor:
    """Convert per-sample scalar costs to PI² importance weights.

    weights_i = exp(-h * (c_i - min(c)) / range(c))    — normalized to sum 1.
    """
    c_min = costs.min()
    c_max = costs.max()
    rng = c_max - c_min

    if rng.item() == 0.0:
        return torch.full_like(costs, 1.0 / costs.numel())

    w = torch.exp(-h * (costs - c_min) / rng)
    return w / w.sum()


def acc_from_pos(pos: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Numerical second derivative of pos w.r.t. times. Returns [N, T, D]."""
    dt = (times[:, 1:] - times[:, :-1]).unsqueeze(-1)         # [N, T-1, 1]
    vel = torch.zeros_like(pos)
    vel[:, :-1] = (pos[:, 1:] - pos[:, :-1]) / dt
    vel[:, -1] = vel[:, -2]
    acc = torch.zeros_like(pos)
    acc[:, :-1] = (vel[:, 1:] - vel[:, :-1]) / dt
    acc[:, -1] = acc[:, -2]
    return acc
