"""Scenario registry: per-task glue between PI², data, and NN policy I/O.

A ``Scenario`` owns everything that differs between tasks (box-avoidance,
insertion, future ones) without forcing every script to special-case the
task. Each scenario provides:

* ``n_task_params``: number of FREE task parameters (avoidance=2 y-locations,
  insertion=1 slot-opening y). The NN input is ``n_task_params + 1`` (the
  +1 is the obstacle/target height).
* ``sample_task_params`` / ``z_targets_from_task``: how to draw a task and
  how to expand it into the ``z_targets`` tensor consumed by the cost (the
  cost interface — two y-borders — is shared across tasks; insertion just
  pins the second border internally).
* ``cost_fn`` + ``cost_cfg_type``: which vectorised cost to call.
* ``encode_nn_input`` / ``decode_obstacles_for_eval``: how raw task data is
  turned into the NN feature vector at train and test time. Centralising
  these prevents the ``[z_min, y0, y1]`` vs ``[z_min, y0]`` split from
  leaking into the scripts.

To add a new task, register a subclass with ``@register_scenario("name")``;
no script changes needed beyond passing ``--scenario name``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Protocol

import numpy as np
import torch

from configs.configs import BoxAvoidCostConfig, Insert2pCostConfig
from core.cost_functions import (
    BoxAvoidResult,
    evaluate_box_avoid,
    evaluate_insert_2p,
)


# ---------------------------------------------------------------------------
# Cost-function signature shared by all scenarios
# ---------------------------------------------------------------------------
class CostFn(Protocol):
    def __call__(self,
                 pos: torch.Tensor,
                 acc_demo: torch.Tensor,
                 z_targets: torch.Tensor,
                 cfg,                                  # scenario-specific cfg
                 x_0_y: float,
                 x_goal_y: float,
                 goal_pos: torch.Tensor | None = None,
                 ) -> BoxAvoidResult: ...


# ---------------------------------------------------------------------------
# Base scenario
# ---------------------------------------------------------------------------
class Scenario:
    """Per-task interface. Subclass + register to add a new task."""

    name: ClassVar[str] = ""
    n_task_params: ClassVar[int] = 0       # free y-params; NN inputs = n_task_params + 1
    n_dims: ClassVar[int] = 2              # perturbed DMP dimensions (same for both tasks)
    cost_fn: ClassVar[CostFn]
    cost_cfg_type: ClassVar[type]

    # ------------------------------------------------------------------ tasks
    def sample_task_params(self,
                           rng: np.random.Generator,
                           low: float,
                           span: float) -> np.ndarray:
        """Draw the free y-parameters for one PI² run.

        Returns shape ``[n_task_params]``. Always sorted ascending so the
        cost's `id0 < id1` invariant holds without extra logic.
        """
        ys = np.sort(low + span * rng.random(self.n_task_params))
        return ys

    def z_targets_from_task(self,
                            task_params: np.ndarray,
                            L_demo: float) -> np.ndarray:
        """Expand free task params into the 2-vector consumed by the cost.

        Always returns shape ``[2]`` so the cost interface stays uniform.
        Scenarios with fewer free params (insertion) pin the remaining
        border here.
        """
        raise NotImplementedError

    # --------------------------------------------------------------- NN I/O
    def encode_nn_input(self,
                        height_norm: float,
                        y_locations_norm: np.ndarray) -> np.ndarray:
        """Build the NN input row from per-run results.

        Used by ``train_policy`` when stitching saved PI² runs into a dataset.

        Args:
            height_norm: scalar, min obstacle/slot z normalized by L_demo.
            y_locations_norm: ``[2]`` y-border locations normalized by L_demo
              (as saved by PI² for both scenarios).

        Returns:
            ``[n_task_params + 1]`` feature row.
        """
        raise NotImplementedError

    def encode_test_obstacles(self,
                              obstacles_raw: np.ndarray,
                              ins_offset: float) -> np.ndarray:
        """At test time, turn raw sampled obstacle rows into NN input rows.

        Args:
            obstacles_raw: ``[N, 3]`` columns ``(height, y0, y1)``. For
              insertion only the first two columns are meaningful; y1 is
              ignored.
            ins_offset: legacy offset applied to NN inputs at test time.

        Returns:
            ``[N, n_task_params + 1]`` feature matrix.
        """
        raise NotImplementedError

    def sample_test_obstacles(self,
                              rng: np.random.Generator,
                              n: int,
                              x_low: float,
                              x_high: float,
                              y_low: float,
                              y_high: float) -> np.ndarray:
        """Generate ``[n, 3]`` test obstacle rows ``(height, y0, y1)``.

        Always returns 3 columns (y1 may be a pinned dummy for insertion)
        so ``test_policy``'s downstream plotting and per-sample evaluation
        loop stay shared across scenarios.
        """
        raise NotImplementedError

    def z_targets_from_obstacle_row(self,
                                    obstacle_row: torch.Tensor,
                                    L_demo: float) -> torch.Tensor:
        """At test time, build the cost's ``z_targets`` tensor from one
        obstacle row. Mirrors ``z_targets_from_task`` but for the
        already-normalized rows produced at test time."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type[Scenario]] = {}


def register_scenario(name: str) -> Callable[[type[Scenario]], type[Scenario]]:
    def _wrap(cls: type[Scenario]) -> type[Scenario]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_scenario(name: str) -> Scenario:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown scenario {name!r}. Registered: {sorted(_REGISTRY)}.")
    return _REGISTRY[name]()


def list_scenarios() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Box-avoidance: 2 free y-locations, NN has 3 inputs
# ---------------------------------------------------------------------------
@register_scenario("avoidance")
class BoxAvoidScenario(Scenario):
    n_task_params: ClassVar[int] = 2
    n_dims: ClassVar[int] = 2
    cost_fn: ClassVar[CostFn] = staticmethod(evaluate_box_avoid)
    cost_cfg_type: ClassVar[type] = BoxAvoidCostConfig

    def z_targets_from_task(self, task_params: np.ndarray,
                            L_demo: float, **_ignored) -> np.ndarray:
        # No pinning — both y's are free. Extra kwargs are accepted for
        # interface uniformity with InsertionScenario.
        assert task_params.shape == (2,)
        return task_params.astype(np.float64, copy=True)

    def encode_nn_input(self, height_norm: float,
                        y_locations_norm: np.ndarray) -> np.ndarray:
        # [min_z, y0, y1] / L_demo  (matches original behaviour)
        return np.concatenate([[height_norm], y_locations_norm.astype(np.float64)])

    def encode_test_obstacles(self, obstacles_raw: np.ndarray,
                              ins_offset: float) -> np.ndarray:
        # Original behaviour: add (+off, -off, +off) to (h, y0, y1) — note the
        # asymmetric sign on the middle column. Kept for backward compat.
        offsets = np.array([ins_offset, -ins_offset, ins_offset])
        return obstacles_raw + offsets[None, :]

    def sample_test_obstacles(self, rng: np.random.Generator, n: int,
                              x_low: float, x_high: float,
                              y_low: float, y_high: float) -> np.ndarray:
        out: list[list[float]] = []
        while len(out) < n:
            batch = (n - len(out)) * 2
            x = rng.uniform(x_low, x_high, batch)
            y0 = rng.uniform(y_low, y_high, batch)
            y1 = rng.uniform(y_low, y_high, batch)
            for xi, y0i, y1i in zip(x, y0, y1):
                if y0i < y1i:
                    out.append([xi, y0i, y1i])
                    if len(out) == n:
                        break
        return np.asarray(out)

    def z_targets_from_obstacle_row(self, obstacle_row: torch.Tensor,
                                    L_demo: float, **_ignored) -> torch.Tensor:
        # obstacle_row: [height, y0, y1]. Cost wants y0, y1 in absolute units.
        return obstacle_row[1:3] * L_demo


# ---------------------------------------------------------------------------
# Insertion: 1 free y-location (slot opening); slot bottom pinned near goal
# ---------------------------------------------------------------------------
@register_scenario("insertion")
class InsertionScenario(Scenario):
    n_task_params: ClassVar[int] = 1
    n_dims: ClassVar[int] = 2
    cost_fn: ClassVar[CostFn] = staticmethod(evaluate_insert_2p)
    cost_cfg_type: ClassVar[type] = Insert2pCostConfig

    # The slot bottom sits at L_demo - goal_proximity (set by cost_cfg). We
    # cache the value we used so train/test can recover it; passed in by the
    # caller (PI² loop or test loop) from the cost config.
    def _pinned_bottom_norm(self, L_demo: float,
                            goal_proximity: float) -> float:
        return 1.0 - goal_proximity / L_demo

    def _pinned_bottom_abs(self, L_demo: float,
                           goal_proximity: float) -> float:
        return L_demo - goal_proximity

    # ------------------------------------------------------------------ tasks
    def z_targets_from_task(self, task_params: np.ndarray,
                            L_demo: float,
                            goal_proximity: float = 0.00075) -> np.ndarray:
        """Pin the second border at ``L_demo - goal_proximity``."""
        assert task_params.shape == (1,)
        y0 = float(task_params[0])
        y1 = self._pinned_bottom_abs(L_demo, goal_proximity)
        # Guard against the rare case where the sampled y0 exceeds the pin.
        if y0 >= y1:
            y0 = y1 - 1e-3
        return np.array([y0, y1], dtype=np.float64)

    # --------------------------------------------------------------- NN I/O
    def encode_nn_input(self, height_norm: float,
                        y_locations_norm: np.ndarray) -> np.ndarray:
        # Only the slot-opening y (y0) is informative; drop the pinned y1.
        return np.array([height_norm, float(y_locations_norm[0])])

    def encode_test_obstacles(self, obstacles_raw: np.ndarray,
                              ins_offset: float) -> np.ndarray:
        # Apply the same per-column offset convention as box-avoid for the
        # columns we actually use, then keep only (height, y0).
        offsets = np.array([ins_offset, -ins_offset])
        return obstacles_raw[:, :2] + offsets[None, :]

    def sample_test_obstacles(self, rng: np.random.Generator, n: int,
                              x_low: float, x_high: float,
                              y_low: float, y_high: float) -> np.ndarray:
        # Only y0 is free; y1 is a dummy (kept as 0.0) so downstream code that
        # expects [N, 3] still works.
        out = np.zeros((n, 3))
        out[:, 0] = rng.uniform(x_low, x_high, n)
        out[:, 1] = rng.uniform(y_low, y_high, n)
        return out

    def z_targets_from_obstacle_row(self, obstacle_row: torch.Tensor,
                                    L_demo: float,
                                    goal_proximity: float = 0.00075) -> torch.Tensor:
        # obstacle_row: [height, y0, _]; pin the bottom.
        y0 = obstacle_row[1] * L_demo
        y1 = torch.tensor(L_demo - goal_proximity,
                          dtype=obstacle_row.dtype, device=obstacle_row.device)
        # Same guard as in z_targets_from_task.
        y0 = torch.minimum(y0, y1 - 1e-3)
        return torch.stack([y0, y1])
