"""Typed configuration objects for the DMP-PI2 obstacle-avoidance pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# DMP configuration
# ---------------------------------------------------------------------------

@dataclass
class DMPConfig:
    """Parameters for the MP_PyTorch DMP."""

    n_dmps: int = 3
    n_bfs: int = 16 # avoid 10                       # 'extra' basis functions; total = n_bfs + 1
    K: float = 25.0                       # spring constant -> alpha (beta = alpha/4)
    alpha_phase: float = 2.0              # canonical-system decay
    basis_bandwidth_factor: float = 0.5   # ~ dmp_pp's h_=0.5
    trajectory_file: str = "poly_traj_3.csv"
    ridge_reg: float = 1e-6
    use_improved: bool = False                 # use ImprovedDMP forward model
    rescale: str | None = None                # 'rotodilatation' | 'diagonal' | None
    # If True, sets rescale='rotodilatation' (only honored when rescale is None
    # AND use_improved is True). Kept as a convenience flag; prefer setting
    # ``rescale`` directly.
    use_rotodilatation: bool = False


# ---------------------------------------------------------------------------
# PI2 configuration
# ---------------------------------------------------------------------------

@dataclass
class PI2Config:
    """PI² policy-improvement parameters."""

    max_iters: int = 1000
    n_samples: int = 20 #avoid 10                   # >> the original 10; GPU makes this cheap
    h: float = 10.0                       # cost-to-weight temperature
    covar_decay: float = 1.0
    sigma_min: float = 0.01 # avoid 0.2
    sigma_max: float = 1.2 # avoid 5.2
    sigma_steepness: float = 1.0
    # which DOFs to perturb (1 = perturb, 0 = freeze) — order matches DMP DOFs
    param_perturb: Tuple[int, ...] = (0, 1, 1)


# ---------------------------------------------------------------------------
# Cost / task configuration
# ---------------------------------------------------------------------------

@dataclass
class BoxAvoidCostConfig:
    """Continuous-box obstacle-avoidance cost (matches old evaluate_rollout_continuous_box_acc)."""

    max_val: float = 0.05
    bounds: Tuple[float, float] = (0.1, 0.00075)  # loose bounds 0.05, 0.05
    table_penalty: float = 10.0
    smoothness_initial_acc: float = 1 / 100
    smoothness_jerk: float = 1 / 20
    # Penalty on |pos[-1] - goal|; set > 0 if your DMP doesn't naturally
    # converge to the goal (current MP_PyTorch upstream does NOT). Disable
    # this once you patch goal_convergence into your fork.
    goal_penalty: float = 2.0


@dataclass
class Insert2pCostConfig:
    """Two-parameter insertion cost.

    Geometric reinterpretation vs. box-avoidance:
      * ``z_targets[0]`` is the slot opening y-location (free task parameter).
      * ``z_targets[1]`` is pinned by the scenario to ``L_demo - goal_proximity``,
        i.e. just shy of the goal y. This forces the cost-relevant segment
        [id0, id1] to span the descent into the slot.

    The asymmetric ``bounds`` envelope (wide on approach, tight near the goal)
    plus ``verticality_penalty`` on lateral motion in the final descent
    [id1 .. T-1] shape a vertical insertion approach.
    """

    max_val: float = 0.05
    # (bL, bR) = (loose left envelope, tight right envelope) -> vertical descent
    bounds: Tuple[float, float] = (0.1, 0.00075)
    table_penalty: float = 10.0
    smoothness_initial_acc: float = 1 / 100
    smoothness_jerk: float = 1 / 20
    goal_penalty: float = 2.0
    # Penalty on lateral (x,y) motion during the final descent segment.
    # 0 reproduces box-avoid behaviour; >0 enforces a vertical approach.
    verticality_penalty: float = 0.0
    # Distance from the goal at which the slot bottom is pinned. The scenario
    # passes this into the cost so both stay in sync.
    goal_proximity: float = 0.00075


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class DataGenConfig:
    """Top-level config for one PI² data-generation run (box-avoidance defaults)."""

    dmp: DMPConfig = field(default_factory=DMPConfig)
    pi2: PI2Config = field(default_factory=PI2Config)
    cost: BoxAvoidCostConfig = field(default_factory=BoxAvoidCostConfig)

    n_runs: int = 10
    # Range used to sample the FREE y-locations of the task. For box-avoid,
    # two y's are sampled per run; for insertion, only the slot-opening y.
    z_height_low: float = 0.005
    z_height_range: float = 0.14
    output_dir: Path = Path("results/default")
    device: str = "cuda"
    seed: int = 0


@dataclass
class InsertionDataGenConfig:
    """Top-level config for one PI² insertion data-generation run.

    Same fields as ``DataGenConfig`` but with insertion-flavoured defaults:
    a denser basis (20 BFs), a wider sigma window for exploration, and the
    insertion cost config.
    """

    dmp: DMPConfig = field(default_factory=lambda: DMPConfig(n_bfs=14, basis_bandwidth_factor=0.5, alpha_phase=2.0, K=25.0))
    pi2: PI2Config = field(default_factory=lambda: PI2Config(
        sigma_min=0.01, sigma_max=1.2, max_iters=2000))
    cost: Insert2pCostConfig = field(default_factory=Insert2pCostConfig)

    n_runs: int = 10
    z_height_low: float = 0.005
    z_height_range: float = 0.14
    output_dir: Path = Path("results/default")
    device: str = "cuda"
    seed: int = 0


# ---------------------------------------------------------------------------
# NN training
# ---------------------------------------------------------------------------

@dataclass
class NNTrainConfig:
    """Hyper-parameters for the MLP policy that maps obstacle features -> DMP weights.

    Defaults are tuned for box-avoidance (3 inputs, 11 basis). For insertion
    you'd typically pass ``--n-inputs 2 --n-basis 20`` (or rely on the
    scenario-specific config defaults wired up in ``scripts/train_policy.py``).
    """

    n_inputs: int = 2                     # obstacle features
    n_basis: int = 15                     # n_bfs + 1
    n_dims: int = 2                       # number of perturbed DMP dimensions
    hidden_sizes: Tuple[int, ...] = (256, 512)
    learning_rate: float = 5e-4
    batch_size: int = 16
    num_epochs: int = 100
    val_split: float = 0.2
    samples_per_run: int = 1000           # capped at min(iters_per_run)
    L_demo: float = 0.15
    data_dir: Path = Path("results/default")
    model_name: str = "model"
    device: str = "cuda"
    seed: int = 0


# ---------------------------------------------------------------------------
# Testing / evaluation
# ---------------------------------------------------------------------------

@dataclass
class TestConfig:
    """Parameters for evaluating a trained NN policy.

    Like ``NNTrainConfig``, defaults are box-avoidance. For insertion, the
    scenario adapter (``scripts/test_policy.py``) overrides ``n_inputs``,
    ``n_basis``, and sampling ranges so the same script works for both tasks.
    """

    n_inputs: int = 2
    n_basis: int = 15
    n_dims: int = 2
    L_demo: float = 0.15

    n_tests: int = 10
    n_tests_up: int = 10
    ins_offset: float = 0.0

    max_in1: float = 0.33
    min_in2: float = 0.03
    max_in3: float = 0.96

    extrapolate_max_in1: float | None = None  # None -> in-distribution test

    data_dir: Path = Path("results/default")
    model_name: str = "model"
    trajectory_file: str = "poly_traj_3.csv"
    device: str = "cuda"
    seed: int = 42
