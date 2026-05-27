"""Thin wrapper around MP_PyTorch's DMP that exposes the operations PI² needs.

Targets the *flat* DMP (mp_type='dmp') only. The fork adds two features to
that class which this wrapper exposes:

1. **Rotation invariance** via the ``rescale`` flag and
   ``set_learned_endpoints``. When on, the forcing term is roto-dilatated at
   rollout time so the demo's shape follows the runtime (init_pos -> goal)
   displacement.

2. **Step-wise execution** via ``reset_state(...)`` and ``step(dt)`` for
   closed-loop or early-terminating rollouts.

The wrapper itself adds:

- Imitation fit via ridge regression on the forcing-term targets (upstream's
  ``learn_mp_params_from_trajs`` raises NotImplementedError).
- Batched rollout of N candidate weight vectors in a single forward pass.
- A clean weights API (``set_weights``/``get_weights``) that hides MP_PyTorch's
  flat ``params`` layout (which interleaves weights and goal per DOF).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from addict import Dict as AddictDict
from mp_pytorch.mp import MPFactory

from configs.configs import DMPConfig


class DMPWrapper:
    """A batched DMP wrapper backed by MP_PyTorch's flat DMP."""

    def __init__(self, cfg: DMPConfig, device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        # Resolve the rescale mode from the config. ``cfg.rescale`` takes
        # precedence; for backward compatibility ``cfg.use_rotodilatation``
        # turns it on with the default mode if rescale is not set.
        rescale = getattr(cfg, "rescale", None)
        if rescale is None and getattr(cfg, "use_rotodilatation", False):
            rescale = "rotodilatation"
        if rescale not in (None, "rotodilatation", "rotodilatation_xy"):
            raise ValueError(
                f"DMPConfig.rescale must be None, 'rotodilatation', or "
                f"'rotodilatation_xy' (got {rescale!r}).")
        self._rescale = rescale

        # Loaded by ``imitate_path``
        self.tau: Optional[float] = None
        # x_0 / x_goal are the *runtime* endpoints — what the next rollout
        # will start/end at. They are initialized from the demo by
        # ``imitate_path`` but callers are free (and expected) to overwrite
        # them with task-specific values before each rollout.
        self.x_0: Optional[torch.Tensor] = None
        self.x_goal: Optional[torch.Tensor] = None
        # x_0_demo / x_goal_demo are the *learned* endpoints — fixed once
        # at imitation time and used by rotodilatation to compute the
        # source displacement. They MUST NOT be overwritten by callers;
        # touching them invalidates the forcing-term rescaling.
        self.x_0_demo: Optional[torch.Tensor] = None
        self.x_goal_demo: Optional[torch.Tensor] = None
        self.times_demo: Optional[torch.Tensor] = None
        self.demo: Optional[torch.Tensor] = None
        self.learned_L: Optional[float] = None
        self.weights_init: Optional[torch.Tensor] = None  # [num_dof, num_basis]
        self.mp = None  # MP_PyTorch DMP, built lazily by _build_mp

    # ------------------------------------------------------------------ utils
    @property
    def num_basis(self) -> int:
        # dmp_pp uses n_bfs+1 basis functions
        return self.cfg.n_bfs + 1

    @property
    def num_dof(self) -> int:
        return self.cfg.n_dmps

    @property
    def n_bfs_p(self) -> int:
        """Compatibility alias for the old PI2 API: 'plus one' basis count."""
        return self.num_basis

    # --------------------------------------------------------- internal build
    def _build_mp(self, batch: int):
        """(Re)create the underlying MP_PyTorch DMP with a given batch size.

        MP_PyTorch's add_dim is fixed once update_inputs is called; rather than
        fight it, we simply rebuild when the batch shape changes.
        """
        config = AddictDict()
        config.num_dof = self.num_dof
        config.tau = float(self.tau)
        config.learn_tau = False
        config.learn_delay = False
        config.device = str(self.device)
        config.dtype = self.dtype

        config.mp_type = "dmp"
        config.mp_args.num_basis = self.num_basis
        config.mp_args.basis_bandwidth_factor = self.cfg.basis_bandwidth_factor
        config.mp_args.num_basis_outside = 0
        config.mp_args.alpha = self.cfg.K
        config.mp_args.alpha_phase = self.cfg.alpha_phase
        config.mp_args.dt = float(self.tau / self.times_demo.shape[-1])
        config.mp_args.weights_scale = 1.0
        config.mp_args.goal_scale = 1.0
        if self._rescale is not None:
            config.mp_args.rescale = self._rescale

        self.mp = MPFactory.init_mp(**config.to_dict())

        # Register the *demo* endpoints (the rotodilatation source frame).
        # These are stored in x_0_demo / x_goal_demo by ``imitate_path``
        # and are deliberately kept separate from x_0 / x_goal so the
        # caller can overwrite the runtime endpoints freely between
        # rollouts without invalidating rotodilatation.
        if self._rescale is not None \
                and self.x_0_demo is not None and self.x_goal_demo is not None:
            self.mp.set_learned_endpoints(self.x_0_demo, self.x_goal_demo)

    # ---------------------------------------------------------------- imitate
    def imitate_path(self, trajectory_file: str) -> torch.Tensor:
        """Load a CSV demo and ridge-regress DMP weights to it.

        Format (matches dmp_pp): columns [t, x..., dx..., ddx...]

        Returns:
            weights_init: [num_dof, num_basis] tensor on the configured device.
        """
        traj = np.loadtxt(trajectory_file, delimiter=',')
        n_dof = self.num_dof

        t = traj[:, 0]
        xs = traj[:, 1:1 + n_dof]

        self.tau = float(t[-1] - t[0])

        # Store the demo's endpoints in the demo-only slots. These define
        # the rotodilatation source frame and must not be modified later.
        self.x_0_demo = torch.tensor(xs[0], dtype=self.dtype, device=self.device)
        self.x_goal_demo = torch.tensor(xs[-1], dtype=self.dtype, device=self.device)
        # Initialize the runtime endpoints to the demo's so a default
        # rollout reproduces the demo. Callers override these freely.
        self.x_0 = self.x_0_demo.clone()
        self.x_goal = self.x_goal_demo.clone()
        self.learned_L = float(np.linalg.norm(xs[-1] - xs[0]))

        # batch dim = 1 for fitting
        self.times_demo = torch.tensor(t, dtype=self.dtype,
                                       device=self.device).unsqueeze(0)
        self.demo = torch.tensor(xs, dtype=self.dtype,
                                 device=self.device).unsqueeze(0)

        self._build_mp(batch=1)

        # _build_mp already registered the demo endpoints via x_0_demo /
        # x_goal_demo, but they may have been None on a prior partial
        # init. Re-register defensively so this is robust to call order.
        if self._rescale is not None:
            self.mp.set_learned_endpoints(self.x_0_demo, self.x_goal_demo)

        with torch.no_grad():
            weights_flat = self._fit_weights_ridge(
                self.times_demo, self.demo, reg=self.cfg.ridge_reg)

        weights_init = self._params_to_weights(weights_flat)[0]
        self.weights_init = weights_init.contiguous()
        return self.weights_init

    # ------------------------------------------------- weight <-> param tools
    def _params_to_weights(self, params: torch.Tensor) -> torch.Tensor:
        """Flat MP_PyTorch params [B, D*(K+1)] -> [B, D, K] weights."""
        B = params.shape[0]
        D, K = self.num_dof, self.num_basis
        wg = params.reshape(B, D, K + 1)
        return wg[..., :-1]

    def _weights_to_params(self, weights: torch.Tensor,
                           goal: torch.Tensor) -> torch.Tensor:
        """[B, D, K] + [B, D] -> flat [B, D*(K+1)] params (with internal scale)."""
        B, D, K = weights.shape
        wg = torch.cat([weights, goal.unsqueeze(-1)], dim=-1)  # [B, D, K+1]

        # MP_PyTorch multiplies params by weights_goal_scale internally;
        # divide here so the rollout uses the regressed weights as-is.
        wgs = self.mp.weights_goal_scale.to(weights.device)    # [K+1]
        wg = wg / wgs.view(1, 1, K + 1)
        return wg.reshape(B, D * (K + 1))

    def _fit_weights_ridge(self, times: torch.Tensor, demos: torch.Tensor,
                           reg: float) -> torch.Tensor:
        """Ridge fit. Returns flat params [B, D*(K+1)] ready for MP_PyTorch.

        Operates in the learned frame: when rotodilatation is on, the
        rollout-side transform is identity at imitation time (because the
        runtime endpoints equal the learned ones), so the regression target
        is unchanged.
        """
        B, T, D = demos.shape
        tau = self.mp.phase_gn.tau
        tau_b = tau.expand(B) if tau.dim() == 0 else tau

        dt = (times[:, 1:] - times[:, :-1]).unsqueeze(-1)
        vel = torch.zeros_like(demos)
        vel[:, :-1] = (demos[:, 1:] - demos[:, :-1]) / dt
        vel[:, -1] = vel[:, -2]
        acc = torch.zeros_like(demos)
        acc[:, :-1] = (vel[:, 1:] - vel[:, :-1]) / dt
        acc[:, -1] = acc[:, -2]

        tau_view = tau_b.view(B, 1, 1)
        vel_s = vel * tau_view
        acc_s = acc * (tau_view ** 2)

        goal = demos[:, -1, :]
        f_target = acc_s - self.mp.alpha * (
            self.mp.beta * (goal.unsqueeze(1) - demos) - vel_s)

        basis_vals = self.mp.basis_gn.basis(times)            # [B, T, K]
        canonical_x = self.mp.phase_gn.phase(times)           # [B, T]
        Phi = canonical_x.unsqueeze(-1) * basis_vals          # [B, T, K]

        K = Phi.shape[-1]
        eye = torch.eye(K, dtype=self.dtype, device=self.device) * reg
        # batched normal equations: (PhiᵀPhi + λI)w = Phiᵀf
        A = Phi.transpose(-1, -2) @ Phi + eye                 # [B, K, K]
        rhs = Phi.transpose(-1, -2) @ f_target                # [B, K, D]
        weights = torch.linalg.solve(A, rhs).transpose(-1, -2)  # [B, D, K]

        return self._weights_to_params(weights, goal)

    # ----------------------------------------------------------- batch rollout
    def rollout_batch(self, weights_batch: torch.Tensor,
                      n_timesteps: Optional[int] = None,
                      x_0: Optional[torch.Tensor] = None,
                      x_goal: Optional[torch.Tensor] = None) -> dict:
        """Roll out N DMPs in parallel on GPU.

        Args:
            weights_batch: [N, num_dof, num_basis] tensor.
            n_timesteps:   number of points in the rollout. Defaults to
                demo length.
            x_0:           runtime start [num_dof] or [N, num_dof].
                Defaults to the learned start.
            x_goal:        runtime goal  [num_dof] or [N, num_dof].
                Defaults to the learned goal.

        Returns:
            dict with keys 'pos' [N, T, D], 'vel' [N, T, D], 'times' [N, T].

        When rotodilatation is enabled and x_0/x_goal differ from the
        learned endpoints, the trajectory shape is rotated and scaled
        accordingly.
        """
        N = weights_batch.shape[0]
        T = n_timesteps or self.times_demo.shape[-1]

        # Resolve runtime endpoints (broadcast scalar inputs to the batch).
        if x_0 is None:
            x_0 = self.x_0
        x_0 = torch.as_tensor(x_0, dtype=self.dtype, device=self.device)
        if x_0.ndim == 1:
            x_0 = x_0.unsqueeze(0).expand(N, -1).contiguous()

        if x_goal is None:
            x_goal = self.x_goal
        x_goal = torch.as_tensor(x_goal, dtype=self.dtype, device=self.device)
        if x_goal.ndim == 1:
            x_goal = x_goal.unsqueeze(0).expand(N, -1).contiguous()

        # Build/rebuild the MP for this batch size
        self._build_mp(batch=N)

        # Goal field of params drives the attractor; PI² perturbs only
        # weights, so we plug x_goal in (per-batch) and leave weights as
        # the freely-varying piece.
        params = self._weights_to_params(weights_batch, x_goal)

        # Times, init conditions, all batched
        times = torch.linspace(0.0, self.tau, T, dtype=self.dtype,
                               device=self.device)
        times = times.unsqueeze(0).expand(N, -1).contiguous()
        init_pos = x_0
        init_vel = torch.zeros_like(init_pos)
        init_time = times[:, 0]

        self.mp.reset()
        self.mp.update_inputs(times=times, params=params,
                              init_time=init_time,
                              init_pos=init_pos, init_vel=init_vel)
        traj = self.mp.get_trajs(get_pos=True, get_vel=True)
        return {
            "pos": traj["pos"],            # [N, T, D]
            "vel": traj["vel"],            # [N, T, D]
            "times": times,                # [N, T]
        }

    # --------------------------------------------------------- step-wise API
    # For closed-loop control: initialize once, then call ``step(dt)`` per
    # control tick. The underlying MP is rebuilt for the requested batch
    # size on each ``reset_step`` so multiple parallel rollouts work.
    # ----------------------------------------------------------------------
    def reset_step(self,
                   weights_batch: torch.Tensor,
                   x_0: Optional[torch.Tensor] = None,
                   x_goal: Optional[torch.Tensor] = None,
                   init_vel: Optional[torch.Tensor] = None,
                   init_time: float = 0.0) -> None:
        """Initialize the step-wise integrator.

        Args:
            weights_batch: [N, num_dof, num_basis] tensor.
            x_0:    runtime start. See ``rollout_batch``.
            x_goal: runtime goal. See ``rollout_batch``.
            init_vel: optional initial velocity [num_dof] or [N, num_dof].
                Defaults to zeros.
            init_time: scalar starting time (default 0).
        """
        N = weights_batch.shape[0]

        if x_0 is None:
            x_0 = self.x_0
        x_0 = torch.as_tensor(x_0, dtype=self.dtype, device=self.device)
        if x_0.ndim == 1:
            x_0 = x_0.unsqueeze(0).expand(N, -1).contiguous()

        if x_goal is None:
            x_goal = self.x_goal
        x_goal = torch.as_tensor(x_goal, dtype=self.dtype, device=self.device)
        if x_goal.ndim == 1:
            x_goal = x_goal.unsqueeze(0).expand(N, -1).contiguous()

        if init_vel is None:
            init_vel = torch.zeros_like(x_0)
        else:
            init_vel = torch.as_tensor(init_vel, dtype=self.dtype, device=self.device)
            if init_vel.ndim == 1:
                init_vel = init_vel.unsqueeze(0).expand(N, -1).contiguous()

        self._build_mp(batch=N)
        params = self._weights_to_params(weights_batch, x_goal)
        self.mp.reset()
        self.mp.set_params(params)
        self.mp.reset_state(init_time=init_time,
                            init_pos=x_0,
                            init_vel=init_vel)

    def step(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance the DMP by one step. Returns (pos, vel) shaped [N, num_dof]."""
        if self.mp is None or self.mp.step_state is None:
            raise RuntimeError(
                "DMPWrapper.step: integrator not initialized. "
                "Call reset_step(...) first.")
        return self.mp.step(dt)
