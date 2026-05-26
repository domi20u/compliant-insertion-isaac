"""Thin wrapper around MP_PyTorch's DMP that exposes the operations PI² needs.

Specifically, this module provides:

1. **Imitation fit** via ridge regression on the forcing-term targets (since
   ``mp_pytorch.mp.dmp.DMP.learn_mp_params_from_trajs`` raises NotImplementedError).
2. **Batched rollout** of N candidate weight vectors in a single forward pass
   on the GPU.
3. A clean weights API (`set_weights`/`get_weights`) that hides MP_PyTorch's
   flat ``params`` layout (which interleaves weights and goal per DOF).
4. A ``use_rotodilatation`` flag that is wired through but currently asserts
   off — placeholder for when the MP_PyTorch fork gains rotodilatation.
"""
from __future__ import annotations

import numpy as np
import torch
from addict import Dict as AddictDict
from mp_pytorch.mp import MPFactory

from configs.configs import DMPConfig


class DMPWrapper:
    """A batched DMP wrapper backed by MP_PyTorch."""

    def __init__(self, cfg: DMPConfig, device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        if cfg.use_rotodilatation:
            raise NotImplementedError(
                "Rotodilatation is not yet implemented in the MP_PyTorch fork. "
                "Set DMPConfig.use_rotodilatation=False."
            )

        # Loaded by ``imitate_path``
        self.tau: float | None = None
        self.x_0: torch.Tensor | None = None
        self.x_goal: torch.Tensor | None = None
        self.times_demo: torch.Tensor | None = None
        self.demo: torch.Tensor | None = None
        self.learned_L: float | None = None
        self.weights_init: torch.Tensor | None = None  # [num_dof, num_basis]
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

        config.mp_args.num_basis = self.num_basis
        config.mp_args.basis_bandwidth_factor = self.cfg.basis_bandwidth_factor
        config.mp_args.num_basis_outside = 0
        config.mp_args.alpha = self.cfg.K
        config.mp_args.alpha_phase = self.cfg.alpha_phase
        config.mp_args.dt = float(self.tau / self.times_demo.shape[-1])
        config.mp_args.weights_scale = 1.0
        config.mp_args.goal_scale = 1.0
        config.mp_args.relative_goal = False

        if self.cfg.use_improved:
            config.mp_type = 'transformation_dmp' #"transformation_dmp" #'improved_dmp'
            # Resolve rescale from either flag (rescale takes precedence)
            rescale = self.cfg.rescale
            if rescale is None and self.cfg.use_rotodilatation:
                rescale = "rotodilatation"
            config.mp_args.rescale = rescale
            # Learned endpoints — only meaningful when rescale != None.
            # We set these after imitation, so they may be None on first build.
            if self.x_0 is not None and self.x_goal is not None and rescale is not None:
                config.mp_args.learned_start = self.x_0
                config.mp_args.learned_goal = self.x_goal
        else:
            config.mp_type = "dmp"

        self.mp = MPFactory.init_mp(**config.to_dict())

        if self.cfg.use_improved and self.cfg.rescale is not None \
                and self.x_0 is not None and self.x_goal is not None:
            self.mp.set_learned_endpoints(self.x_0, self.x_goal)

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
        self.x_0 = torch.tensor(xs[0], dtype=self.dtype, device=self.device)
        self.x_goal = torch.tensor(xs[-1], dtype=self.dtype, device=self.device)
        self.learned_L = float(np.linalg.norm(xs[-1] - xs[0]))

        # batch dim = 1 for fitting
        self.times_demo = torch.tensor(t, dtype=self.dtype,
                                       device=self.device).unsqueeze(0)
        self.demo = torch.tensor(xs, dtype=self.dtype,
                                 device=self.device).unsqueeze(0)

        self._build_mp(batch=1)

        # If rotodilatation is on, register the learned endpoints with the MP
        # AFTER the build (the build may not have had them yet).
        if self.cfg.use_improved and self.cfg.rescale is not None:
            self.mp.set_learned_endpoints(self.x_0, self.x_goal)

        with torch.no_grad():
            if self.cfg.use_improved:
                weights_flat = self._fit_weights_ridge_improved(
                    self.times_demo, self.demo, reg=self.cfg.ridge_reg)
            else:
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

    def _fit_weights_ridge_improved(self, times, demos, reg):
        """Ridge fit for the ImprovedDMP formulation.

        Differences from the vanilla _fit_weights_ridge:
        - regressor Phi uses NORMALIZED basis: psi / sum(psi), times phase s.
        - target adds the transient term +alpha*beta*(g - x_0)*s.
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

        x_0  = demos[:, 0, :]
        goal = demos[:, -1, :]

        # Normalized basis * phase  ->  Phi
        psi = self.mp.basis_gn.basis(times)                     # [B, T, K]
        s   = self.mp.phase_gn.phase(times)                     # [B, T]
        psi_sum  = psi.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        psi_norm = psi / psi_sum
        Phi = psi_norm * s.unsqueeze(-1)                        # [B, T, K]

        # Improved-formulation target: vanilla target + transient cancellation
        transient = self.mp.alpha * self.mp.beta * \
                    (goal - x_0).unsqueeze(1) * s.unsqueeze(-1)  # [B, T, D]
        f_target = acc_s \
                - self.mp.alpha * (self.mp.beta * (goal.unsqueeze(1) - demos)
                                    - vel_s) \
                + transient                                   # [B, T, D]

        K = Phi.shape[-1]
        eye = torch.eye(K, dtype=self.dtype, device=self.device) * reg
        A   = Phi.transpose(-1, -2) @ Phi + eye                  # [B, K, K]
        rhs = Phi.transpose(-1, -2) @ f_target                   # [B, K, D]
        weights = torch.linalg.solve(A, rhs).transpose(-1, -2)   # [B, D, K]

        return self._weights_to_params(weights, goal)
    
    def _fit_weights_ridge(self, times: torch.Tensor, demos: torch.Tensor,
                           reg: float) -> torch.Tensor:
        """Ridge fit. Returns flat params [B, D*(K+1)] ready for MP_PyTorch."""
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
                      n_timesteps: int | None = None) -> dict:
        """Roll out N DMPs in parallel on GPU.

        Args:
            weights_batch: [N, num_dof, num_basis] tensor.
            n_timesteps:   number of points in the rollout. Defaults to demo length.

        Returns:
            dict with keys 'pos' [N, T, D], 'vel' [N, T, D], 'times' [N, T].
        """
        N = weights_batch.shape[0]
        T = n_timesteps or self.times_demo.shape[-1]

        # Build/rebuild the MP for this batch size
        self._build_mp(batch=N)

        # Goal stays at the demo goal; PI² perturbs only weights, not goal
        goal = self.x_goal.unsqueeze(0).expand(N, -1).contiguous()
        params = self._weights_to_params(weights_batch, goal)

        # Times, init conditions, all batched
        times = torch.linspace(0.0, self.tau, T, dtype=self.dtype,
                               device=self.device)
        times = times.unsqueeze(0).expand(N, -1).contiguous()
        init_pos = self.x_0.unsqueeze(0).expand(N, -1).contiguous()
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
