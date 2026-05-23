"""GPU-batched PI² with optional per-iteration trajectory plotting.

A variant of ``PI2Optimizer`` that snapshots the policy-mean trajectory
every ``plot_every`` iterations. Two display modes:

* **Static** (default): saves a single overlay plot at the end of the run.
* **Live**: opens a matplotlib window and updates it in place as PI²
  progresses. Useful for watching convergence and tuning sigma/h in real
  time. Still saves the final plot.

The optimization logic is identical to ``PI2Optimizer``; only diagnostics
are added.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.configs import BoxAvoidCostConfig, PI2Config
from core.cost_functions import (
    acc_from_pos,
    costs_to_weights,
    evaluate_box_avoid,
)
from core.dmp_wrapper import DMPWrapper
from core.pi2_optimizer import PI2Optimizer, PI2Result


@dataclass
class PI2PlotResult(PI2Result):
    snapshot_iters: np.ndarray = None
    snapshot_trajectories: np.ndarray = None
    snapshot_costs: np.ndarray = None


class PI2OptimizerPlotting(PI2Optimizer):
    """PI² with optional trajectory snapshots and overlay plot.

    Args:
        dmp, cfg, cost_cfg: same as PI2Optimizer.
        plot_every: snapshot the policy mean every N iterations.
            Set to 0 to disable plotting entirely.
        plot_dir: directory to save the final plot.
        plot_name: filename for the saved plot (no extension).
        plot_initial: snapshot iter 0 (the imitated mean) for reference.
        show_samples: overlay a random subset of exploration samples.
        n_show_samples: how many exploration samples to overlay per snapshot.
        live_plot: open a matplotlib window and update it during optimization.
            Falls back to static plotting at the end (the saved file is
            identical either way).
        live_pause: seconds to pause after each live update. Small values
            keep optimization fast; larger values (~0.1-0.5) let you read
            the plot evolve. Set to 0 for fastest live mode.
    """

    def __init__(self, dmp: DMPWrapper, cfg: PI2Config,
                 cost_cfg: BoxAvoidCostConfig,
                 plot_every: int = 0,
                 plot_dir: str | Path = ".",
                 plot_name: str = "pi2_progress",
                 plot_initial: bool = True,
                 show_samples: bool = False,
                 n_show_samples: int = 5,
                 live_plot: bool = False,
                 live_pause: float = 0.05):
        super().__init__(dmp, cfg, cost_cfg)
        self.plot_every = plot_every
        self.plot_dir = Path(plot_dir)
        self.plot_name = plot_name
        self.plot_initial = plot_initial
        self.show_samples = show_samples
        self.n_show_samples = n_show_samples
        self.live_plot = live_plot
        self.live_pause = live_pause

        # Live-plot state, set up lazily on first snapshot
        self._fig = None
        self._ax_traj = None
        self._ax_cost = None
        self._cmap = plt.cm.viridis

    # ----------------------------------------------------------- main routine
    def optimize(self, z_targets: torch.Tensor) -> PI2PlotResult:
        import time

        cfg = self.cfg
        cost_cfg = self.cost_cfg
        dmp = self.dmp

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

        snapshot_iters: list[int] = []
        snapshot_trajs: list[np.ndarray] = []
        snapshot_costs_list: list[np.ndarray] = []
        snapshot_sample_trajs: list[np.ndarray] = []

        # ---- Set up live figure if requested
        if self.live_plot and self.plot_every > 0:
            self._setup_live_figure(z_targets.cpu().numpy())

        # ---- Initial snapshot
        if self.plot_every > 0 and self.plot_initial:
            roll_init = dmp.rollout_batch(self.current_weights.unsqueeze(0))
            pos_init = roll_init["pos"]
            acc_init = acc_from_pos(pos_init, roll_init["times"])
            res_init = evaluate_box_avoid(pos_init, acc_init, z_targets, cost_cfg,
                                          x_0_y, x_goal_y, goal_pos=dmp.x_goal)
            snapshot_iters.append(0)
            snapshot_trajs.append(pos_init[0].detach().cpu().numpy())
            snapshot_costs_list.append(res_init.costs[0].detach().cpu().numpy())
            if self.show_samples:
                snapshot_sample_trajs.append(
                    np.zeros((0, pos_init.shape[1], self.n_dof)))
            if self.live_plot:
                self._update_live(snapshot_iters, snapshot_trajs,
                                  snapshot_sample_trajs, avg_costs[:0],
                                  finished=False, current_iter=0)

        t_start = time.time()
        finished = False
        ii = 0

        for ii in range(max_iters):
            # ---- 1) explore
            #print(f"#{ii}: Max current weights: {torch.max(self.current_weights):.4f}")
            samples = self.explore(self.current_weights)
            #print(f"#{ii}: Max samples: {torch.max(samples):.4f}")
            # ---- 2) batched rollout
            roll = dmp.rollout_batch(samples)
            pos = roll["pos"]
            times = roll["times"]
            acc = acc_from_pos(pos, times)
            #print(f"#{ii}: Max pos: {torch.max(pos):.4f}")
            # ---- 3) costs
            res = evaluate_box_avoid(pos, acc, z_targets, cost_cfg,
                                     x_0_y, x_goal_y, goal_pos=dmp.x_goal)
            costs_total = res.costs[:, 0]
            print(f"#{ii}: costs: {costs_total.min().item():.4f} .. {costs_total.max().item():.4f} ")
            # ---- 4) update
            w = costs_to_weights(costs_total, h=cfg.h)
            mean_new = (w.view(-1, 1, 1) * samples).sum(dim=0)
            self.current_weights = mean_new
            self.covar = (cfg.covar_decay ** 2) * self.covar

            # ---- 5) evaluate the mean
            roll_mean = dmp.rollout_batch(self.current_weights.unsqueeze(0))
            pos_m = roll_mean["pos"]
            acc_m = acc_from_pos(pos_m, roll_mean["times"])
            res_m = evaluate_box_avoid(pos_m, acc_m, z_targets, cost_cfg,
                                       x_0_y, x_goal_y, goal_pos=dmp.x_goal)

            train_data[ii] = mean_new.detach().cpu().numpy().T
            avg_costs[ii] = res_m.costs[0].detach().cpu().numpy()
            label_z[ii] = res_m.height_at_borders[0].detach().cpu().numpy()
            y_locations[ii] = res_m.y_borders[0].detach().cpu().numpy()

            # ---- 5b) snapshot
            should_snapshot = (
                self.plot_every > 0
                and ((ii + 1) % self.plot_every == 0)
            )
            if should_snapshot:
                snapshot_iters.append(ii + 1)
                snapshot_trajs.append(pos_m[0].detach().cpu().numpy())
                snapshot_costs_list.append(res_m.costs[0].detach().cpu().numpy())
                if self.show_samples:
                    n = min(self.n_show_samples, pos.shape[0])
                    idx = torch.randperm(pos.shape[0])[:n]
                    snapshot_sample_trajs.append(pos[idx].detach().cpu().numpy())
                if self.live_plot:
                    self._update_live(snapshot_iters, snapshot_trajs,
                                      snapshot_sample_trajs, avg_costs[:ii + 1],
                                      finished=False, current_iter=ii + 1)

            if res_m.finished[0].item():
                finished = True
                if self.plot_every > 0 and (not should_snapshot):
                    snapshot_iters.append(ii + 1)
                    snapshot_trajs.append(pos_m[0].detach().cpu().numpy())
                    snapshot_costs_list.append(res_m.costs[0].detach().cpu().numpy())
                    if self.show_samples:
                        snapshot_sample_trajs.append(
                            np.zeros((0, pos_m.shape[1], self.n_dof)))
                    if self.live_plot:
                        self._update_live(snapshot_iters, snapshot_trajs,
                                          snapshot_sample_trajs, avg_costs[:ii + 1],
                                          finished=True, current_iter=ii + 1)
                break

        duration = time.time() - t_start
        n_iter = ii + 1
        final_pos = pos_m[0].detach().cpu().numpy()

        # ---- Save the final plot (whether or not live was on)
        if self.plot_every > 0 and snapshot_trajs:
            self._save_final_plot(
                snapshot_iters=snapshot_iters,
                snapshot_trajs=snapshot_trajs,
                snapshot_sample_trajs=snapshot_sample_trajs,
                z_targets=z_targets.cpu().numpy(),
                avg_costs=avg_costs[:n_iter],
                finished=finished,
            )

        # Close the live figure cleanly
        if self.live_plot and self._fig is not None:
            plt.ioff()
            plt.close(self._fig)
            self._fig = None

        return PI2PlotResult(
            train_data=train_data[:n_iter],
            label_z=label_z[:n_iter],
            avg_costs=avg_costs[:n_iter],
            y_locations=y_locations[:n_iter],
            final_x_track=final_pos,
            n_iter=n_iter,
            duration=duration,
            finished=finished,
            snapshot_iters=np.asarray(snapshot_iters),
            snapshot_trajectories=np.stack(snapshot_trajs) if snapshot_trajs else None,
            snapshot_costs=np.stack(snapshot_costs_list) if snapshot_costs_list else None,
        )

    # --------------------------------------------------------------- live UI
    def _setup_live_figure(self, z_targets: np.ndarray) -> None:
        """Initialize the live figure once at the start of a run."""
        plt.ion()
        self._fig, (self._ax_traj, self._ax_cost) = plt.subplots(
            1, 2, figsize=(14, 6))

        self._ax_traj.set_xlabel("y")
        self._ax_traj.set_ylabel("z")
        self._ax_traj.set_title("Policy-mean trajectory (live)")
        self._ax_traj.grid(True, alpha=0.3)
        self._ax_traj.set_aspect("equal", adjustable="datalim")

        # Pre-draw obstacle borders (these don't change)
        for zt in z_targets:
            self._ax_traj.axvline(zt, color="orange", linestyle="--",
                                  lw=1.5, alpha=0.7)

        self._ax_cost.set_xlabel("iteration")
        self._ax_cost.set_ylabel("cost")
        self._ax_cost.set_title("Per-iteration cost")
        self._ax_cost.grid(True, alpha=0.3)
        self._ax_cost.set_yscale("symlog", linthresh=1e-3)

        plt.tight_layout()
        self._fig.canvas.draw()
        plt.pause(0.001)

    def _update_live(self,
                     snapshot_iters: list[int],
                     snapshot_trajs: list[np.ndarray],
                     snapshot_sample_trajs: list[np.ndarray],
                     avg_costs: np.ndarray,
                     finished: bool,
                     current_iter: int) -> None:
        """Redraw the live figure with current snapshots."""
        if self._fig is None:
            return

        # Check if window was closed by user — bail out gracefully
        if not plt.fignum_exists(self._fig.number):
            self.live_plot = False  # disable future updates
            self._fig = None
            return

        # ---- Trajectory panel: clear lines but keep obstacle markers
        # Easier than tracking individual line handles; only a few lines.
        for line in list(self._ax_traj.lines):
            line.remove()
        # axvlines are stored under .lines too in older mpl, so re-add them
        # by storing a reference at setup. Simpler: just clear() and re-setup.
        self._ax_traj.clear()
        self._ax_traj.set_xlabel("y")
        self._ax_traj.set_ylabel("z")
        self._ax_traj.grid(True, alpha=0.3)
        self._ax_traj.set_aspect("equal", adjustable="datalim")

        n_snaps = len(snapshot_trajs)

        # Exploration samples (faint)
        if self.show_samples and snapshot_sample_trajs:
            for i, sample_trajs in enumerate(snapshot_sample_trajs):
                if sample_trajs.shape[0] == 0:
                    continue
                color = self._cmap(i / max(n_snaps - 1, 1))
                for traj in sample_trajs:
                    self._ax_traj.plot(traj[:, 1], traj[:, 2], "-",
                                       lw=0.5, color=color, alpha=0.25)

        # Policy-mean trajectories
        for i, (it, traj) in enumerate(zip(snapshot_iters, snapshot_trajs)):
            color = self._cmap(i / max(n_snaps - 1, 1))
            is_latest = (i == n_snaps - 1)
            label = f"iter {it}" + (" ✓" if (is_latest and finished) else "")
            lw = 2.5 if is_latest else 1.2
            self._ax_traj.plot(traj[:, 1], traj[:, 2], "-", lw=lw,
                               color=color, label=label)

        # Re-draw obstacle borders after clear()
        # Read them off the cost-eval inputs we stored at setup — easiest is
        # to grab from the snapshot_trajs context. Here, we re-pull from cfg.
        # (z_targets is a Tensor in optimize(); we can't reach it from here
        # cleanly without storing it. Store on self at setup time.)
        if hasattr(self, "_live_z_targets"):
            for zt in self._live_z_targets:
                self._ax_traj.axvline(zt, color="orange", linestyle="--",
                                       lw=1.5, alpha=0.7)

        title = (f"Policy-mean trajectory  |  iter {current_iter}"
                 + (f"  |  converged ✓" if finished else ""))
        self._ax_traj.set_title(title)
        if n_snaps <= 12:  # legend gets unreadable past this
            self._ax_traj.legend(loc="best", fontsize=8, ncol=2)

        # ---- Cost panel: redraw from scratch (cheap)
        self._ax_cost.clear()
        self._ax_cost.set_xlabel("iteration")
        self._ax_cost.set_ylabel("cost")
        self._ax_cost.grid(True, alpha=0.3)
        self._ax_cost.set_yscale("symlog", linthresh=1e-3)
        if len(avg_costs) > 0:
            self._ax_cost.plot(avg_costs[:, 0], "b-", lw=1.5, label="total")
            if avg_costs.shape[1] >= 5:
                self._ax_cost.plot(avg_costs[:, 1], "g-", lw=1, alpha=0.7, label="c1")
                self._ax_cost.plot(avg_costs[:, 2], "r-", lw=1, alpha=0.7, label="c2")
                self._ax_cost.plot(avg_costs[:, 3], "m-", lw=1, alpha=0.7, label="c3")
                self._ax_cost.plot(avg_costs[:, 4], "c-", lw=1, alpha=0.7, label="c4")
            for it in snapshot_iters[1:]:
                if it - 1 < len(avg_costs):
                    self._ax_cost.axvline(it - 1, color="gray",
                                          linestyle=":", lw=0.8, alpha=0.5)
            self._ax_cost.legend(fontsize=8)
        self._ax_cost.set_title(f"Cost  |  current: "
                                 f"{avg_costs[-1, 0]:.4f}" if len(avg_costs) else "Cost")

        # Pump the event loop
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        if self.live_pause > 0:
            plt.pause(self.live_pause)

    def _save_final_plot(self,
                         snapshot_iters: list[int],
                         snapshot_trajs: list[np.ndarray],
                         snapshot_sample_trajs: list[np.ndarray],
                         z_targets: np.ndarray,
                         avg_costs: np.ndarray,
                         finished: bool) -> None:
        """Render the final overlay to disk (used in both live and static modes)."""
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        cmap = self._cmap

        # ---- Left panel
        ax = axes[0]
        n_snaps = len(snapshot_trajs)

        if self.show_samples and snapshot_sample_trajs:
            for i, sample_trajs in enumerate(snapshot_sample_trajs):
                if sample_trajs.shape[0] == 0:
                    continue
                color = cmap(i / max(n_snaps - 1, 1))
                for traj in sample_trajs:
                    ax.plot(traj[:, 1], traj[:, 2], "-", lw=0.5,
                            color=color, alpha=0.25)

        for i, (it, traj) in enumerate(zip(snapshot_iters, snapshot_trajs)):
            color = cmap(i / max(n_snaps - 1, 1))
            is_latest = (i == n_snaps - 1)
            label = f"iter {it}" + (" ✓" if (is_latest and finished) else "")
            lw = 2.5 if is_latest else 1.5
            ax.plot(traj[:, 1], traj[:, 2], "-", lw=lw, color=color, label=label)

        ax.plot(snapshot_trajs[0][0, 1], snapshot_trajs[0][0, 2], "ko",
                ms=8, label="start")
        ax.plot(snapshot_trajs[-1][-1, 1], snapshot_trajs[-1][-1, 2], "k^",
                ms=8, label="end")

        for zt in z_targets:
            ax.axvline(zt, color="orange", linestyle="--", lw=1.5, alpha=0.7)

        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_title(f"Policy-mean trajectory over PI² iterations "
                     f"(every {self.plot_every}; "
                     f"{'converged' if finished else 'maxed out'})")
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

        # ---- Right panel
        ax = axes[1]
        ax.plot(avg_costs[:, 0], "b-", lw=1.5, label="total cost")
        if avg_costs.shape[1] >= 5:
            ax.plot(avg_costs[:, 1], "g-", lw=1, alpha=0.7, label="component 1")
            ax.plot(avg_costs[:, 2], "r-", lw=1, alpha=0.7, label="component 2")
            ax.plot(avg_costs[:, 3], "m-", lw=1, alpha=0.7, label="component 3")
            ax.plot(avg_costs[:, 4], "c-", lw=1, alpha=0.7, label="component 4")
        for it in snapshot_iters[1:]:
            if it - 1 < len(avg_costs):
                ax.axvline(it - 1, color="gray", linestyle=":",
                            lw=0.8, alpha=0.5)
        ax.set_xlabel("iteration")
        ax.set_ylabel("cost")
        ax.set_title("Per-iteration cost (mean policy)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("symlog", linthresh=1e-3)

        plt.tight_layout()
        out_path = self.plot_dir / f"{self.plot_name}.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[pi2_plot] saved progress plot to {out_path}")