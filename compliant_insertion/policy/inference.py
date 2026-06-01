"""Runtime wrapper to deploy a trained diffusion policy in the sim loop.

Loads a checkpoint saved by ``scripts/train/train_diffusion_policy.py`` and
exposes a smoothed receding-horizon controller.

By default it uses **temporal ensembling** (à la ACT): every control tick it
re-samples an action chunk (cheap DDIM) and the action executed for the current
timestep is a recency-weighted average of all overlapping predictions that cover
it. This removes the chunk-boundary jerk and per-step diffusion noise that
otherwise lurch the stiff OSC setpoint and shake the friction-held peg loose. An
optional per-tick step clamp bounds how far the commanded TCP target can jump.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .diffusion_policy import (
    DPConfig, DiffusionPolicy, DDPM, make_image_encoder, preprocess_images,
)


class DiffusionPolicyRunner:
    def __init__(self, ckpt_path, device="cuda", temporal_ensemble=True,
                 ddim_steps=8, ensemble_weight=0.05, smooth_beta=0.5,
                 max_step_m=0.0):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cfg = DPConfig(**ckpt["cfg"])
        self.model = DiffusionPolicy(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.ddpm = DDPM(self.cfg.num_diffusion_iters, device=self.device)
        self.encoder = make_image_encoder(self.device)
        s = ckpt["stats"]
        self.stats = {
            k: (v.detach().clone() if torch.is_tensor(v)
                else torch.tensor(np.asarray(v))).to(self.device, torch.float32)
            for k, v in s.items()}
        self.te = temporal_ensemble
        self.ddim_steps = ddim_steps
        self.ens_w = ensemble_weight        # recency weight (newer preds higher)
        self.smooth_beta = smooth_beta      # EMA low-pass on the output command
        self.max_step_m = max_step_m
        self._obs = deque(maxlen=self.cfg.n_obs_steps)
        self.reset()

    def reset(self):
        self._obs.clear()
        self._queue: list[np.ndarray] = []
        self._buf: dict[int, list[np.ndarray]] = {}   # abs timestep -> [actions]
        self._t = 0
        self._last_cmd = None

    @torch.no_grad()
    def _per_step(self, wrist_uint8, external_uint8, lowdim):
        feats = []
        for img in (wrist_uint8, external_uint8):
            t = torch.as_tensor(img, device=self.device)
            if t.shape[-1] == 4:
                t = t[..., :3]
            feats.append(self.encoder(preprocess_images(t.unsqueeze(0)))[0])
        img_feat = torch.cat(feats)
        img_feat = (img_feat - self.stats["img_mean"]) / self.stats["img_std"]
        low = torch.as_tensor(lowdim, device=self.device, dtype=torch.float32)
        low = (low - self.stats["low_mean"]) / self.stats["low_std"]
        return torch.cat([img_feat, low])

    @torch.no_grad()
    def _sample_chunk(self):
        while len(self._obs) < self.cfg.n_obs_steps:
            self._obs.appendleft(self._obs[0])
        obs = torch.stack(list(self._obs)).unsqueeze(0)
        steps = self.ddim_steps if self.te else None
        pred = self.model.predict(obs, self.ddpm, ddim_steps=steps)[0]   # [ph,3] norm
        amin, amax = self.stats["act_min"], self.stats["act_max"]
        pred = (pred + 1) / 2 * (amax - amin + 1e-9) + amin
        return pred.cpu().numpy()

    def _post(self, action):
        """EMA low-pass + optional per-tick step clamp on the output command."""
        if self._last_cmd is not None:
            if self.smooth_beta > 0:
                action = self.smooth_beta * self._last_cmd + (1 - self.smooth_beta) * action
            if self.max_step_m > 0:
                d = action - self._last_cmd
                n = float(np.linalg.norm(d))
                if n > self.max_step_m:
                    action = self._last_cmd + d * (self.max_step_m / n)
        self._last_cmd = action
        return action

    @torch.no_grad()
    def act(self, wrist_uint8, external_uint8, lowdim) -> np.ndarray:
        """Return the next base-frame TCP position target [3] (numpy)."""
        self._obs.append(self._per_step(wrist_uint8, external_uint8, lowdim))

        if not self.te:                          # plain chunked execution
            if not self._queue:
                chunk = self._sample_chunk()
                self._queue = list(chunk[:self.cfg.n_action_steps])
            return self._post(self._queue.pop(0))

        # Temporal ensembling: re-sample each tick, blend overlapping chunks.
        chunk = self._sample_chunk()             # [ph, 3]
        for k in range(self.cfg.pred_horizon):
            self._buf.setdefault(self._t + k, []).append(chunk[k])
        preds = self._buf.pop(self._t, [chunk[0]])
        n = len(preds)
        w = np.exp(self.ens_w * np.arange(n))    # last (newest) weighted highest
        w /= w.sum()
        action = np.einsum("i,ij->j", w.astype(np.float32), np.stack(preds))
        for key in [k for k in self._buf if k < self._t]:
            del self._buf[key]
        self._t += 1
        return self._post(action)
