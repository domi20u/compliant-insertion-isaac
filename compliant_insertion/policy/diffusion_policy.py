"""Self-contained CNN diffusion policy (Chi et al. 2023 style).

No external policy framework (no lerobot / diffusers) — just torch + torchvision,
so it trains in the repo's existing env without disturbing the Isaac/torch
install. Shared by the trainer (``scripts/train/train_diffusion_policy.py``) and
the sim rollout (``scripts/sim/eval_diffusion_policy.py``).

Architecture:
  - A FROZEN pretrained ResNet18 embeds each camera image to 512-D. The two
    cameras (wrist, external) are concatenated → 1024-D visual feature. Frozen so
    we can precompute features once and train the head in minutes.
  - Per-step observation = [visual(1024), low-dim state + goal]. ``n_obs_steps``
    of these are flattened and MLP-encoded to a global conditioning vector.
  - A 1-D conv U-Net with FiLM conditioning denoises the action chunk
    (``pred_horizon`` × action_dim) under a DDPM schedule (ε-prediction).

Action = absolute TCP position (3-D) in the robot base frame; the down-quat is
constant and reattached at deployment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ─── Image encoder (frozen pretrained ResNet18 → 512-D). ─────────────────────
def make_image_encoder(device) -> nn.Module:
    """Pretrained ResNet18 with the classifier removed, frozen + eval."""
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_FEAT_DIM = 512


def preprocess_images(imgs_uint8: torch.Tensor) -> torch.Tensor:
    """[..., H, W, 3] uint8 → [..., 3, H, W] float, ImageNet-normalized."""
    x = imgs_uint8.float() / 255.0
    x = x.movedim(-1, -3)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x - mean) / std


# ─── 1-D conditional U-Net (FiLM). ───────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inp, out, kernel=3, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel, padding=kernel // 2),
            nn.GroupNorm(min(n_groups, out), out),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, inp, out, cond_dim, kernel=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp, out, kernel, n_groups),
            Conv1dBlock(out, out, kernel, n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, out * 2), nn.Unflatten(-1, (-1, 1)))
        self.out_channels = out
        self.residual_conv = (nn.Conv1d(inp, out, 1) if inp != out
                              else nn.Identity())

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)          # [B, out*2, 1]
        scale = embed[:, :self.out_channels]
        bias = embed[:, self.out_channels:]
        out = scale * out + bias                 # FiLM
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=128,
                 down_dims=(256, 512, 1024), kernel_size=3, n_groups=8):
        super().__init__()
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4),
            nn.Mish(), nn.Linear(dsed * 4, dsed))
        cond_dim = dsed + global_cond_dim

        all_dims = [input_dim, *down_dims]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])
        self.down_modules = nn.ModuleList()
        for i, (din, dout) in enumerate(in_out):
            last = i >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(din, dout, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dout, dout, cond_dim, kernel_size, n_groups),
                Downsample1d(dout) if not last else nn.Identity(),
            ]))
        self.up_modules = nn.ModuleList()
        for i, (din, dout) in enumerate(reversed(in_out[1:])):
            last = i >= len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dout * 2, din, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(din, din, cond_dim, kernel_size, n_groups),
                Upsample1d(din) if not last else nn.Identity(),
            ]))
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, sample, timestep, global_cond):
        # sample: [B, T, input_dim] → [B, input_dim, T]
        sample = sample.movedim(-1, -2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        timestep = timestep.expand(sample.shape[0]).to(sample.device)
        gfeat = self.diffusion_step_encoder(timestep)
        gfeat = torch.cat([gfeat, global_cond], dim=-1)

        x = sample
        h = []
        for res1, res2, down in self.down_modules:
            x = res1(x, gfeat); x = res2(x, gfeat); h.append(x); x = down(x)
        for m in self.mid_modules:
            x = m(x, gfeat)
        for res1, res2, up in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = res1(x, gfeat); x = res2(x, gfeat); x = up(x)
        x = self.final_conv(x)
        return x.movedim(-1, -2)                  # [B, T, input_dim]


# ─── DDPM schedule (ε-prediction, cosine betas). ─────────────────────────────
class DDPM:
    def __init__(self, num_timesteps=100, device="cpu"):
        self.T = num_timesteps
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        ac = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, 0)
        acp_prev = torch.cat([torch.ones(1), acp[:-1]])
        self.betas = betas.to(device)
        self.acp = acp.to(device)
        self.sqrt_acp = acp.sqrt().to(device)
        self.sqrt_1m_acp = (1 - acp).sqrt().to(device)
        self.sqrt_recip_acp = (1.0 / acp).sqrt().to(device)
        self.sqrt_recipm1_acp = (1.0 / acp - 1).sqrt().to(device)
        self.post_var = (betas * (1 - acp_prev) / (1 - acp)).to(device)
        self.post_c1 = (betas * acp_prev.sqrt() / (1 - acp)).to(device)
        self.post_c2 = ((1 - acp_prev) * alphas.sqrt() / (1 - acp)).to(device)

    def add_noise(self, x0, noise, t):
        return (self.sqrt_acp[t].view(-1, 1, 1) * x0
                + self.sqrt_1m_acp[t].view(-1, 1, 1) * noise)

    @torch.no_grad()
    def sample(self, model, global_cond, shape, device):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            eps = model.unet(x, torch.full((shape[0],), t, device=device), global_cond)
            x0 = self.sqrt_recip_acp[t] * x - self.sqrt_recipm1_acp[t] * eps
            x0 = x0.clamp(-1, 1)
            mean = self.post_c1[t] * x0 + self.post_c2[t] * x
            if t > 0:
                mean = mean + self.post_var[t].sqrt() * torch.randn_like(x)
            x = mean
        return x

    @torch.no_grad()
    def sample_ddim(self, model, global_cond, shape, device, n_steps=16):
        """Deterministic DDIM (eta=0) over a strided subset of timesteps.

        Far fewer network evals than full DDPM — used at deployment, where we
        re-sample every control tick for temporal ensembling."""
        ts = torch.linspace(self.T - 1, 0, n_steps, device=device).round().long()
        x = torch.randn(shape, device=device)
        for i in range(len(ts)):
            t = ts[i]
            eps = model.unet(x, t.expand(shape[0]), global_cond)
            acp_t = self.acp[t]
            x0 = ((x - (1 - acp_t).sqrt() * eps) / acp_t.sqrt()).clamp(-1, 1)
            acp_prev = self.acp[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=device)
            x = acp_prev.sqrt() * x0 + (1 - acp_prev).sqrt() * eps
        return x


# ─── Policy head (obs encoder + U-Net). ──────────────────────────────────────
class ObsEncoder(nn.Module):
    def __init__(self, per_step_dim, n_obs_steps, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(per_step_dim * n_obs_steps, cond_dim), nn.Mish(),
            nn.Linear(cond_dim, cond_dim))

    def forward(self, obs):                       # obs: [B, n_obs, per_step_dim]
        return self.net(obs.flatten(start_dim=1))


@dataclass
class DPConfig:
    action_dim: int = 3
    lowdim_dim: int = 22          # ee_pose(7)+ee_vel(6)+gripper(2)+goal(7)
    img_feat_dim: int = 2 * IMG_FEAT_DIM   # wrist+external
    n_obs_steps: int = 2
    pred_horizon: int = 16
    n_action_steps: int = 8
    cond_dim: int = 256
    num_diffusion_iters: int = 100
    image_keys: tuple = ("wrist_image", "external_image")


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: DPConfig):
        super().__init__()
        self.cfg = cfg
        per_step = cfg.img_feat_dim + cfg.lowdim_dim
        self.obs_encoder = ObsEncoder(per_step, cfg.n_obs_steps, cfg.cond_dim)
        self.unet = ConditionalUnet1D(cfg.action_dim, cfg.cond_dim)

    def global_cond(self, obs_feats):             # [B, n_obs, per_step_dim]
        return self.obs_encoder(obs_feats)

    def loss(self, obs_feats, actions, ddpm: DDPM):
        B = actions.shape[0]
        noise = torch.randn_like(actions)
        t = torch.randint(0, ddpm.T, (B,), device=actions.device)
        noisy = ddpm.add_noise(actions, noise, t)
        gc = self.global_cond(obs_feats)
        pred = self.unet(noisy, t, gc)
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def predict(self, obs_feats, ddpm: DDPM, ddim_steps=None):
        gc = self.global_cond(obs_feats)
        shape = (obs_feats.shape[0], self.cfg.pred_horizon, self.cfg.action_dim)
        if ddim_steps:
            return ddpm.sample_ddim(self, gc, shape, obs_feats.device, ddim_steps)
        return ddpm.sample(self, gc, shape, obs_feats.device)
