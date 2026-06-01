"""Train the self-contained diffusion policy on the cluttered-insertion HDF5.

Plain python (no isaaclab, no lerobot) — uses the repo's existing torch +
torchvision. Pipeline:

  1. Load the HDF5 (one group per successful demo).
  2. Precompute FROZEN ResNet18 features for every wrist+external frame (once),
     so the trainable head (obs-MLP + 1-D U-Net) trains in minutes.
  3. Episode-level train/val split; per-dim normalization from TRAIN only.
  4. DDPM ε-training of the action chunk (3-D TCP position).
  5. Periodic held-out eval: sample action chunks and report L2 error (mm)
     against ground truth over the executed horizon.
  6. Save best checkpoint (head weights + norm stats + config).

Usage::

    python scripts/train/train_diffusion_policy.py \
        --hdf5 data/cluttered_insertion_demos.hdf5 \
        --out assets/dp_models/cluttered_dp.pt --epochs 200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compliant_insertion.policy.diffusion_policy import (  # noqa: E402
    DPConfig, DiffusionPolicy, DDPM, make_image_encoder, preprocess_images,
    IMG_FEAT_DIM,
)


# ─── Data loading + frozen-feature precompute. ───────────────────────────────
def load_episodes(h5_path: Path):
    """Return list of dicts per episode with raw arrays + goal."""
    eps = []
    with h5py.File(h5_path, "r") as f:
        data = f["data"]
        for k in sorted(data.keys(), key=lambda s: int(s.split("_")[1])):
            g = data[k]
            o = g["obs"]
            eps.append(dict(
                wrist=o["wrist_image"][:],            # [T,H,W,3] uint8
                external=o["external_image"][:],
                ee_pose=o["ee_pose_b"][:].astype(np.float32),
                ee_vel=o["ee_vel_b"][:].astype(np.float32),
                gripper=o["gripper_pos"][:].astype(np.float32),
                actions=g["actions"][:].astype(np.float32)[:, :3],  # position only
                goal=np.asarray(g.attrs["goal_pose_b"], np.float32),
            ))
    return eps


@torch.no_grad()
def precompute_features(eps, device, batch=256):
    """Encode every frame's wrist+external image to a 1024-D frozen feature.

    Adds ``img_feat`` [T, 1024] and ``lowdim`` [T, 22] to each episode dict.
    """
    enc = make_image_encoder(device)
    for ep in tqdm(eps, desc="encode images"):
        feats = []
        for cam in ("wrist", "external"):
            imgs = torch.from_numpy(ep[cam]).to(device)
            outs = []
            for i in range(0, imgs.shape[0], batch):
                x = preprocess_images(imgs[i:i + batch])
                outs.append(enc(x).cpu())
            feats.append(torch.cat(outs, 0))         # [T, 512]
        ep["img_feat"] = torch.cat(feats, dim=1).numpy()   # [T, 1024]
        T = ep["ee_pose"].shape[0]
        goal = np.broadcast_to(ep["goal"], (T, 7))
        ep["lowdim"] = np.concatenate(
            [ep["ee_pose"], ep["ee_vel"], ep["gripper"], goal], axis=1)  # [T,22]
        for k in ("wrist", "external"):
            del ep[k]                                  # free RAM


# ─── Normalization. ──────────────────────────────────────────────────────────
def fit_stats(train_eps):
    low = np.concatenate([e["lowdim"] for e in train_eps], 0)
    img = np.concatenate([e["img_feat"] for e in train_eps], 0)
    act = np.concatenate([e["actions"] for e in train_eps], 0)
    return dict(
        low_mean=low.mean(0), low_std=low.std(0) + 1e-6,
        img_mean=img.mean(0), img_std=img.std(0) + 1e-6,
        act_min=act.min(0), act_max=act.max(0),
    )


def norm_low(x, s):  return (x - s["low_mean"]) / s["low_std"]
def norm_img(x, s):  return (x - s["img_mean"]) / s["img_std"]
def norm_act(x, s):  return 2 * (x - s["act_min"]) / (s["act_max"] - s["act_min"] + 1e-9) - 1
def unnorm_act(x, s): return (x + 1) / 2 * (s["act_max"] - s["act_min"] + 1e-9) + s["act_min"]


# ─── Windowed sequence dataset. ──────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, eps, stats, cfg: DPConfig):
        self.cfg = cfg
        self.samples = []           # (ep_idx, t)
        self.ep = []
        for i, e in enumerate(eps):
            T = e["actions"].shape[0]
            per_step = np.concatenate(
                [norm_img(e["img_feat"], stats), norm_low(e["lowdim"], stats)], axis=1
            ).astype(np.float32)    # [T, 1046]
            act = norm_act(e["actions"], stats).astype(np.float32)  # [T, 3]
            self.ep.append((per_step, act))
            self.samples += [(i, t) for t in range(T)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, t = self.samples[idx]
        per_step, act = self.ep[i]
        T = per_step.shape[0]
        no, ph = self.cfg.n_obs_steps, self.cfg.pred_horizon
        obs_idx = [max(0, t - no + 1 + k) for k in range(no)]
        act_idx = [min(T - 1, t + k) for k in range(ph)]
        return (torch.from_numpy(per_step[obs_idx]),     # [no, 1046]
                torch.from_numpy(act[act_idx]))          # [ph, 3]


# ─── EMA. ────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


# ─── Eval: sampled action L2 (mm) over the executed horizon. ─────────────────
@torch.no_grad()
def evaluate(model, ddpm, loader, stats, cfg, device, max_batches=4):
    model.eval()
    amin = torch.tensor(stats["act_min"], device=device)
    amax = torch.tensor(stats["act_max"], device=device)
    errs = []
    for bi, (obs, act) in enumerate(loader):
        if bi >= max_batches:
            break
        obs = obs.to(device)
        pred = model.predict(obs, ddpm)                 # [B, ph, 3] normalized
        pu = (pred + 1) / 2 * (amax - amin + 1e-9) + amin
        au = (act.to(device) + 1) / 2 * (amax - amin + 1e-9) + amin
        n = cfg.n_action_steps
        errs.append((pu[:, :n] - au[:, :n]).norm(dim=-1).mean().item())
    model.train()
    return float(np.mean(errs)) * 1000.0            # mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("assets/dp_models/cluttered_dp.pt"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pred-horizon", type=int, default=16)
    ap.add_argument("--obs-steps", type=int, default=2)
    ap.add_argument("--action-steps", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[dp] loading {args.hdf5}")
    eps = load_episodes(args.hdf5)
    print(f"[dp] {len(eps)} episodes; precomputing frozen image features…")
    precompute_features(eps, device)

    # Episode-level split.
    idx = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * args.val_frac))
    val_eps = [eps[i] for i in idx[:n_val]]
    train_eps = [eps[i] for i in idx[n_val:]]
    print(f"[dp] split: {len(train_eps)} train / {len(val_eps)} val episodes")

    stats = fit_stats(train_eps)
    cfg = DPConfig(
        action_dim=3, lowdim_dim=train_eps[0]["lowdim"].shape[1],
        img_feat_dim=2 * IMG_FEAT_DIM, n_obs_steps=args.obs_steps,
        pred_horizon=args.pred_horizon, n_action_steps=args.action_steps)

    train_ds = SeqDataset(train_eps, stats, cfg)
    val_ds = SeqDataset(val_eps, stats, cfg)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2)
    print(f"[dp] {len(train_ds)} train / {len(val_ds)} val windows")

    model = DiffusionPolicy(cfg).to(device)
    ddpm = DDPM(cfg.num_diffusion_iters, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dp] trainable head params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(train_dl))
    ema = EMA(model)

    def save(tag):
        torch.save(dict(
            model=ema.shadow, cfg=vars(cfg),
            # Store stats as torch tensors (not numpy arrays) so the checkpoint
            # unpickles in the Isaac env, which may have a different numpy major
            # version (numpy 2.x pickles reference numpy._core, absent in 1.x).
            stats={k: torch.as_tensor(np.asarray(v), dtype=torch.float32)
                   for k, v in stats.items()},
            args={k: str(v) for k, v in vars(args).items()},
        ), args.out if tag == "best" else args.out.with_suffix(f".{tag}.pt"))

    best = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        losses = []
        for obs, act in train_dl:
            obs, act = obs.to(device), act.to(device)
            loss = model.loss(obs, act, ddpm)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model)
            losses.append(loss.item())
        line = f"[dp] epoch {ep:03d} loss={np.mean(losses):.4f}"
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            eval_model = DiffusionPolicy(cfg).to(device)
            eval_model.load_state_dict(ema.shadow)
            mm = evaluate(eval_model, ddpm, val_dl, stats, cfg, device)
            line += f"  val_action_L2={mm:.1f}mm"
            if mm < best:
                best = mm; save("best"); line += "  *saved"
        print(line)
    print(f"[dp] done in {time.time()-t0:.0f}s. best val L2 = {best:.1f}mm → {args.out}")


if __name__ == "__main__":
    main()
