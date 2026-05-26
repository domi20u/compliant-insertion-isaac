"""MLP policy that maps obstacle features to DMP basis-function weights."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Feed-forward MLP with built-in input/output normalization.

    Normalization stats are stored as buffers so they travel with the
    scripted model. During training, call ``forward`` and compare against
    *normalized* targets. At inference, call ``predict`` to get outputs
    on the original (DMP-weight) scale.

    Replaces the misnamed ``RBFNet`` from the original code (it never had
    RBF kernels — it's a plain MLP).
    """

    def __init__(self, input_size: int, output_size: int,
                 hidden_sizes: Sequence[int] = (256, 512)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))  # linear output for regression
        self.net = nn.Sequential(*layers)

        # Normalization buffers — identity by default, set via set_norm_stats().
        # Registered as buffers so they (a) move with .to(device), (b) save
        # with state_dict, and (c) survive torch.jit.script.
        self.register_buffer("x_mean", torch.zeros(input_size))
        self.register_buffer("x_std", torch.ones(input_size))
        self.register_buffer("y_mean", torch.zeros(output_size))
        self.register_buffer("y_std", torch.ones(output_size))

    def set_norm_stats(self,
                       x_mean: np.ndarray | torch.Tensor,
                       x_std: np.ndarray | torch.Tensor,
                       y_mean: np.ndarray | torch.Tensor,
                       y_std: np.ndarray | torch.Tensor,
                       eps: float = 1e-8) -> None:
        """Populate normalization buffers from training-split statistics.

        Call this once before training, after computing stats on the
        training split only (no leakage from validation).
        """
        def _to_tensor(a: np.ndarray | torch.Tensor) -> torch.Tensor:
            if isinstance(a, np.ndarray):
                return torch.from_numpy(a).float()
            return a.float()

        self.x_mean.copy_(_to_tensor(x_mean))
        self.x_std.copy_(_to_tensor(x_std).clamp(min=eps))
        self.y_mean.copy_(_to_tensor(y_mean))
        self.y_std.copy_(_to_tensor(y_std).clamp(min=eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns *normalized* predictions. Use during training."""
        x = (x - self.x_mean) / self.x_std
        return self.net(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions on the original DMP-weight scale.

        Use at inference time when generating DMP rollouts.
        """
        return self.forward(x) * self.y_std + self.y_mean


class TrajectoryDataset(torch.utils.data.Dataset):
    """In-memory tensor dataset for (obstacle_features -> DMP weights) pairs."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.float()
        self.y = y.float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]