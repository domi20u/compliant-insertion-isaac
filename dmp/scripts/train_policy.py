"""Train an MLP policy that maps task features to DMP weights.

Usage:

    # Box-avoidance (default — backward compatible)
    python scripts/train_policy.py --data-dir results/avoidance_run0 \
        --model-name model

    # Insertion
    python scripts/train_policy.py --scenario insertion \
        --data-dir results/insertion_run0 --model-name model \
        --n-basis 20

``n_inputs`` is set from the scenario; do not pass it manually.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader

from configs.configs import NNTrainConfig
from core.nn_policy import MLPPolicy, TrajectoryDataset
from core.scenarios import Scenario, get_scenario, list_scenarios


def _pop_scenario_flag(argv: list[str]) -> str:
    name = "avoidance"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--scenario" and i + 1 < len(argv):
            name = argv[i + 1]
            del argv[i:i + 2]
            continue
        if a.startswith("--scenario="):
            name = a.split("=", 1)[1]
            del argv[i]
            continue
        i += 1
    if name not in list_scenarios():
        raise SystemExit(
            f"Unknown --scenario {name!r}. Choices: {list_scenarios()}")
    return name


def load_pi2_data(cfg: NNTrainConfig,
                  scenario: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all per-run PI² data and stitch into (X, y, iters).

    The per-run NN input row is built via ``scenario.encode_nn_input``, so
    avoidance gets ``[z_min, y0, y1] / L_demo`` and insertion gets
    ``[z_min, y0] / L_demo``.

    Returns:
        X: [N, n_inputs] task features
        y: [N, n_basis * n_dims] DMP weight targets
        iters: [n_runs] number of episodes per run
    """
    data_dir = Path(cfg.data_dir)
    run_dirs = sorted(p for p in data_dir.iterdir()
                      if p.is_dir() and p.name.startswith("train"))
    print(f"[train_policy] found {len(run_dirs)} run directories in {data_dir}")
    iters = np.zeros(len(run_dirs), dtype=int)
    nn_outputs: list[np.ndarray] = []
    nn_inputs: list[np.ndarray] = []

    for j, run_dir in enumerate(run_dirs):
        npz_path = data_dir / f"train{j}" / f"train_{j}.npz"
        if not npz_path.exists():
            print(f"[train_policy] warning: missing {npz_path}, skipping run {j}")
            continue
        data = np.load(npz_path, allow_pickle=True)

        # If the npz was generated with a scenario tag, sanity-check that it
        # matches the one the user passed at the CLI.
        if "scenario" in data.files:
            saved = str(data["scenario"])
            if saved != scenario.name:
                print(f"[train_policy] WARNING: run {j} was generated with "
                      f"scenario={saved!r} but you passed --scenario={scenario.name!r}")

        n_eps = int(data["n_episodes"])
        iters[j] = n_eps
        print(f"[train_policy] run {j}: {n_eps} episodes")
        if n_eps == 0:
            continue

        # Output: weights for the perturbed dofs (1 and 2), flattened.
        # n_dims is fixed to 2 by the scenario (both tasks perturb DOFs 1 and 2).
        nn_out = data["training_data"][:, :, 1:3].reshape(
            n_eps, cfg.n_basis * scenario.n_dims)
        nn_outputs.append(nn_out)

        # Input row per episode: scenario-specific encoding of (min_z, y_borders).
        z_min = np.min(data["labels"], axis=1) / cfg.L_demo        # [n_eps]
        y_loc = data["y_locations"] / cfg.L_demo                   # [n_eps, 2]
        rows = np.stack([
            scenario.encode_nn_input(z_min[k], y_loc[k])
            for k in range(n_eps)
        ], axis=0)                                                 # [n_eps, n_inputs]
        nn_inputs.append(rows)

    if not nn_inputs:
        raise RuntimeError(f"No usable runs found in {data_dir}")

    # Subsample to a uniform number per run.
    valid_iters = iters[iters > 0]
    min_iters = int(np.min(valid_iters))
    n_per_run = min(cfg.samples_per_run, min_iters)
    print(f"[train_policy] min iters per run = {min_iters}, sampling {n_per_run}/run")

    X_all, y_all = [], []
    for X_run, y_run in zip(nn_inputs, nn_outputs):
        idxs = np.round(np.linspace(0, len(X_run) - 1, n_per_run)).astype(int)
        X_all.append(X_run[idxs])
        y_all.append(y_run[idxs])

    return np.concatenate(X_all), np.concatenate(y_all), iters


def main(cfg: NNTrainConfig, scenario: Scenario) -> None:
    # The scenario owns n_inputs and n_dims — override whatever the CLI/defaults
    # said so users can't accidentally pick 3 inputs for an insertion model.
    expected_n_inputs = scenario.n_task_params + 1
    if cfg.n_inputs != expected_n_inputs:
        print(f"[train_policy] overriding n_inputs {cfg.n_inputs} -> "
              f"{expected_n_inputs} (from scenario {scenario.name!r})")
        cfg.n_inputs = expected_n_inputs
    if cfg.n_dims != scenario.n_dims:
        print(f"[train_policy] overriding n_dims {cfg.n_dims} -> "
              f"{scenario.n_dims} (from scenario {scenario.name!r})")
        cfg.n_dims = scenario.n_dims

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    X, y, iters = load_pi2_data(cfg, scenario)
    print(f"[train_policy] dataset: X={X.shape}, y={y.shape}")

    # Split BEFORE computing normalization stats to avoid leakage.
    n_total = len(X)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_total)
    n_val = int(cfg.val_split * n_total)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Stats from training split only.
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    print(f"[train_policy] y scale: mean range [{y_mean.min():.2f}, {y_mean.max():.2f}], "
          f"std range [{y_std.min():.2f}, {y_std.max():.2f}]")

    # Normalize targets (and inputs) for training.
    y_train_n = (y_train - y_mean) / np.maximum(y_std, 1e-8)
    y_val_n = (y_val - y_mean) / np.maximum(y_std, 1e-8)
    # X normalization happens inside the model, so pass raw X.

    train_ds = TrajectoryDataset(torch.tensor(X_train), torch.tensor(y_train_n))
    val_ds = TrajectoryDataset(torch.tensor(X_val), torch.tensor(y_val_n))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MLPPolicy(cfg.n_inputs, cfg.n_basis * cfg.n_dims,
                      hidden_sizes=cfg.hidden_sizes).to(device)
    model.set_norm_stats(x_mean, x_std, y_mean, y_std)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(f"[train_policy] model: {model}")
    t_start = time.time()
    for epoch in range(cfg.num_epochs):
        # train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= max(len(val_loader), 1)

        print(f"[train_policy] epoch {epoch+1:>3}/{cfg.num_epochs} | "
              f"train {train_loss:.6f} | val {val_loss:.6f}")

    duration = time.time() - t_start
    print(f"[train_policy] training done in {duration:.1f}s")

    # Save scripted model + dataset metadata
    out_dir = Path(cfg.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu())
    model_path = out_dir / f"{cfg.model_name}.pth"
    scripted.save(str(model_path))
    print(f"[train_policy] saved model to {model_path}")

    np.savez(
        out_dir / f"{cfg.model_name}_data.npz",
        nn_input=X,
        nn_output=y,
        iters=iters,
        t_nn=duration,
        max_min_ins=[float(np.max(X[:, 0])),
                     float(np.min(X[:, 1])),
                     float(np.max(X[:, -1]))],  # last input col, scenario-agnostic
        scenario=scenario.name,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    scenario_name = _pop_scenario_flag(argv)
    sys.argv = [sys.argv[0]] + argv

    cfg = tyro.cli(NNTrainConfig)
    scenario = get_scenario(scenario_name)
    main(cfg, scenario)
