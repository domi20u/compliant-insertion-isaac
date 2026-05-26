# DMP-PI² insertion & avoidance policies

GPU-batched PI² for learning DMP policies, with two registered scenarios:

- `avoidance` — three task params (height, two y-borders), 11 basis fns
- `insertion` — two task params (height, slot-opening y); slot bottom
  is pinned to `L_demo - goal_proximity`. 15 basis fns by default.

## Quick start

```bash
# Insertion
python scripts/generate_data.py --scenario insertion \
    --output-dir results/ins_v0 --n-runs 10
python scripts/train_policy.py  --scenario insertion \
    --data-dir results/ins_v0 --model-name model --n-basis 15
python scripts/test_policy.py   --scenario insertion \
    --data-dir results/ins_v0 --model-name model --n-basis 15

# Avoidance (default, no flag needed)
python scripts/generate_data.py --output-dir results/av_v0 --n-runs 10
```

Per-scenario defaults (basis count, sigmas, cost) live in
`configs/configs.py`. Adding a new task is one `Scenario` subclass
in `core/scenarios.py` plus one line in `_DATAGEN_CFG` in
`scripts/generate_data.py`.

## Layout
```
configs/   typed dataclass configs (tyro-driven CLIs)
core/      DMP wrapper, PI² optimizer, cost fns, scenario registry, NN policy
scripts/   generate_data / train_policy / test_policy entry points
results/   gitignored; PI² runs + trained models land here
```
## Dependencies

- `mp_pytorch` fork at https://github.com/domi20u/MP_PyTorch.git
- `torch`, `numpy`, `tyro`, `addict`, `matplotlib`
