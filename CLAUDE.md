# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Activate the virtual environment before running any Python commands:
```bash
source emimic/bin/activate
```

## Common Commands

### Training
```bash
# Basic training (interactive)
python egomimic/trainHydra.py --config-name=train_zarr_cartesian

# Training with specific data/model
python egomimic/trainHydra.py --config-name=train_zarr_cartesian data=aria model=hpt_bc_flow_aria

# Debug run (fewer epochs, no wandb)
python egomimic/trainHydra.py --config-name=train_zarr_cartesian trainer=debug logger=debug norm_stats.sample_frac=0.001

# SLURM submission (add -m flag)
python egomimic/trainHydra.py --config-name=train_zarr_cartesian data=aria model=hpt_bc_flow_aria -m

# Override config inline
python egomimic/trainHydra.py --config-name=train_zarr_cartesian data=aria train.batch_size=64 train.num_workers=8
```

### Eval (from a previous training run)
```bash
python egomimic/trainHydra.py \
  --config-path=../logs/<run>/.hydra/ \
  --config-name=config \
  hydra.searchpath=[file://egomimic/hydra_configs] \
  ++mode=eval +evaluator=eval_video \
  +norm_stats.precomputed_norm_path="logs/<run>/norm_stats/norm_stats.json" \
  ++ckpt_path="'logs/<run>/checkpoints/last.ckpt'"
```

### Testing
```bash
pytest                        # run all tests
pytest egomimic/algo/test_pi.py  # run a single test file
```

### Linting & Formatting
Pre-commit hooks run ruff automatically on commit. To run manually:
```bash
ruff check . --fix   # lint + autofix
ruff format .        # format
```

Ruff config: line-length 88, rules E/F/I, ignores E501. Excludes `external/`, `egomimic/robot/eva/stanford_repo`, `*.ipynb`, `egomimic/robot/oculus_reader`.

### Data Download
```bash
python egomimic/scripts/data_download/sync_s3.py --local-dir <dir> --filters <filter>
```

### SLURM GPU Allocation (sky1/sky2)
```bash
salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G
```

## Architecture Overview

EgoVerse is a robot learning framework for training manipulation policies from egocentric (first-person) demonstrations across multiple robot embodiments.

### Training Pipeline

```
trainHydra.py (entry point)
  -> Hydra config composition (hydra_configs/)
  -> S3EpisodeResolver: SQL filters -> episode discovery -> S3 download
  -> MultiDataset: merges zarr episodes per embodiment
  -> Embodiment transforms: raw keys -> canonical representation
  -> DataSchematic: shape inference + normalization stats
  -> MultiDataModuleWrapper: CombinedLoader across embodiments
  -> ModelWrapper (LightningModule): wraps Algo subclass
  -> PyTorch Lightning Trainer (DDP)
```

### Key Packages

- **`egomimic/algo/`** - Algorithm implementations inheriting from `Algo` base class (`algo.py`). Core methods: `process_batch_for_training()`, `forward_training()`, `compute_losses()`, `forward_eval()`. Implementations: HPT (`hpt.py`), ACT (`act.py`), Pi/OpenPI (`pi.py`).

- **`egomimic/models/`** - Neural network modules: vision encoders, transformer trunks, denoising networks, diffusion/flow-matching policy wrappers.

- **`egomimic/rldb/`** - Data pipeline. `zarr/zarr_dataset_multi.py` has `MultiDataset` and `S3EpisodeResolver`. `zarr/utils.py` has `DataSchematic` for normalization and shape tracking. `embodiment/` defines per-robot transforms and key mappings.

- **`egomimic/pl_utils/`** - PyTorch Lightning integration. `pl_model.py` has `ModelWrapper` (LightningModule wrapping Algo). `pl_data_utils.py` has `MultiDataModuleWrapper`.

- **`egomimic/scripts/`** - Data processing pipelines per source (aloha_process, aria_process, eva_process, mecka_process). `run_conversion.py` orchestrates Ray-parallel conversions.

- **`egomimic/robot/`** - Hardware integration: EVA kinematics, rollout execution, demo collection, Oculus reader.

- **`egomimic/eval/`** - Evaluation callbacks (video generation during validation).

- **`egomimic/utils/`** - Shared utilities: camera transforms, pose math, tensor ops, AWS/S3/SQL access.

### Configuration System (Hydra)

All configs in `egomimic/hydra_configs/`. Key subdirs: `data/` (dataset configs per embodiment), `model/` (20+ model variants), `trainer/` (Lightning trainer), `logger/` (wandb), `callbacks/` (checkpoints), `hydra/launcher/` (SLURM submitit).

For most use cases, only the data YAML needs modification (`folder_path`, `batch_size`, `num_workers`).

### Embodiment System

Robots are abstracted via `Embodiment` base class in `rldb/embodiment/`. Each embodiment defines `get_keymap()` (raw dataset keys -> canonical names) and `get_transform_list()` (frame transforms, action chunking). Supported: EVE, EVA, Aria, MECKA, SCALE (bimanual, left, right variants).

### Data Format

Episodes stored as zarr arrays (`episode_{hash}.zarr/`) on S3/local. Contains observations (JPEG-compressed images, state vectors), actions, and metadata. `DataSchematic` handles normalization (zscore/minmax/quantile) with per-embodiment statistics caching.

### External Dependencies

Git submodules in `external/`: `lerobot` (dataset format), `openpi` (Pi-0.5 model), `scale`, `rpl_vision_utils`. For Pi-0.5 setup see `pi05.md`.
