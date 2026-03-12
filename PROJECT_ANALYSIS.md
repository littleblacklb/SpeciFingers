# SpeciFingers Project Analysis

This document captures the current understanding of the repository so we can discuss and refine next steps.

## 1. What This Project Is

SpeciFingers is a research-focused PyTorch pipeline for finger identification on capacitive touchscreen raw logs.

- Paper implementation reference: [README.md](./README.md)
- Main flow: raw logs -> processed frame arrays -> packed training data -> model training -> results export

## 2. Repository Shape

Top-level scripts are mostly self-contained:

- `draw_RawFinger_optimized.py`: parse and render raw `.log` data into `processed_frames/*.npz`
- `pack_data.py`: convert rendered frames into normalized, fixed-shape training arrays under `packed_data/`
- `functions.py`: original dataset/model utilities (AlexNet/ResNet encoder + LSTM decoder)
- `functions_optimized.py`: optimized dataloading and additional encoders (ViT/FasterViT/EfficientViT)
- `model_optimized.py`: training entrypoint
- `fix_manifest.py`: relabel packed manifest without repacking
- `export_training_results.py`: export `.npy` training metrics to readable text

## 3. End-to-End Data Flow

### Step A: Raw log preprocessing

- Input: `raw_log_data/**/*.log`
- Script: `draw_RawFinger_optimized.py`
- Key behavior:
  - Parses log lines containing `Type` and `rawPoints`
  - Renders each frame to a grayscale matrix (touch sensitivity map)
  - Saves each sample as compressed `.npz` (`frames` array)
  - Uses multiprocessing for throughput

### Step B: Packing for training

- Input: `processed_frames/**/*.npz`
- Script: `pack_data.py`
- Key behavior:
  - Limits/interpolates each sample to `NUM_FRAMES=50`
  - Resizes frames to `224x224`
  - Replicates grayscale to 3 channels and applies ImageNet mean/std normalization
  - Writes normalized sample arrays to `packed_data/<user>/...npz`
  - Builds `packed_data/manifest.json`

### Step C: Training

- Script: `model_optimized.py`
- Pipeline:
  - Loads train/validation splits via `create_fast_dataloaders(...)`
  - Encoder options: `alexnet`, `vit`, `fastervit`, `efficientvit`
  - Decoder: LSTM-based classifier head
  - Saves per-epoch checkpoint files (`ckpt_<user_index>`)
  - Saves metric arrays (`CRNN_epoch_*.npy`)

### Step D: Metrics export

- Script: `export_training_results.py`
- Output: `training_results.txt`

## 4. Notable Current Behaviors and Risks

## 4.1 Class filtering mismatch (important)

Current filtering and labeling logic are inconsistent:

- Preprocessing keeps only left-hand samples and excludes `LeftThumbSide`
- Data loader later drops `ThumbFront` samples
- Label encoder still expects 3 classes: `ThumbSide`, `3Middle`, `LittleFinger`

Consequence:

- Effective training set can collapse to 2 classes (`3Middle`, `LittleFinger`) while model output is configured for 3 classes (`k = 3`).
- This may not always crash, but it introduces a dataset/model mismatch and evaluation ambiguity.

## 4.2 Cross-validation scope currently narrowed

`model_optimized.py` sets:

- `user_range = range(0, 1)`

So only user `0` is used as held-out test user, despite comments describing leave-one-out cross-validation.

## 4.3 Environment/version mismatch risk

- `environment.yml` pins Python `3.7.11`
- Code uses modern type syntax like `dict | None` (Python 3.10+)

This can break execution if users strictly follow the conda env file without updating Python.

## 4.4 Documentation drift

README lists encoder options as `alexnet`, `vit`, `fastervit`, but code also supports `efficientvit`.

## 5. Quick Reality Check from Dataset Archive

Based on archive-level inspection of `raw_log_data.zip`:

- Total logs: `37771`
- Users: `20`
- Left-hand excluding `LeftThumbSide`: `15753` samples
- After loader-level `ThumbFront` removal: `12634` samples
- Label distribution after that second filter is effectively:
  - `3Middle`: `9483`
  - `LittleFinger`: `3151`

This supports the class-mismatch concern above.

## 6. Current State of Workspace

- No generated `processed_frames/manifest.json` currently present
- No generated `packed_data/manifest.json` currently present
- Training was not executed in this environment because dependencies are not installed in the active interpreter

## 7. Discussion Starters (for next iteration)

1. Should the target task be 2-class or 3-class?
2. If 3-class is intended, which thumb class should be kept (`ThumbFront` or `ThumbSide`)?
3. Should leave-one-out run over all users by default?
4. Should environment be upgraded (Python >= 3.10) or code backported to 3.7 syntax?
5. Do we want a reproducible "quick smoke test" mode with minimal data and deterministic seed?

