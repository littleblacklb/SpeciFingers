# SpeciFingers: Finger Identification and Error Correction on Capacitive Touchscreens

This is the official implementation of the [paper](https://dl.acm.org/doi/10.1145/3643559) "SpeciFingers: Finger Identification and Error Correction on Capacitive Touchscreens".

## Set Up Environment

```bash
conda env create -f environment.yml
conda activate specifingers
```

_Note: Make sure to install any extra dependencies required by the updated workflow, such as `fastervit`:_

```bash
pip install -r requirements.txt
```

## Dataset Preparation (Optimized Workflow)

The data processing pipeline has been heavily optimized for speed and memory efficiency. The previous pipeline generating individual JPGs/MP4s has been replaced with a faster `.npz` array packing method.

1. **Download and Extract Data**
   Download the dataset and place the `raw_log_data.zip` file at the root of this project.

   ```bash
   unzip raw_log_data.zip
   ```

2. **Generate Processed Frames**
   Convert the raw capacitive log data directly into `.npz` frame arrays in the `processed_frames/` directory. This script now utilizes parallel processing.

   ```bash
   python draw_RawFinger_optimized.py
   ```

3. **Pack Data for Training**
   Normalize the processed frames, resize them to 224x224, and pack them into a training-ready format in the `packed_data/` directory. This also constructs a `manifest.json`.
   ```bash
   python pack_data.py
   ```
   _Tip: If you ever need to correct the dataset labels without repacking all arrays, you can run `python fix_manifest.py`._

## Training the Model

The new training script supports RAM caching and includes modern backbone encoders like ViT.

```bash
python model_optimized.py --encoder alexnet
```

**Training Arguments:**

- `--encoder`: Choose the model backbone - `alexnet` (default), `vit`, or `fastervit`.
- `--test`: Run in test mode with a minimal subset of data for quick verification.
- `--resume <epoch>`: Resume training from a specific epoch (loads from `ckpt_<user_index>`).

Models will be saved periodically inside the `ckpt_<user_index>` folders. Training metrics are output to `.npy` files.

## Exporting Results

To convert the resulting array metric files into a human-readable text summary:

```bash
python export_training_results.py
```

This will generate `training_results.txt` showing epoch-by-epoch losses and scores.

## Citation

If you find our work and this repository useful, please consider citing:

```bibtex
@article{huang2024specifingers,
author = {Huang, Zeyuan and Gao, Cangjun and Wang, Haiyan and Deng, Xiaoming and Lai, Yu-Kun and Ma, Cuixia and Qin, Sheng-feng and Liu, Yong-Jin and Wang, Hongan},
title = {SpeciFingers: Finger Identification and Error Correction on Capacitive Touchscreens},
year = {2024},
issue_date = {March 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {8},
number = {1},
url = {https://doi.org/10.1145/3643559},
doi = {10.1145/3643559},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = {mar},
articleno = {8},
numpages = {28},
keywords = {Capacitive touchscreen, Deep learning, Error correction, Finger identification, Finger-specific interaction}
}
```

## Contact

If you have any questions, please create an issue on this repository or contact at *zeyuan2020@iscas.ac.cn*.
