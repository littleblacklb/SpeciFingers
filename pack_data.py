"""
Pack processed frames into training-ready format.
Reads from processed_frames/ (output of draw_RawFinger_optimized.py)
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
INPUT_DIR = "processed_frames"
OUTPUT_DIR = "packed_data"
RES_SIZE = 224
NUM_FRAMES = 50
NUM_WORKERS = 16

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def process_npz_file(args) -> dict | None:
    """Process a single .npz file from processed_frames."""
    npz_path, output_base = args

    try:
        with np.load(npz_path) as data:
            frames = data["frames"]  # Shape: (num_frames, H, W) grayscale

        if len(frames) < NUM_FRAMES:
            # Interpolate to get enough frames
            from scipy.ndimage import zoom

            ratio = NUM_FRAMES / len(frames)
            frames = zoom(frames, (ratio, 1, 1), order=1)[:NUM_FRAMES]
        else:
            frames = frames[:NUM_FRAMES]

        # Resize and normalize
        output_frames = np.empty((NUM_FRAMES, 3, RES_SIZE, RES_SIZE), dtype=np.float32)

        for i, frame in enumerate(frames):
            # Resize
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((RES_SIZE, RES_SIZE), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0

            # Grayscale to RGB (replicate channels)
            arr_rgb = np.stack([arr, arr, arr], axis=-1)

            # Normalize
            arr_rgb = (arr_rgb - MEAN) / STD

            # HWC -> CHW
            output_frames[i] = arr_rgb.transpose(2, 0, 1)

        # Extract label from filename (format: v_timestamp_ActionType_Action)
        # e.g., v_0124123015_LeftThumbSide_Press -> LeftThumbSide -> ThumbSide
        path = Path(npz_path)
        folder_name = path.stem
        parts = folder_name.split("_")

        # Action type is at index 2, need to normalize (remove Left/Right, group middle fingers)
        if len(parts) >= 3:
            action_raw = parts[2]
            # Remove Left/Right prefix
            action = action_raw.replace("Left", "").replace("Right", "")
            # Group ForeFinger, MiddleFinger, RingFinger as 3Middle
            if action in ("ForeFinger", "MiddleFinger", "RingFinger"):
                action = "3Middle"
            label = action
        else:
            label = "unknown"
        user_id = path.parent.name

        # Save
        output_dir = output_base / user_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{folder_name}.npz"
        np.savez_compressed(output_file, frames=output_frames)

        return {
            "file": str(output_file.relative_to(output_base)),
            "label": label,
            "user": user_id,
            "folder": folder_name,
            "num_frames": NUM_FRAMES,
        }
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        return None


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # Find all .npz files
    npz_files = list(input_path.rglob("*.npz"))
    npz_files = [f for f in npz_files if f.name != "manifest.json"]

    # Filter: only Left hand samples, exclude LeftThumbSide
    def is_valid_sample(path: Path) -> bool:
        parts = path.stem.split("_")
        if len(parts) >= 3:
            action_raw = parts[2]
            # Only Left hand, exclude LeftThumbSide
            return action_raw.startswith("Left") and action_raw != "LeftThumbSide"
        return False

    npz_files = [f for f in npz_files if is_valid_sample(f)]

    print(f"Found {len(npz_files)} Left hand samples (excluding LeftThumbSide)")

    args_list = [(str(f), output_path) for f in npz_files]

    manifest = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_npz_file, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Packing"):
            result = future.result()
            if result:
                manifest.append(result)

    # Save manifest
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(
            {
                "samples": manifest,
                "num_frames": NUM_FRAMES,
                "resolution": RES_SIZE,
                "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
            },
            f,
            indent=2,
        )

    print(f"\nPacked {len(manifest)} samples to {output_path}")


if __name__ == "__main__":
    main()
