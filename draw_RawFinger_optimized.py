"""
Optimized version of draw_RawFinger.py
Performance improvements:
- Removed cv.waitKey(1000) delay (was waiting 1 sec per frame!)
- Parallel processing with multiprocessing
- Direct frame output (skip video encoding, use NPZ)
- Batch JSON parsing
"""

import json
import os
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
INPUT_DIR = "raw_log_data"
OUTPUT_DIR = "processed_frames"  # Output as numpy arrays
NUM_WORKERS = os.cpu_count() or 4
OUTPUT_FORMAT = "npz"  # 'npz' (fast) or 'video' (compatible with gen_jpgs.py)


def parse_log_file(file_path: str) -> list:
    """Parse log file and extract touch point data."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Find start of JSON data
    start_idx = content.find("{")
    if start_idx == -1:
        return []

    data_str = content[start_idx:].replace("'", '"')
    lines = data_str.split("\n")

    frames = []
    for line in lines:
        if "Type" in line and "rawPoints" in line:
            try:
                frame_data = json.loads(line)
                frames.append(frame_data)
            except json.JSONDecodeError:
                continue

    return frames


def render_frame(frame_data: dict, img_size=(160, 160)) -> np.ndarray:
    """Render a single frame from touch point data."""
    # Create larger canvas for rendering
    canvas = np.zeros((1080, 1920), dtype=np.float32)

    for point in frame_data.get("rawPoints", []):
        sensitivity = point.get("Sensitivity", 0)
        raw_x = point.get("RawX", 0)
        raw_y = point.get("RawY", 0)

        # Calculate rectangle bounds
        x_scale = 1920 / 122
        y_scale = 1080 / 70
        x_center = int(raw_x * x_scale)
        y_center = int(raw_y * y_scale)
        half_w = int(x_scale / 2)
        half_h = int(y_scale / 2)

        x1 = max(0, x_center - half_w)
        x2 = min(1920, x_center + half_w)
        y1 = max(0, y_center - half_h)
        y2 = min(1080, y_center + half_h)

        canvas[y1:y2, x1:x2] = sensitivity / 4

    # Find bounding box of non-zero region
    nonzero = np.nonzero(canvas)
    if len(nonzero[0]) == 0:
        return np.zeros(img_size, dtype=np.uint8)

    y_min, y_max = nonzero[0].min(), nonzero[0].max()
    x_min, x_max = nonzero[1].min(), nonzero[1].max()

    # Extract ROI with padding
    pad = 50
    y1 = max(0, y_min - pad)
    y2 = min(1080, y_max + pad + 60)  # +60 to match original 110 offset
    x1 = max(0, x_min - pad)
    x2 = min(1920, x_max + pad + 60)

    roi = canvas[y1:y2, x1:x2]

    # Resize to target size
    from PIL import Image

    roi_img = Image.fromarray(roi.astype(np.uint8))
    roi_img = roi_img.resize(img_size, Image.BILINEAR)

    return np.array(roi_img)


def process_log_file(args) -> dict | None:
    """Process a single log file and save frames."""
    file_path, output_base = args

    try:
        frames_data = parse_log_file(file_path)
        if not frames_data:
            return None

        # Render all frames
        frames = []
        for frame_data in frames_data:
            frame = render_frame(frame_data)
            frames.append(frame)

        if not frames:
            return None

        frames_array = np.stack(frames, axis=0)  # Shape: (num_frames, H, W)

        # Extract user number and filename
        path = Path(file_path)
        parts = path.parent.name.split("_")
        user_num = parts[0] if parts else "0"
        video_name = "v_" + path.stem

        # Save output
        output_dir = output_base / user_num
        output_dir.mkdir(parents=True, exist_ok=True)

        if OUTPUT_FORMAT == "npz":
            output_file = output_dir / f"{video_name}.npz"
            np.savez_compressed(output_file, frames=frames_array)
        else:
            # Save as video for compatibility with gen_jpgs.py
            import cv2

            output_file = output_dir / f"{video_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frames_array.shape[1:3]
            writer = cv2.VideoWriter(str(output_file), fourcc, 2, (w, h))
            for frame in frames_array:
                frame_rgb = np.stack([frame, frame, frame], axis=-1)
                writer.write(frame_rgb)
            writer.release()

        return {"file": str(output_file), "user": user_num, "num_frames": len(frames)}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # Find all log files
    log_files = list(input_path.rglob("*.log"))

    # Filter: only Left hand samples, exclude LeftThumbSide
    def is_valid_sample(path: Path) -> bool:
        # Filename format: timestamp_ActionType_Action.log (e.g., 0124123015_LeftThumbSide_Press.log)
        parts = path.stem.split("_")
        if len(parts) >= 2:
            action_raw = parts[1]  # ActionType is at index 1 in log filenames
            # Only Left hand, exclude LeftThumbSide
            return action_raw.startswith("Left") and action_raw != "LeftThumbSide"
        return False

    log_files = [f for f in log_files if is_valid_sample(f)]
    print(f"Found {len(log_files)} Left hand samples (excluding LeftThumbSide)")

    # Prepare arguments
    args_list = [(str(f), output_path) for f in log_files]

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_log_file, args): args for args in args_list}

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            result = future.result()
            if result:
                results.append(result)

    print(f"\nProcessed {len(results)} files to {output_path}")

    # Save manifest
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump({"files": results, "format": OUTPUT_FORMAT}, f, indent=2)


if __name__ == "__main__":
    main()
