"""Quick fix: Update manifest.json labels without re-packing data."""

import json
from pathlib import Path

PACKED_DIR = "packed_data"


def fix_label(folder_name: str) -> str:
    """Extract correct label from filename."""
    # Format: v_timestamp_ActionType_Action
    # e.g., v_0124123015_LeftThumbSide_Press -> ThumbSide
    parts = folder_name.split("_")

    if len(parts) >= 3:
        action_raw = parts[2]
        action = action_raw.replace("Left", "").replace("Right", "")
        if action in ("ForeFinger", "MiddleFinger", "RingFinger"):
            action = "3Middle"
        return action
    return "unknown"


def main():
    manifest_path = Path(PACKED_DIR) / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    fixed_count = 0
    for sample in manifest["samples"]:
        old_label = sample["label"]
        new_label = fix_label(sample["folder"])
        if old_label != new_label:
            sample["label"] = new_label
            fixed_count += 1

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Fixed {fixed_count} labels in {manifest_path}")

    # Show label distribution
    labels = [s["label"] for s in manifest["samples"]]
    from collections import Counter

    print("Label distribution:", dict(Counter(labels)))


if __name__ == "__main__":
    main()
