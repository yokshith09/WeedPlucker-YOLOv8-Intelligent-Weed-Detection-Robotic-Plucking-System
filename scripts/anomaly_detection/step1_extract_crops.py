"""
STEP 1 — Extract cauliflower crops from your training dataset
Run this first. It reads your existing YOLO labels and saves
cropped cauliflower images for anomaly head training.

Usage:
    python step1_extract_crops.py
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT = r"C:\NEWDRIVE\Model_train\dataset\balanced"
OUTPUT_DIR   = r"C:\New folder\Model_train\anomaly_data\cauliflower_crops"
CROP_CLASS   = 0          # 0 = crop (cauliflower), 1 = weed
CROP_SIZE    = 64         # resize all crops to 64x64 for autoencoder
PADDING      = 0.10       # add 10% padding around each bbox
MIN_CROP_PX  = 32         # skip crops smaller than this (too small to be useful)
# ─────────────────────────────────────────────────────────────────────────────

def yolo_bbox_to_pixels(x_c, y_c, w, h, img_w, img_h, pad=0.0):
    """Convert YOLO normalised bbox → pixel coords with optional padding."""
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    pw = (x2 - x1) * pad
    ph = (y2 - y1) * pad
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(img_w, int(x2 + pw))
    y2 = min(img_h, int(y2 + ph))
    return x1, y1, x2, y2


def extract_crops(split="train"):
    images_dir = Path(DATASET_ROOT) / "images" / split
    labels_dir = Path(DATASET_ROOT) / "labels" / split
    out_dir    = Path(OUTPUT_DIR) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0

    img_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"\n[{split}] Found {len(img_paths)} images")

    for img_path in img_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path) as f:
            lines = f.read().strip().splitlines()

        for i, line in enumerate(lines):
            parts = line.split()
            cls = int(parts[0])
            if cls != CROP_CLASS:
                continue

            # YOLO seg format: class x1 y1 x2 y2 ... (polygon) or bbox
            # Handle both bbox (5 values) and segmentation (many values)
            if len(parts) == 5:
                x_c, y_c, bw, bh = map(float, parts[1:5])
            else:
                # Segmentation polygon — compute bounding box
                coords = list(map(float, parts[1:]))
                xs = coords[0::2]
                ys = coords[1::2]
                x_c = (min(xs) + max(xs)) / 2
                y_c = (min(ys) + max(ys)) / 2
                bw  = max(xs) - min(xs)
                bh  = max(ys) - min(ys)

            x1, y1, x2, y2 = yolo_bbox_to_pixels(x_c, y_c, bw, bh, w, h, PADDING)

            if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                skipped += 1
                continue

            crop = img[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
            out_name = f"{img_path.stem}_crop{i}.jpg"
            cv2.imwrite(str(out_dir / out_name), crop_resized)
            saved += 1

    print(f"  Saved : {saved} crops")
    print(f"  Skipped (too small): {skipped}")
    return saved


if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 1 — Extracting cauliflower crops")
    print("=" * 60)

    total = 0
    for split in ["train", "val"]:
        split_dir = Path(DATASET_ROOT) / "images" / split
        if split_dir.exists():
            total += extract_crops(split)
        else:
            print(f"  [skip] {split} split not found at {split_dir}")

    print(f"\n✅ Done. Total crops saved: {total}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"\n👉 Next: run  python step2_train_anomaly_head.py")
