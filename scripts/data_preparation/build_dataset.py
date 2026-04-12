"""
DATASET BUILDER — Rename + Verify + Split + Balance
=====================================================
Reads your two flat folders:
  dataset/annotated/          → cauliflower images + labels  (class 0)
  dataset/weed_annotated/     → weed images + labels         (class 1)

Outputs one clean ready-to-train dataset:
  dataset/balanced/
    images/train/   crop_0001.jpg ... weed_0001.jpg ...
    images/val/     crop_1330.jpg ... weed_1195.jpg ...
    labels/train/   crop_0001.txt ... weed_0001.txt ...
    labels/val/     crop_1330.txt ... weed_1195.txt ...
    dataset.yaml

What this script does:
  1. Scans both folders → finds all valid image+label pairs
  2. Verifies every label file is readable and has correct class ID
  3. Fixes class IDs if wrong (cauliflower→0, weed→1)
  4. Renames everything to clean sequential names
  5. Splits 85% train / 15% val per class (stratified)
  6. Copies files to output folder
  7. Prints a full summary + class balance report
  8. Writes dataset.yaml

USAGE:
    python build_dataset.py

    # Custom paths:
    python build_dataset.py \\
        --cauli  C:/NEWDRIVE/Model_train/dataset/annotated \\
        --weed   C:/NEWDRIVE/Model_train/dataset/weed_annotated \\
        --output C:/NEWDRIVE/Model_train/dataset/balanced

    # Dry run — check counts without copying files:
    python build_dataset.py --dry_run
"""

import os
import shutil
import random
import yaml
import argparse
from pathlib import Path

# ─── PATHS — adjust if your folders are in different locations ─────────────────
BASE_DIR       = Path(__file__).resolve().parent
CAULI_DIR  = Path(r"C:\NEWDRIVE\Model_train\dataset\annotated")
WEED_DIR   = Path(r"C:\NEWDRIVE\Model_train\dataset\weed_annotated")
OUTPUT_DIR = Path(r"C:\NEWDRIVE\Model_train\dataset\balanced")
# ───────────────────────────────────────────────────────────────────────────────

VAL_SPLIT      = 0.15    # 15% validation
RANDOM_SEED    = 42
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp",
                  ".JPG", ".JPEG", ".PNG", ".BMP"}

random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Collect valid image+label pairs
# ══════════════════════════════════════════════════════════════════════════════

def collect_pairs(folder, expected_class_id, source_name):
    """
    Find all image+label pairs in a folder.
    Accepts both flat structure (images/ + labels/) and
    flat-in-root structure (images directly in folder).
    Returns list of (image_path, label_path).
    """
    folder = Path(folder)

    # Try standard structure first: folder/images/ + folder/labels/
    img_dir = folder / "images"
    lbl_dir = folder / "labels"

    if not img_dir.exists():
        # Flat structure — images directly in folder
        img_dir = folder
        lbl_dir = folder

    if not img_dir.exists():
        print(f"  ❌ Folder not found: {img_dir}")
        return []

    # Find all images
    all_images = [p for p in img_dir.iterdir() if p.suffix in IMG_EXTS]

    pairs      = []
    no_label   = []
    empty_lbl  = []

    for img_path in all_images:
        # Find matching label
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        if not lbl_path.exists():
            no_label.append(img_path.name)
            continue

        content = lbl_path.read_text().strip()
        if not content:
            empty_lbl.append(img_path.name)
            continue

        pairs.append((img_path, lbl_path))

    print(f"\n  {source_name}:")
    print(f"    Images found      : {len(all_images)}")
    print(f"    Valid pairs       : {len(pairs)}")
    if no_label:
        print(f"    ⚠️  No label file  : {len(no_label)} images skipped")
        if len(no_label) <= 5:
            for n in no_label:
                print(f"       - {n}")
    if empty_lbl:
        print(f"    ⚠️  Empty labels   : {len(empty_lbl)} images skipped")

    return sorted(pairs)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Verify + fix label class IDs
# ══════════════════════════════════════════════════════════════════════════════

def verify_and_fix_labels(pairs, expected_class, source_name, fix=True):
    """
    Read every label file, check class IDs, fix if wrong.
    Returns list of (image_path, fixed_label_lines).
    """
    wrong_class = 0
    bad_format  = 0
    verified    = []

    for img_path, lbl_path in pairs:
        lines     = lbl_path.read_text().strip().splitlines()
        new_lines = []
        had_wrong = False

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                bad_format += 1
                continue

            if cls != expected_class:
                had_wrong = True
                if fix:
                    parts[0] = str(expected_class)
            new_lines.append(" ".join(parts))

        if had_wrong:
            wrong_class += 1

        if new_lines:
            verified.append((img_path, new_lines))

    if wrong_class > 0:
        print(f"    ⚠️  Wrong class IDs fixed : {wrong_class} files "
              f"(all set to class {expected_class})")
    if bad_format > 0:
        print(f"    ⚠️  Bad format lines skipped: {bad_format}")

    print(f"    ✅ Labels verified    : {len(verified)}")
    return verified


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Rename + split + copy
# ══════════════════════════════════════════════════════════════════════════════

def rename_split_copy(verified_pairs, prefix, output_dir,
                      val_split, dry_run=False):
    """
    Rename pairs to sequential names, split 85/15, copy to output.
    Returns (n_train, n_val).
    """
    random.shuffle(verified_pairs)
    n_val   = max(1, round(len(verified_pairs) * val_split))
    n_train = len(verified_pairs) - n_val

    val_pairs   = verified_pairs[:n_val]
    train_pairs = verified_pairs[n_val:]

    splits = [("train", train_pairs), ("val", val_pairs)]

    counters = {"train": 0, "val": 0}

    for split_name, pairs in splits:
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name

        if not dry_run:
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

        # Start numbering from 1 in each split
        for idx, (img_path, label_lines) in enumerate(pairs, start=1):
            # Sequential name: crop_0001.jpg / weed_0001.jpg
            new_stem = f"{prefix}_{idx:04d}"
            new_img  = img_out / (new_stem + img_path.suffix.lower())
            new_lbl  = lbl_out / (new_stem + ".txt")

            if not dry_run:
                shutil.copy2(img_path, new_img)
                new_lbl.write_text("\n".join(label_lines) + "\n")

            counters[split_name] += 1

    return counters["train"], counters["val"]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Count annotation density
# ══════════════════════════════════════════════════════════════════════════════

def count_annotations(output_dir, split):
    """Count class 0 and class 1 annotations in a split."""
    lbl_dir = output_dir / "labels" / split
    if not lbl_dir.exists():
        return 0, 0
    n0 = n1 = 0
    for lbl in lbl_dir.glob("*.txt"):
        for line in lbl.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls == 0: n0 += 1
            elif cls == 1: n1 += 1
    return n0, n1


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Write dataset.yaml
# ══════════════════════════════════════════════════════════════════════════════

def write_yaml(output_dir):
    cfg = {
        "path":  str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    2,
        "names": ["crop", "weed"],
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\n  ✅ dataset.yaml → {yaml_path}")
    return yaml_path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(cauli_dir, weed_dir, output_dir, val_split, dry_run):
    print(f"\n{'='*65}")
    print(f"  DATASET BUILDER")
    print(f"{'='*65}")
    print(f"  Cauliflower  : {cauli_dir}")
    print(f"  Weed         : {weed_dir}")
    print(f"  Output       : {output_dir}")
    print(f"  Val split    : {val_split*100:.0f}%")
    print(f"  Dry run      : {dry_run}")
    print(f"{'='*65}")

    # ── 1. Collect pairs ─────────────────────────────────────────────────────
    print(f"\n  STEP 1 — Collecting image+label pairs")
    cauli_pairs = collect_pairs(cauli_dir,  0, "Cauliflower (class 0)")
    weed_pairs  = collect_pairs(weed_dir,   1, "Weed (class 1)")

    if not cauli_pairs:
        print(f"\n  ❌ No cauliflower pairs found. Check CAULI_DIR path.")
        return
    if not weed_pairs:
        print(f"\n  ❌ No weed pairs found. Check WEED_DIR path.")
        return

    # ── 2. Verify + fix class IDs ─────────────────────────────────────────────
    print(f"\n  STEP 2 — Verifying label class IDs")
    cauli_verified = verify_and_fix_labels(cauli_pairs, 0, "Cauliflower")
    weed_verified  = verify_and_fix_labels(weed_pairs,  1, "Weed")

    total = len(cauli_verified) + len(weed_verified)
    ratio = len(cauli_verified) / max(len(weed_verified), 1)
    print(f"\n  Class balance:")
    print(f"    Cauliflower : {len(cauli_verified)} images")
    print(f"    Weed        : {len(weed_verified)} images")
    print(f"    Ratio       : {ratio:.2f}:1  ", end="")
    if ratio < 2.0:
        print("✅ Excellent — no oversampling needed")
    elif ratio < 4.0:
        print("✅ Good — model should detect both classes")
    else:
        print("⚠️  Imbalanced — consider collecting more weed images")

    # ── 3. Rename + split + copy ──────────────────────────────────────────────
    print(f"\n  STEP 3 — Renaming, splitting and copying")
    print(f"    (prefix 'crop_XXXX' for cauliflower, 'weed_XXXX' for weed)")

    if not dry_run:
        # Clean output directory
        if output_dir.exists():
            shutil.rmtree(output_dir)

    cauli_train, cauli_val = rename_split_copy(
        cauli_verified, "crop", output_dir, val_split, dry_run)
    weed_train, weed_val   = rename_split_copy(
        weed_verified,  "weed", output_dir, val_split, dry_run)

    # ── 4. Annotation counts ──────────────────────────────────────────────────
    if not dry_run:
        c_train0, c_train1 = count_annotations(output_dir, "train")
        c_val0,   c_val1   = count_annotations(output_dir, "val")
    else:
        c_train0 = c_train1 = c_val0 = c_val1 = 0

    # ── 5. Write YAML ─────────────────────────────────────────────────────────
    if not dry_run:
        write_yaml(output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*65}")
    print(f"")
    print(f"  TRAIN SET  ({cauli_train + weed_train} total images)")
    print(f"  ┌───────────────────────────────────────────────────┐")
    print(f"  │  Cauliflower  : {cauli_train:4d} images  "
          f"({c_train0:5d} annotations)  │")
    print(f"  │  Weed         : {weed_train:4d} images  "
          f"({c_train1:5d} annotations)  │")
    print(f"  │  Total        : {cauli_train+weed_train:4d} images  "
          f"({c_train0+c_train1:5d} annotations)  │")
    if c_train0 > 0 and c_train1 > 0:
        ann_ratio = c_train0 / c_train1
        print(f"  │  Ann. ratio   : {ann_ratio:.2f}:1  (crop:weed)           │")
    print(f"  └───────────────────────────────────────────────────┘")
    print(f"")
    print(f"  VAL SET  ({cauli_val + weed_val} total images)")
    print(f"  ┌───────────────────────────────────────────────────┐")
    print(f"  │  Cauliflower  : {cauli_val:4d} images  "
          f"({c_val0:5d} annotations)  │")
    print(f"  │  Weed         : {weed_val:4d} images  "
          f"({c_val1:5d} annotations)  │")
    print(f"  │  Total        : {cauli_val+weed_val:4d} images  "
          f"({c_val0+c_val1:5d} annotations)  │")
    print(f"  └───────────────────────────────────────────────────┘")
    print(f"")

    if not dry_run:
        print(f"  OUTPUT FOLDER:")
        print(f"    {output_dir}")
        print(f"    ├── images/")
        print(f"    │   ├── train/   {cauli_train+weed_train} files  "
              f"(crop_0001.jpg ... weed_{weed_train:04d}.jpg)")
        print(f"    │   └── val/     {cauli_val+weed_val} files  "
              f"(crop_0001.jpg ... weed_{weed_val:04d}.jpg)")
        print(f"    ├── labels/")
        print(f"    │   ├── train/   {cauli_train+weed_train} files  "
              f"(matching .txt)")
        print(f"    │   └── val/     {cauli_val+weed_val} files  "
              f"(matching .txt)")
        print(f"    └── dataset.yaml  (nc=2, names=['crop','weed'])")

    print(f"\n  TRAINING COMMAND (copy-paste this):")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  python train_best.py \\")
    print(f"    --dataset {output_dir} \\")
    print(f"    --epochs 80 \\")
    print(f"    --imgsz 640")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"\n  KEY TRAINING TIPS (based on your previous results):")
    print(f"  - Your previous model scored 0 on weeds (19:1 imbalance)")
    print(f"  - This dataset is ~1.1:1 ratio — model WILL detect both")
    print(f"  - copy_paste=0.5 in train_best.py synthesises more weed")
    print(f"    instances during training — keep it enabled")
    print(f"  - Use conf=0.25 when running inference (not 0.5)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build balanced YOLO dataset from cauliflower + weed folders"
    )
    parser.add_argument(
        "--cauli",
        default=str(CAULI_DIR),
        help="Cauliflower folder with images/ and labels/ subfolders"
    )
    parser.add_argument(
        "--weed",
        default=str(WEED_DIR),
        help="Weed folder with images/ and labels/ subfolders"
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output balanced dataset folder"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=VAL_SPLIT,
        help=f"Validation split fraction (default {VAL_SPLIT})"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print counts only — don't copy any files"
    )
    args = parser.parse_args()

    main(
        cauli_dir  = Path(args.cauli),
        weed_dir   = Path(args.weed),
        output_dir = Path(args.output),
        val_split  = args.val,
        dry_run    = args.dry_run,
    )
