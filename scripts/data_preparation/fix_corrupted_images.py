"""
FIX DATASET SYNC
================
Ensures every label has a matching image and vice versa.
Removes orphaned labels (no image) and orphaned images (no label).
Re-encodes all JPEGs cleanly.
Deletes YOLO cache so it rescans fresh.

Run this before step2_train.py:
    python fix_corrupt_images.py
"""

import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR.parent / 'dataset' / 'crop'
ANNOTATED  = BASE_DIR.parent / 'dataset' / 'annotated'
IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp'}
# ───────────────────────────────────────────────────────────────────────────────


def fix_split(split):
    img_dir = ANNOTATED / 'images' / split
    lbl_dir = ANNOTATED / 'labels' / split

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"⚠️  {split} folders not found — skipping")
        return

    # ── Build sets of stems ──────────────────────────────────────────────────
    img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.suffix == '.txt'}

    orphan_labels = lbl_stems - img_stems   # label exists but no image
    orphan_images = img_stems - lbl_stems   # image exists but no label
    paired        = img_stems & lbl_stems   # both exist

    print(f"\n  [{split}]")
    print(f"    Images        : {len(img_stems)}")
    print(f"    Labels        : {len(lbl_stems)}")
    print(f"    Paired        : {len(paired)}")
    print(f"    Orphan labels : {len(orphan_labels)}  ← will delete")
    print(f"    Orphan images : {len(orphan_images)}  ← will delete")

    # ── Remove orphan labels (no matching image) ─────────────────────────────
    for stem in orphan_labels:
        lbl = lbl_dir / (stem + '.txt')
        lbl.unlink(missing_ok=True)

    # ── Remove orphan images (no matching label) ─────────────────────────────
    for stem in orphan_images:
        for ext in IMG_EXTS:
            img = img_dir / (stem + ext)
            if img.exists():
                img.unlink()
                break

    # ── Re-encode all paired images cleanly ─────────────────────────────────
    print(f"    Re-encoding {len(paired)} paired images...")
    recovered = 0
    failed    = 0

    for stem in tqdm(paired, desc=f"    {split}", unit="img"):
        # Find the image file
        img_path = None
        for ext in IMG_EXTS:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            failed += 1
            continue

        img = cv2.imread(str(img_path))

        # If unreadable, try recovering from original source
        if img is None:
            src = SOURCE_DIR / img_path.name
            if src.exists():
                img = cv2.imread(str(src))
                if img is not None:
                    recovered += 1

        # Still unreadable — remove both image and label
        if img is None:
            img_path.unlink(missing_ok=True)
            (lbl_dir / (stem + '.txt')).unlink(missing_ok=True)
            failed += 1
            continue

        # Re-save as clean JPEG
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"    ✅ Clean       : {len(paired) - failed}")
    if recovered:
        print(f"    🔁 Recovered   : {recovered}")
    if failed:
        print(f"    ❌ Removed     : {failed}  (unrecoverable — label also deleted)")


def main():
    print(f"\n{'='*50}")
    print(f"  Fix & Sync Dataset")
    print(f"{'='*50}")
    print(f"  Source  : {SOURCE_DIR}")
    print(f"  Dataset : {ANNOTATED}")

    if not ANNOTATED.exists():
        print(f"\n❌ Annotated folder not found. Run step1_auto_annotate.py first.")
        return

    fix_split('train')
    fix_split('val')

    # Delete ALL cache files so YOLO rescans from scratch
    print(f"\n🗑️  Clearing YOLO cache files...")
    for cache in ANNOTATED.rglob('*.cache'):
        cache.unlink()
        print(f"   Deleted: {cache.name}")

    # Final count
    n_train = len([p for p in (ANNOTATED/'images'/'train').iterdir()
                   if p.suffix.lower() in IMG_EXTS])
    n_val   = len([p for p in (ANNOTATED/'images'/'val').iterdir()
                   if p.suffix.lower() in IMG_EXTS])

    print(f"\n{'='*50}")
    print(f"  ✅ Dataset synced and clean!")
    print(f"     Train : {n_train} images")
    print(f"     Val   : {n_val} images")
    print(f"{'='*50}")
    print(f"\n👉 Now run: python step2_train.py")


if __name__ == '__main__':
    main()