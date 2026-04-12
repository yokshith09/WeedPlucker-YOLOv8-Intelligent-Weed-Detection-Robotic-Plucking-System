# Data Preparation Scripts

Tools for building, annotating, and cleaning the cauliflower + weed dataset.

## Pipeline Order

```
1. step1_auto_annotate.py      → Auto-annotate crop images (HSV color segmentation)
2. annotate_cauliflower.py     → Advanced annotation (ExG + HSV + Watershed, v3)
3. annotate_weeds.py           → Annotate weed images
4. create_seg_labels.py        → Convert bounding box labels → segmentation polygons
5. fix_corrupted_images.py     → Fix dataset sync, re-encode images, remove orphans
6. build_dataset.py            → Merge crop + weed data, balance, split 85/15
7. remove_empty_files.py       → Clean up empty label files
8. weed_labels_checking.py     → Quick sanity check on label counts
```

## Key Scripts

### `annotate_cauliflower.py` / `annotate_weeds.py`
Advanced auto-annotation pipeline using:
- **ExG (Excess Green Index)** — `2G - R - B` vegetation detection
- **HSV color masking** — tuned for cauliflower leaf tones
- **Sky zone exclusion** — removes background trees and sky
- **Watershed splitting** — separates touching plants into instances

```bash
# Preview annotations first
python annotate_cauliflower.py --source ./images/train --preview_only

# Tune parameters
python annotate_cauliflower.py --source ./images/train --sky 0.35 --exg 0.03

# Generate final annotations
python annotate_cauliflower.py --source ./images/train --output ./dataset --split train
```

### `build_dataset.py`
Merges cauliflower (class 0) and weed (class 1) folders into a balanced dataset:
- Verifies all labels have correct class IDs
- Fixes wrong class IDs automatically
- Renames to clean sequential names (`crop_0001.jpg`, `weed_0001.jpg`)
- Stratified 85/15 train/val split
- Writes `dataset.yaml`

```bash
python build_dataset.py
python build_dataset.py --dry_run  # preview counts only
```
