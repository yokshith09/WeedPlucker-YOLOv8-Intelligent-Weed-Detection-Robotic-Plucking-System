"""
BEST TRAINING SCRIPT — Cauliflower + Weed (YOLOv8n-seg)
=========================================================
Merges the best of train_yolo.py and step6_retrain.py.

Fixes applied vs step6_retrain.py:
  1. imgsz=640 for training (better accuracy), 320 only for Pi export
  2. patience increased to 20 (your model stopped at 86 — was stopping too early)
  3. copy_paste raised to 0.5 (more weed instance injection)
  4. workers fixed (was 0, now auto — faster dataloading)
  5. cls_weight capped smarter (max 3.0 not 5.0 — avoids instability)
  6. Added overlap_mask=True (better seg masks when plants overlap)
  7. Added nbs=64 (normalised batch size — stable LR regardless of batch)
  8. save_period removed (save only best/last — saves disk space)
  9. Proper resume logic (won't accidentally overwrite a good run)
 10. Auto CUDA/MPS/CPU with correct batch sizes per device

Fixes applied vs train_yolo.py:
  1. Starts from your existing best.pt (transfer learning, not scratch)
  2. Adds all augmentation (mosaic, copy_paste, mixup, etc.)
  3. Adds class weighting for weed imbalance
  4. Adds early stopping
  5. Adds lower LR for fine-tuning

Usage:
    python train_best.py                         # auto-detect everything
    python train_best.py --epochs 80             # more epochs
    python train_best.py --imgsz 640             # full resolution (recommended)
    python train_best.py --resume                # continue from last.pt
    python train_best.py --scratch               # ignore best.pt, train fresh
    python train_best.py --eval                  # only evaluate, no training
"""

import argparse
import os
import torch
import yaml
from pathlib import Path

os.environ["YOLO_DISABLE_GIT_CHECK"] = "1"

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DEFAULT_DATASET = str(BASE_DIR.parent / 'dataset' / 'annotated')
PROJECT_NAME    = str(BASE_DIR / 'runs' / 'segment')
RUN_NAME        = 'weed_detect_v3'
# ───────────────────────────────────────────────────────────────────────────────

# ─── DEVICE CONFIG ─────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEFAULT_DEVICE = 'mps'
    DEFAULT_BATCH  = 8     # 640px on MPS — keep batch small to avoid OOM
    _device_label  = 'Mac Apple Silicon (MPS)'
elif torch.cuda.is_available():
    DEFAULT_DEVICE = 0
    # Scale batch by VRAM: 8GB→16, 12GB→32, 24GB→64
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    DEFAULT_BATCH  = 16 if vram_gb < 10 else (32 if vram_gb < 20 else 64)
    _device_label  = f'CUDA — {torch.cuda.get_device_name(0)} ({vram_gb:.0f}GB)'
else:
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_BATCH  = 4
    _device_label  = 'CPU (slow — consider Google Colab)'
# ───────────────────────────────────────────────────────────────────────────────


def find_prev_model(scratch=False):
    """
    Find best existing model to warm-start from.
    Transfer learning from cauliflower model = faster convergence on weeds.
    """
    if scratch:
        print(f"  Starting from scratch: yolov8n-seg.pt")
        return 'yolov8n-seg.pt'

    # Search in order: v2 → v1 → any best.pt
    search_order = ['weed_detect_v2', 'weed_detect_v1', 'weed_detect',
                    'weed_detect7', 'weed_detect6', 'weed_detect5']
    for name in search_order:
        p = BASE_DIR / 'runs' / 'segment' / name / 'weights' / 'best.pt'
        if p.exists():
            print(f"  Warm-starting from: {p}")
            return str(p)

    print(f"  No previous model found — starting from yolov8n-seg.pt")
    return 'yolov8n-seg.pt'


def update_yaml(dataset_path):
    """Ensure dataset.yaml is set to 2 classes."""
    yaml_path = Path(dataset_path) / 'dataset.yaml'

    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            'path':  str(Path(dataset_path).resolve()),
            'train': 'images/train',
            'val':   'images/val',
        }

    cfg['nc']    = 2
    cfg['names'] = ['crop', 'weed']

    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"  ✅ dataset.yaml: nc=2, names=['crop','weed']")
    return str(yaml_path)


def verify_dataset(dataset_path):
    """Remove orphan labels (labels with no matching image), wipe cache."""
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    removed  = 0
    dp       = Path(dataset_path)

    for split in ['train', 'val']:
        img_dir = dp / 'images' / split
        lbl_dir = dp / 'labels' / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        on_disk = {p.stem for p in img_dir.iterdir()
                   if p.suffix.lower() in img_exts}
        for lbl in list(lbl_dir.iterdir()):
            if lbl.suffix == '.txt' and lbl.stem not in on_disk:
                lbl.unlink(missing_ok=True)
                removed += 1

    for cache in dp.rglob('*.cache'):
        cache.unlink()

    return removed


def count_classes(dataset_path):
    """Count crop vs weed annotations in train split."""
    dp         = Path(dataset_path)
    crop_count = 0
    weed_count = 0
    lbl_dir    = dp / 'labels' / 'train'

    if not lbl_dir.exists():
        return 0, 0

    for lbl in lbl_dir.iterdir():
        if lbl.suffix != '.txt':
            continue
        for line in lbl.read_text().strip().splitlines():
            if line.strip():
                cls = int(line.split()[0])
                if cls == 0:
                    crop_count += 1
                else:
                    weed_count += 1

    return crop_count, weed_count


def compute_class_weight(n_crop, n_weed):
    """
    Compute weed class weight for imbalanced dataset.
    
    Formula: sqrt(n_crop / n_weed) — square root dampens extreme ratios.
    Cap at 3.0 — above this, training becomes unstable (loss spikes).
    
    Example:
      1300 crop, 200 weed → ratio 6.5 → sqrt = 2.55 → weight = 2.55
      1300 crop,  50 weed → ratio 26  → sqrt = 5.1  → capped at 3.0
    """
    if n_crop == 0 or n_weed == 0:
        return 1.0
    import math
    weight = math.sqrt(n_crop / n_weed)
    return round(min(weight, 3.0), 2)


def train(dataset_path, epochs, batch, imgsz, device, resume, scratch):
    from ultralytics import YOLO

    dataset_path = Path(dataset_path)

    # ── Pre-flight ────────────────────────────────────────────────────────────
    print(f"\n🔍 Pre-flight check...")
    removed = verify_dataset(dataset_path)
    if removed:
        print(f"   ⚠️  Removed {removed} orphan labels")
    else:
        print(f"   ✅ Dataset clean")

    yaml_path      = update_yaml(dataset_path)
    n_crop, n_weed = count_classes(dataset_path)

    print(f"   Crop annotations : {n_crop}")
    print(f"   Weed annotations : {n_weed}")

    if n_weed == 0:
        print(f"\n❌ No weed labels found in {dataset_path}/labels/train/")
        print(f"   Add weed images + labels first.")
        return None

    cls_weight = compute_class_weight(n_crop, n_weed)
    print(f"   Class weight     : {cls_weight}x  (weed penalty)")

    # ── Image count ───────────────────────────────────────────────────────────
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    n_train  = len([p for p in (dataset_path/'images'/'train').iterdir()
                    if p.suffix.lower() in img_exts])
    n_val    = len([p for p in (dataset_path/'images'/'val').iterdir()
                    if p.suffix.lower() in img_exts])

    # ── Time estimate ─────────────────────────────────────────────────────────
    sec_per_batch  = 8.0 if imgsz == 640 else 3.5
    batches        = n_train // batch
    mins_per_epoch = (batches * sec_per_batch) / 60
    est_hrs        = (mins_per_epoch * epochs) / 60

    print(f"\n{'='*60}")
    print(f"  TRAINING: Cauliflower + Weed (YOLOv8n-seg)")
    print(f"{'='*60}")
    print(f"  Device    : {_device_label}")
    print(f"  Image sz  : {imgsz}px")
    print(f"  Epochs    : {epochs}")
    print(f"  Batch     : {batch}")
    print(f"  Train     : {n_train}  |  Val: {n_val}")
    print(f"  Estimated : ~{est_hrs:.1f} hrs")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    if resume:
        ckpt = Path(PROJECT_NAME) / RUN_NAME / 'weights' / 'last.pt'
        if ckpt.exists():
            print(f"  Resuming from: {ckpt}")
            model = YOLO(str(ckpt))
        else:
            print(f"  ⚠️  last.pt not found — starting fresh")
            model = YOLO(find_prev_model(scratch))
    else:
        model = YOLO(find_prev_model(scratch))

    # ── Train ─────────────────────────────────────────────────────────────────
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,        # 640 for training (better accuracy)
        batch=batch,
        device=device,

        # ── Augmentation ─────────────────────────────────────────────────────
        # These are critical when weed images are fewer than crop images
        mosaic=1.0,          # mix 4 images — creates diverse scenes
        copy_paste=0.5,      # ★ KEY: pastes weed instances into crop images
                             #   This synthetically adds more weed training data
        mixup=0.1,           # blend images — helps generalisation
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,        # rotation
        translate=0.1,
        scale=0.5,           # zoom in/out
        shear=2.0,
        perspective=0.0005,
        hsv_h=0.02,          # hue shift — different lighting
        hsv_s=0.7,           # saturation
        hsv_v=0.4,           # brightness
        erasing=0.4,         # random erasing — occlusion robustness

        # ── Class imbalance ───────────────────────────────────────────────────
        cls=cls_weight,      # weed misses penalised more

        # ── Mask quality ─────────────────────────────────────────────────────
        overlap_mask=True,   # better masks when plants overlap each other

        # ── Optimiser ────────────────────────────────────────────────────────
        lr0=0.005,           # low initial LR (fine-tuning from best.pt)
        lrf=0.001,           # final LR = lr0 × lrf
        warmup_epochs=3,     # ramp up LR for first 3 epochs
        nbs=64,              # normalised batch size — keeps LR stable
                             # regardless of your actual batch size

        # ── Regularisation ───────────────────────────────────────────────────
        weight_decay=0.0005,
        dropout=0.0,

        # ── Training control ─────────────────────────────────────────────────
        patience=20,         # stop if no improvement for 20 epochs
                             # (your previous run stopped at 86 with patience=10
                             #  — this gives more room to improve)
        workers=2,           # 2 dataloader workers (was 0 — very slow)
        cache=False,         # don't cache to RAM (safe for all machine sizes)
        amp=True,            # mixed precision — faster on GPU/MPS

        # ── Output ───────────────────────────────────────────────────────────
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=resume,
        save=True,
        plots=True,
        val=True,
    )

    best = Path(PROJECT_NAME) / RUN_NAME / 'weights' / 'best.pt'

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"  Best model : {best}")
    print(f"{'='*60}")

    try:
        box_map  = results.results_dict.get('metrics/mAP50(B)', 0)
        mask_map = results.results_dict.get('metrics/mAP50(M)', 0)
        print(f"\n  Box  mAP@50 : {box_map:.4f}")
        print(f"  Mask mAP@50 : {mask_map:.4f}")

        if mask_map > 0.65:
            print(f"  ✅ Good — ready for robot deployment")
        elif mask_map > 0.45:
            print(f"  ⚠️  Acceptable — works, but collect more field weeds to improve")
        else:
            print(f"  ❌ Low — need more field-specific weed images")
            print(f"     Tip: Even 80-100 photos of your actual field weeds")
            print(f"          will improve accuracy more than any setting change.")
    except Exception:
        pass

    print(f"\n👉 Next steps:")
    print(f"   Evaluate : python train_best.py --eval")
    print(f"   Export   : python export_for_pi.py")
    return str(best)


def evaluate(dataset_path, model_path):
    """Evaluate model on validation set."""
    from ultralytics import YOLO
    yaml_path = Path(dataset_path) / 'dataset.yaml'
    if not yaml_path.exists():
        print(f"❌ dataset.yaml not found at {dataset_path}")
        return

    print(f"\n  Evaluating: {model_path}")
    model   = YOLO(model_path)
    metrics = model.val(data=str(yaml_path), imgsz=640)

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Box  mAP@50    : {metrics.box.map50:.4f}")
    print(f"  Box  mAP@50-95 : {metrics.box.map:.4f}")
    print(f"  Mask mAP@50    : {metrics.seg.map50:.4f}")
    print(f"  Mask mAP@50-95 : {metrics.seg.map:.4f}")
    print(f"\n  Interpretation:")
    print(f"   > 0.65 mask mAP50 = good for robot use")
    print(f"   > 0.50 mask mAP50 = usable, coordinates will work")
    print(f"   < 0.40 mask mAP50 = needs more training data")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Best YOLOv8-seg training script for cauliflower + weed"
    )
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help='Path to annotated dataset folder')
    parser.add_argument('--epochs',  type=int, default=80,
                        help='Number of training epochs (default: 80)')
    parser.add_argument('--batch',   type=int, default=DEFAULT_BATCH,
                        help=f'Batch size (default: {DEFAULT_BATCH} for your device)')
    parser.add_argument('--imgsz',   type=int, default=640,
                        help='Image size (default: 640 — use 320 only if very slow)')
    parser.add_argument('--device',  default=str(DEFAULT_DEVICE),
                        help='Device: 0 (CUDA), mps, cpu')
    parser.add_argument('--resume',  action='store_true',
                        help='Resume from last.pt checkpoint')
    parser.add_argument('--scratch', action='store_true',
                        help='Train from scratch (ignore existing best.pt)')
    parser.add_argument('--eval',    action='store_true',
                        help='Only evaluate — no training')
    parser.add_argument('--eval_model', default=None,
                        help='Model path for evaluation (default: auto-find best.pt)')
    args = parser.parse_args()

    if args.eval:
        model_path = (args.eval_model or
                      f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
        evaluate(args.dataset, model_path)
    else:
        best = train(
            args.dataset,
            args.epochs,
            args.batch,
            args.imgsz,
            args.device,
            args.resume,
            args.scratch,
        )
        if best:
            evaluate(args.dataset, best)
