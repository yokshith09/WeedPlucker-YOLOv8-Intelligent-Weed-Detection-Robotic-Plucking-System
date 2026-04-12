# Training Scripts

## `train_best.py` — Main Training Script

The primary training script combining all best practices:

- **Transfer learning** from existing best.pt (not from scratch)
- **Class imbalance handling** via dynamic `cls_weight` (sqrt ratio, capped at 3.0)
- **Heavy augmentation**: mosaic, copy_paste=0.5, mixup, flipud, degrees, erasing
- **Auto device detection**: CUDA → MPS → CPU with appropriate batch sizes
- **Early stopping** with patience=20
- **Overlap mask** for better segmentation when plants overlap

### Usage

```bash
# Auto-detect everything (recommended)
python train_best.py

# Customize
python train_best.py --epochs 80 --imgsz 640 --batch 16

# Resume interrupted training
python train_best.py --resume

# Train from scratch (no transfer learning)
python train_best.py --scratch

# Evaluate only (no training)
python train_best.py --eval
```

### Key Training Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `copy_paste` | 0.5 | Synthetically injects weed instances into crop images |
| `cls_weight` | dynamic | Penalizes weed misses more (√(crop/weed) ratio) |
| `overlap_mask` | True | Better masks when plants overlap |
| `patience` | 20 | Prevents premature early stopping |
| `lr0` | 0.005 | Low LR for fine-tuning from existing weights |
| `nbs` | 64 | Normalized batch size for stable LR |

---

## `simclr_pipeline.py` — Self-Supervised Pretraining

SimCLR (Simple Contrastive Learning of Representations) pipeline that:

1. **Pretrains** YOLOv8 backbone on unlabelled field images using NT-Xent contrastive loss
2. **Injects** the pretrained weights back into YOLOv8
3. **Fine-tunes** the enhanced model on the labelled dataset

### Usage

```bash
# Step 1: Pretrain backbone on unlabelled images
python simclr_pipeline.py --mode pretrain \
    --image_dir ./unlabelled_field_images/ \
    --base_weights best.pt

# Step 2: Fine-tune with pretrained backbone
python simclr_pipeline.py --mode retrain \
    --data_yaml dataset.yaml \
    --base_weights best.pt \
    --simclr_weights simclr_backbone_weights.pt
```

### Architecture

```
YOLOv8 Backbone (layers 0-9)
    ├── Layers 0-4: FROZEN during SimCLR pretraining
    ├── Layers 5-9: TRAINED with contrastive loss
    └── Projection Head: 256 → 256 → 128 (MLP, discarded after pretraining)

NT-Xent Loss: temperature=0.5, batch_size=32
Augmentation: RandomResizedCrop, ColorJitter(0.8), RandomGrayscale(0.2)
```
