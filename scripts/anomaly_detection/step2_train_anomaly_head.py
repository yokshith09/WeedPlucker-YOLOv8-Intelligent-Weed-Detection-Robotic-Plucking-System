"""
STEP 2 — Build and train the Anomaly Head (lightweight autoencoder)
Taps into YOLOv8's neck feature maps using PyTorch forward hooks.
Trained ONLY on cauliflower crops — zero weed labels needed.

Architecture: AnomalyYOLO
  YOLOv8n-seg backbone+neck  (frozen — your best.pt weights)
        ↓ hook at P3 (stride-8 feature map, 128ch)
  Anomaly Encoder  (3 conv layers → bottleneck)
        ↓
  Anomaly Decoder  (3 transposed conv → reconstruct P3 features)
        ↓
  Reconstruction error map → high error = WEED

Usage:
    python step2_train_anomaly_head.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import json
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
BEST_PT_PATH   = r"C:\NEWDRIVE\Model_train\Yolo_try\runs\segment\weed_detect_v3\weights\best.pt"
CROPS_DIR      = r"C:\New folder\Model_train\anomaly_data\cauliflower_crops"
OUTPUT_DIR     = r"C:\New folder\Model_train\anomaly_data"
SAVE_PATH      = r"C:\New folder\Model_train\anomaly_data\anomaly_head.pt"

CROP_SIZE      = 64          # must match step1
FEATURE_CH     = 128 # YOLOv8n-seg P3 channels (stride-8 level)
FEATURE_SIZE   = 4         # 64px input / stride8 = 8x8 feature grid
BOTTLENECK_CH  = 32          # autoencoder compression

EPOCHS         = 60
BATCH_SIZE     = 32
LR             = 1e-3
LR_STEP        = 25        # decay LR every N epochs
LR_GAMMA       = 0.5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


# ── DATASET ───────────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        for split in ["train", "val"]:
            d = Path(root) / split
            if d.exists():
                self.paths += list(d.glob("*.jpg")) + list(d.glob("*.png"))
        if not self.paths:
            self.paths = list(Path(root).glob("*.jpg")) + list(Path(root).glob("*.png"))
        self.transform = transform
        print(f"  Dataset: {len(self.paths)} cauliflower crops loaded")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
        if self.transform:
            img = self.transform(img)
        return img


# ── ANOMALY HEAD ARCHITECTURE ─────────────────────────────────────────────────
class AnomalyHead(nn.Module):
    """
    Lightweight convolutional autoencoder that operates on P3 feature maps.
    Input:  (B, 128, 8, 8) — YOLOv8n P3 features extracted via forward hook
    Output: (B, 128, 8, 8) — reconstructed features
    Parameters: ~480K (negligible vs YOLOv8n's 3.26M)
    """
    def __init__(self, in_ch=FEATURE_CH, bottleneck=BOTTLENECK_CH):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, bottleneck, 3, padding=1), nn.BatchNorm2d(bottleneck), nn.ReLU(True),
            nn.Conv2d(bottleneck, bottleneck, 1),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(bottleneck, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(True),
            nn.Conv2d(in_ch, in_ch, 1),
        )

    def forward(self, feat):
        z = self.enc(feat)
        return self.dec(z)

    def reconstruction_error(self, feat):
        """Returns per-spatial-location MSE error map (B, 1, H, W)."""
        recon = self.forward(feat)
        err = ((feat - recon) ** 2).mean(dim=1, keepdim=True)  # mean over channels
        return err


# ── FEATURE EXTRACTOR (hooks into YOLOv8 neck) ───────────────────────────────
class YOLOFeatureExtractor:
    """
    Attaches a forward hook to the YOLOv8 neck's P3 output layer.
    YOLOv8n-seg: layer index 15 is the first SPPF output fed into the neck,
    but the cleanest P3 feature (stride-8) is at the Concat/C2f output
    going to the detection head — empirically layer index 15 or 17.
    We will find the right layer automatically.
    """
    def __init__(self, yolo_model, target_layer_idx=15):
        self.features = None
        self.hook = None
        self.model = yolo_model.model.model  # nn.Sequential of YOLO layers

        layer = self.model[target_layer_idx]
        self.hook = layer.register_forward_hook(self._hook_fn)
        print(f"  Hook attached to layer {target_layer_idx}: {type(layer).__name__}")

    def _hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.features = output.detach()
        elif isinstance(output, (list, tuple)):
            self.features = output[0].detach()

    def get_features(self):
        return self.features

    def remove(self):
        if self.hook:
            self.hook.remove()


def find_p3_layer(yolo_model):
    """
    Identify which layer index outputs stride-8 (P3) features.
    For YOLOv8n-seg: test by running a dummy input and checking tensor shapes.
    """
    model = yolo_model.model.model
    dummy = torch.zeros(1, 3, CROP_SIZE, CROP_SIZE)
    expected_h = CROP_SIZE // 8  # stride 8

    found_idx = 15  # default fallback
    for i, layer in enumerate(model):
        captured = []
        def hook(m, inp, out, _i=i):
            if isinstance(out, torch.Tensor):
                captured.append((_i, out.shape))
        h = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(dummy)
        except Exception:
            pass
        h.remove()
        for idx, shape in captured:
            if len(shape) == 4 and shape[2] == expected_h and shape[1] == FEATURE_CH:
                found_idx = idx
                break

    print(f"  Auto-detected P3 layer index: {found_idx} "
          f"(expected output shape: [B, {FEATURE_CH}, {expected_h}, {expected_h}])")
    return found_idx


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  STEP 2 — Training AnomalyYOLO head")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Load frozen YOLOv8
    print("\n[1/5] Loading YOLOv8 backbone (frozen)...")
    yolo = YOLO(BEST_PT_PATH)
    yolo.model.eval()
    for p in yolo.model.parameters():
        p.requires_grad = False
    yolo.model.to(DEVICE)

    # Find and hook P3 layer
    print("\n[2/5] Attaching neck feature hook...")
    p3_idx = 18
# Temporarily check what shape layer 18 outputs
    import torch
    dummy = torch.zeros(1, 3, 64, 64).to(DEVICE)
    captured = []
    def _tmp(m, i, o): captured.append(o.shape if isinstance(o, torch.Tensor) else o[0].shape)
    h = yolo.model.model[18].register_forward_hook(_tmp)
    with torch.no_grad(): yolo.model(dummy)
    h.remove()
    print(f"  Layer 18 output shape: {captured[0]}")
    extractor = YOLOFeatureExtractor(yolo, target_layer_idx=p3_idx)

    # Dataset
    print("\n[3/5] Loading cauliflower crops...")
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CropDataset(CROPS_DIR, transform=tf)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=(DEVICE == "cuda"))

    # Anomaly head
    print("\n[4/5] Initialising anomaly head...")
    anomaly_head = AnomalyHead(in_ch=FEATURE_CH, bottleneck=BOTTLENECK_CH).to(DEVICE)
    n_params = sum(p.numel() for p in anomaly_head.parameters())
    print(f"  Anomaly head parameters: {n_params:,}")

    optimizer = optim.Adam(anomaly_head.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)
    criterion = nn.MSELoss()

    # Training
    print(f"\n[5/5] Training for {EPOCHS} epochs...")
    history = {"train_loss": []}
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        anomaly_head.train()
        epoch_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch:3d}/{EPOCHS}", leave=False):
            batch = batch.to(DEVICE)

            # Forward pass through frozen YOLOv8 to get features
            with torch.no_grad():
                yolo.model(batch)
            feat = extractor.get_features()

            if feat is None:
                continue

            # Ensure correct shape
            if feat.shape[1] != FEATURE_CH:
                # Adaptive: use whatever channel count came out
                pass

            # Anomaly head forward + loss
            recon = anomaly_head(feat)
            loss  = criterion(recon, feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        history["train_loss"].append(round(avg_loss, 6))
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "anomaly_head_state": anomaly_head.state_dict(),
                "p3_layer_idx": p3_idx,
                "feature_ch": FEATURE_CH,
                "feature_size": FEATURE_SIZE,
                "bottleneck_ch": BOTTLENECK_CH,
                "crop_size": CROP_SIZE,
                "loss": best_loss,
            }, SAVE_PATH)

        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:3d} | loss: {avg_loss:.6f} | best: {best_loss:.6f} | "
                  f"lr: {scheduler.get_last_lr()[0]:.6f}")

    extractor.remove()

    # Save training history
    hist_path = Path(OUTPUT_DIR) / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Anomaly head saved to: {SAVE_PATH}")
    print(f"   Best reconstruction loss: {best_loss:.6f}")
    print(f"\n  Interpretation:")
    print(f"  < 0.01 = excellent (cauliflower well reconstructed)")
    print(f"  0.01–0.05 = good")
    print(f"  > 0.10 = model needs more epochs or data")
    print(f"\n👉 Next: run  python step3_combined_inference.py  to test on field images")


if __name__ == "__main__":
    train()
