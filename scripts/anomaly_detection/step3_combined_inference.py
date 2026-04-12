"""
STEP 3 — Combined AnomalyYOLO inference + threshold calibration

Usage:
    python step3_combined_inference.py --calibrate
    python step3_combined_inference.py --test path/to/image.jpg
    python step3_combined_inference.py --benchmark
"""

import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json

# ── CONFIG ────────────────────────────────────────────────────────────────────
BEST_PT_PATH   = r"C:\NEWDRIVE\Model_train\Yolo_try\runs\segment\weed_detect_v3\weights\best.pt"
ANOMALY_PT     = r"C:\New folder\Model_train\anomaly_data\anomaly_head.pt"
VAL_IMAGES_DIR = r"C:\NEWDRIVE\Model_train\dataset\balanced\images\val"
VAL_LABELS_DIR = r"C:\NEWDRIVE\Model_train\dataset\balanced\labels\val"
OUTPUT_DIR     = r"C:\NEWDRIVE\Model_train\anomaly_data\results"
THRESHOLD_FILE = r"C:\NEWDRIVE\Model_train\anomaly_data\threshold.json"

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
# ─────────────────────────────────────────────────────────────────────────────


# ── ANOMALY HEAD ──────────────────────────────────────────────────────────────
class AnomalyHead(nn.Module):
    def __init__(self, in_ch=64, bottleneck=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, bottleneck, 3, padding=1), nn.BatchNorm2d(bottleneck), nn.ReLU(True),
            nn.Conv2d(bottleneck, bottleneck, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(bottleneck, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(True),
            nn.Conv2d(in_ch, in_ch, 1),
        )

    def forward(self, feat):
        return self.dec(self.enc(feat))

    def error_map(self, feat):
        recon = self.forward(feat)
        return ((feat - recon) ** 2).mean(dim=1, keepdim=True)


# ── COMBINED MODEL ────────────────────────────────────────────────────────────
class AnomalyYOLO:
    def __init__(self, yolo_path, anomaly_path):
        print(f"  Loading YOLOv8 from {yolo_path}")
        self.yolo = YOLO(yolo_path)
        self.yolo.model.eval().to(DEVICE)

        print(f"  Loading anomaly head from {anomaly_path}")
        ckpt = torch.load(anomaly_path, map_location=DEVICE, weights_only=False)
        self.p3_idx     = ckpt["p3_layer_idx"]
        self.feature_ch = ckpt["feature_ch"]
        self.crop_size  = ckpt["crop_size"]

        self.head = AnomalyHead(in_ch=self.feature_ch, bottleneck=ckpt["bottleneck_ch"])
        self.head.load_state_dict(ckpt["anomaly_head_state"])
        self.head.eval().to(DEVICE)

        self._feat = None
        layer = self.yolo.model.model[self.p3_idx]
        self._hook = layer.register_forward_hook(self._capture)
        print(f"  Hook at layer {self.p3_idx} | feature_ch={self.feature_ch} | crop_size={self.crop_size}")

        self.error_threshold = 0.05
        if Path(THRESHOLD_FILE).exists():
            with open(THRESHOLD_FILE) as f:
                d = json.load(f)
            self.error_threshold = d.get("threshold", 0.05)
            print(f"  Loaded threshold: {self.error_threshold:.4f}")

    def _capture(self, module, inp, out):
        self._feat = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()

    def preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (self.crop_size, self.crop_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    def get_region_error(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return 0.0
        h, w = img_bgr.shape[:2]
        if h < 8 or w < 8:
            img_bgr = cv2.resize(img_bgr, (32, 32))
        tensor = self.preprocess(img_bgr)
        with torch.no_grad():
            self.yolo.model(tensor)
            feat = self._feat
            if feat is None:
                return 0.0
            err_map = self.head.error_map(feat)
        return float(err_map.mean().cpu())

    def detect(self, img_bgr, use_anomaly=True):
        h_orig, w_orig = img_bgr.shape[:2]
        results = self.yolo.predict(
            img_bgr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            verbose=False, device=DEVICE
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confs, clses):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_orig, x2); y2 = min(h_orig, y2)

            anomaly_error     = 0.0
            is_weed_confirmed = (cls == 1)

            if use_anomaly and (x2 - x1) > 8 and (y2 - y1) > 8:
                crop          = img_bgr[y1:y2, x1:x2]
                anomaly_error = self.get_region_error(crop)
                if cls == 0 and anomaly_error > self.error_threshold:
                    is_weed_confirmed = True
                elif cls == 1:
                    is_weed_confirmed = True

            detections.append({
                "bbox":              (x1, y1, x2, y2),
                "class":             cls,
                "class_name":        "crop" if cls == 0 else "weed",
                "conf":              float(conf),
                "anomaly_error":     round(anomaly_error, 5),
                "is_weed_confirmed": is_weed_confirmed,
            })

        return detections

    def visualise(self, img_bgr, detections, save_path=None):
        vis = img_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            if d["is_weed_confirmed"]:
                color = (0, 0, 220)
                label = f"WEED {d['conf']:.2f} err:{d['anomaly_error']:.3f}"
            elif d["class"] == 0:
                color = (50, 200, 50)
                label = f"CROP {d['conf']:.2f}"
            else:
                color = (0, 165, 255)
                label = f"weed {d['conf']:.2f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if save_path:
            cv2.imwrite(str(save_path), vis)
            print(f"  Saved: {save_path}")
        return vis

    def cleanup(self):
        self._hook.remove()


# ── CALIBRATE THRESHOLD ───────────────────────────────────────────────────────
def calibrate_threshold(model):
    print("\n[Calibrating anomaly threshold]")
    val_images = list(Path(VAL_IMAGES_DIR).glob("*.jpg")) + \
             list(Path(VAL_IMAGES_DIR).glob("*.png")) + \
             list(Path(VAL_IMAGES_DIR).glob("*.jpeg"))
    print(f"  Found {len(val_images)} val images")

    crop_errors = []
    weed_errors = []
    skipped     = 0

    for img_path in tqdm(val_images, desc="Calibrating"):
        label_path = Path(VAL_LABELS_DIR) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path) as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            cls    = int(parts[0])
            coords = list(map(float, parts[1:]))

            # Works for both bbox (4 values) and segmentation polygon (many values)
            xs = coords[0::2]
            ys = coords[1::2]
            x1 = int(max(0,     (min(xs) * w) - 10))
            y1 = int(max(0,     (min(ys) * h) - 10))
            x2 = int(min(w - 1, (max(xs) * w) + 10))
            y2 = int(min(h - 1, (max(ys) * h) + 10))

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w < 4 or crop_h < 4:
                skipped += 1
                continue

            crop = img[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                skipped += 1
                continue

            # Force resize tiny crops
            if crop_w < 32 or crop_h < 32:
                crop = cv2.resize(crop, (32, 32))

            err = model.get_region_error(crop)

            if cls == 0:
                crop_errors.append(err)
            else:
                weed_errors.append(err)

    print(f"  Cauliflower samples: {len(crop_errors)}")
    print(f"  Weed samples:        {len(weed_errors)}")
    print(f"  Skipped (too small): {skipped}")

    if not crop_errors or not weed_errors:
        print("\n  WARNING: Could not calibrate — using default threshold 0.05")
        return 0.05

    crop_mean  = float(np.mean(crop_errors))
    weed_mean  = float(np.mean(weed_errors))
    threshold  = (crop_mean + weed_mean) / 2.0
    separation = (weed_mean - crop_mean) / \
                 (np.std(crop_errors) + np.std(weed_errors) + 1e-6)

    print(f"\n  Cauliflower error: mean={crop_mean:.4f}  std={np.std(crop_errors):.4f}")
    print(f"  Weed error:        mean={weed_mean:.4f}  std={np.std(weed_errors):.4f}")
    print(f"  ✅ Optimal threshold: {threshold:.4f}")
    print(f"  Separation (d'):   {separation:.2f}  "
          f"({'excellent' if separation > 2 else 'good' if separation > 1 else 'marginal'})")

    result = {
        "threshold":      round(threshold, 5),
        "crop_mean":      round(crop_mean, 5),
        "weed_mean":      round(weed_mean, 5),
        "separation":     round(float(separation), 3),
        "n_crop_samples": len(crop_errors),
        "n_weed_samples": len(weed_errors),
    }
    Path(THRESHOLD_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to: {THRESHOLD_FILE}")
    return threshold


# ── BENCHMARK ─────────────────────────────────────────────────────────────────
def benchmark(model):
    print("\n[Benchmark: baseline YOLO vs AnomalyYOLO]")
    val_images = list(Path(VAL_IMAGES_DIR).glob("*.jpg"))

    yolo_only_weeds = 0
    anomaly_extra   = 0
    total_images    = 0

    for img_path in tqdm(val_images, desc="Benchmarking"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        total_images += 1

        base_dets = model.detect(img, use_anomaly=False)
        full_dets = model.detect(img, use_anomaly=True)

        yolo_weeds    = sum(1 for d in base_dets if d["class"] == 1)
        anomaly_weeds = sum(1 for d in full_dets if d["is_weed_confirmed"])

        yolo_only_weeds += yolo_weeds
        anomaly_extra   += max(0, anomaly_weeds - yolo_weeds)

    print(f"\n  Images tested:                      {total_images}")
    print(f"  Weeds detected by YOLO alone:       {yolo_only_weeds}")
    print(f"  Extra weeds caught by anomaly head: {anomaly_extra}")
    extra_pct = 100 * anomaly_extra / max(yolo_only_weeds, 1)
    print(f"  Recall improvement: +{extra_pct:.1f}%")
    print(f"\n  → Paper result: +{extra_pct:.1f}% recall at zero weed annotation cost")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--test",      type=str, default=None)
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model = AnomalyYOLO(BEST_PT_PATH, ANOMALY_PT)

    if args.calibrate:
        threshold = calibrate_threshold(model)
        model.error_threshold = threshold

    elif args.test:
        img = cv2.imread(args.test)
        if img is None:
            print(f"ERROR: Cannot read {args.test}")
            return
        dets     = model.detect(img, use_anomaly=True)
        out_path = Path(OUTPUT_DIR) / ("result_" + Path(args.test).name)
        model.visualise(img, dets, save_path=str(out_path))
        print(f"\n  Detections: {len(dets)}")
        for d in dets:
            flag = " ← ANOMALY CAUGHT" if d["is_weed_confirmed"] and d["class"] == 0 else ""
            print(f"    {d['class_name']:6s} conf={d['conf']:.3f}  "
                  f"err={d['anomaly_error']:.4f}{flag}")

    elif args.benchmark:
        benchmark(model)

    else:
        print("Usage:")
        print("  python step3_combined_inference.py --calibrate")
        print("  python step3_combined_inference.py --test image.jpg")
        print("  python step3_combined_inference.py --benchmark")

    model.cleanup()


if __name__ == "__main__":
    main()