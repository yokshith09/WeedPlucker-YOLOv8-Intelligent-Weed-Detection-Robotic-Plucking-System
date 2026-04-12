
"""
LAPTOP TEST — Hybrid Weed Detector
====================================
Tests your best.pt / best.onnx on images.

Outputs:
  - Segmentation masks drawn on image (saved as PNG)
  - Cauliflower pixel coordinates  (cx, cy, Z_fixed)
  - Weed pixel coordinates         (cx, cy, Z_fixed)
  - JSON results file per image

Z is fixed depth — set Z_FIXED_MM below to your robot's camera height.

Usage:
    # Test on your 3 images:
    python test_laptop.py --model best.pt   --source IMG_20250904_160815466.jpg
    python test_laptop.py --model best.onnx --source IMG_20250904_160815466.jpg

    # Test on a whole folder:
    python test_laptop.py --model best.pt --source ./images/

    # Tune ExG sensitivity if too many / too few weeds detected:
    python test_laptop.py --model best.pt --source ./images/ --exg 0.06
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path

# ─── CONFIGURE THESE ──────────────────────────────────────────────────────────
Z_FIXED_MM        = 500     # mm — your camera height above ground (fixed)
CONF_THRESH       = 0.25    # lower = detects more cauliflower (try 0.20 if missing)
IOU_THRESH        = 0.45
INPUT_SIZE        = 320     # must match what you trained with

# ExG vegetation index thresholds
EXG_THRESHOLD     = 0.08    # 0.05 = very sensitive | 0.15 = less sensitive
MIN_WEED_AREA_PX  = 80      # ignore blobs smaller than this (noise/soil)
MAX_WEED_AREA_PX  = 20000   # ignore huge blobs (background trees/grass)
CAULI_BUFFER_PX   = 30      # safety margin around cauliflower (pixels)
MORPH_KERNEL      = 5       # morphology cleanup kernel
# ───────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — YOLOv8 Cauliflower Segmentation
# ══════════════════════════════════════════════════════════════════════════════

def detect_cauliflower(model, frame):
    """
    Run YOLOv8-seg inference.
    Returns:
        combined_mask : H×W uint8 (255 = cauliflower pixel)
        crops         : list of dicts with pixel coords + seg mask
    """
    h, w = frame.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    results = model(frame,
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    imgsz=INPUT_SIZE,
                    verbose=False)[0]

    crops = []

    if results.masks is None or results.boxes is None:
        return combined_mask, crops

    raw_masks = results.masks.data.cpu().numpy()   # (N, mh, mw)
    boxes_xywh = results.boxes.xywh.cpu().numpy()  # (N, 4) cx cy w h
    boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # (N, 4) x1 y1 x2 y2
    confs      = results.boxes.conf.cpu().numpy()

    for idx, (raw_mask, box_xywh, box_xyxy, conf) in enumerate(
            zip(raw_masks, boxes_xywh, boxes_xyxy, confs)):

        # Resize mask to full image size
        mask_full = cv2.resize(raw_mask, (w, h))
        mask_bin  = (mask_full > 0.5).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(combined_mask, mask_bin)

        # Centroid from mask pixels (more accurate than bbox center)
        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())

        x1, y1, x2, y2 = map(int, box_xyxy)

        crops.append({
            "id":            idx,
            "class":         "cauliflower",
            "confidence":    round(float(conf), 3),
            "centroid_px":   {"x": cx, "y": cy},
            "centroid_norm": {"x": round(cx/w, 4), "y": round(cy/h, 4)},
            "Z_mm":          Z_FIXED_MM,
            "bbox_px":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "mask_area_px":  int(len(xs)),
            "_mask":         mask_bin,   # internal — removed before JSON save
        })

    return combined_mask, crops


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — ExG Vegetation Index
# ══════════════════════════════════════════════════════════════════════════════

def excess_green_mask(bgr_img):
    """
    ExG = 2G - R - B  (Woebbecke et al. 1995)
    All-green vegetation on plain RGB — no NIR needed.
    Returns binary mask (255 = green vegetation).
    """
    f = bgr_img.astype(np.float32) / 255.0
    b, g, r = cv2.split(f)
    exg = 2.0 * g - r - b

    mask = (exg > EXG_THRESHOLD).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, exg


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Weed Localization by Exclusion
# ══════════════════════════════════════════════════════════════════════════════

def find_weeds(vegetation_mask, cauliflower_mask, img_h, img_w):
    """
    Weed mask = All green vegetation  MINUS  cauliflower regions
    Connected components → individual weed instances with pixel coords.
    """
    # Dilate cauliflower mask → safety buffer (don't flag edges as weed)
    buf_kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (CAULI_BUFFER_PX, CAULI_BUFFER_PX))
    cauli_buffered = cv2.dilate(cauliflower_mask, buf_kernel, iterations=1)

    # Remove cauliflower from vegetation map
    weed_mask = cv2.bitwise_and(vegetation_mask,
                                cv2.bitwise_not(cauli_buffered))

    # Connected components = individual weed plants
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        weed_mask, connectivity=8
    )

    weeds = []
    weed_id = 0
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (MIN_WEED_AREA_PX <= area <= MAX_WEED_AREA_PX):
            continue

        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        x1 = int(stats[i, cv2.CC_STAT_LEFT])
        y1 = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Individual weed mask
        weed_instance_mask = (labels == i).astype(np.uint8) * 255

        weeds.append({
            "id":            weed_id,
            "class":         "weed",
            "centroid_px":   {"x": cx, "y": cy},
            "centroid_norm": {"x": round(cx/img_w, 4), "y": round(cy/img_h, 4)},
            "Z_mm":          Z_FIXED_MM,
            "bbox_px":       {"x1": x1, "y1": y1, "x2": x1+bw, "y2": y1+bh},
            "mask_area_px":  area,
            "_mask":         weed_instance_mask,  # internal
        })
        weed_id += 1

    return weeds, weed_mask


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Rich Annotation Rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_annotated(frame, crops, weeds, weed_mask, cauli_mask):
    """
    Draws:
      - Semi-transparent coloured segmentation masks
      - Bounding boxes
      - Centroid dots
      - Pixel coordinate labels (cx, cy, Z)
      - Summary panel
    """
    out  = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()

    # ── Cauliflower masks — green fill ──────────────────────────────────────
    for c in crops:
        mask = c["_mask"]
        overlay[mask > 0] = (overlay[mask > 0] * 0.4 +
                             np.array([30, 200, 30]) * 0.6).astype(np.uint8)

    # ── Weed mask — red fill (all weeds combined) ───────────────────────────
    overlay[weed_mask > 0] = (overlay[weed_mask > 0] * 0.4 +
                              np.array([30, 30, 220]) * 0.6).astype(np.uint8)

    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # ── Cauliflower boxes + labels ──────────────────────────────────────────
    for c in crops:
        b  = c["bbox_px"]
        cx = c["centroid_px"]["x"]
        cy = c["centroid_px"]["y"]
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                      (0, 220, 0), 2)
        cv2.circle(out, (cx, cy), 7, (0, 255, 0), -1)

        label = f"CAULI ({cx},{cy}) Z={Z_FIXED_MM}mm"
        cv2.putText(out, label,
                    (b["x1"], max(b["y1"]-7, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.putText(out, f"conf:{c['confidence']:.2f}",
                    (b["x1"], max(b["y1"]-20, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1)

    # ── Weed boxes + labels ─────────────────────────────────────────────────
    for i, weed in enumerate(weeds):
        b  = weed["bbox_px"]
        cx = weed["centroid_px"]["x"]
        cy = weed["centroid_px"]["y"]

        # Primary target = weed closest to center
        is_primary = (i == 0)
        color = (0, 0, 255) if is_primary else (0, 80, 200)
        thick = 2 if is_primary else 1

        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thick)
        cv2.circle(out, (cx, cy), 8, color, -1)

        if is_primary:
            # Crosshair on primary target
            cv2.line(out, (cx-22, cy), (cx+22, cy), color, 2)
            cv2.line(out, (cx, cy-22), (cx, cy+22), color, 2)

        coord_label = f"WEED ({cx},{cy}) Z={Z_FIXED_MM}mm"
        cv2.putText(out, coord_label,
                    (b["x1"], max(b["y1"]-7, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # ── Summary panel (top bar) ─────────────────────────────────────────────
    panel_h = 45
    cv2.rectangle(out, (0, 0), (w, panel_h), (20, 20, 20), -1)
    summary = (f"Cauliflower: {len(crops)}  |  "
               f"Weeds: {len(weeds)}  |  "
               f"Z_fixed: {Z_FIXED_MM}mm")
    cv2.putText(out, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    # ── Legend (bottom-left) ────────────────────────────────────────────────
    lx, ly = 10, h - 55
    cv2.rectangle(out, (lx, ly), (lx+18, ly+14), (0, 220, 0), -1)
    cv2.putText(out, "Cauliflower (crop)", (lx+22, ly+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1)
    cv2.rectangle(out, (lx, ly+20), (lx+18, ly+34), (0, 0, 220), -1)
    cv2.putText(out, "Weed (pluck target)", (lx+22, ly+32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 80, 220), 1)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_image(model, img_path, output_dir, show=True):
    """Full pipeline on one image. Saves annotated PNG + JSON."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  ❌ Cannot read: {img_path}")
        return None

    h, w = frame.shape[:2]
    print(f"\n{'═'*65}")
    print(f"  Image : {img_path.name}  ({w}×{h})")

    # ── Step 1: Cauliflower segmentation ────────────────────────────────────
    cauli_mask, crops = detect_cauliflower(model, frame)
    print(f"\n  ✅ Cauliflower detected : {len(crops)}")
    for c in crops:
        print(f"     #{c['id']}  centroid=({c['centroid_px']['x']}, "
              f"{c['centroid_px']['y']})  Z={Z_FIXED_MM}mm  "
              f"conf={c['confidence']}  area={c['mask_area_px']}px²")

    # ── Step 2: ExG vegetation map ───────────────────────────────────────────
    veg_mask, exg_map = excess_green_mask(frame)

    # ── Step 3: Weed localization ────────────────────────────────────────────
    weeds, weed_mask = find_weeds(veg_mask, cauli_mask, h, w)
    print(f"\n  🌿 Weeds detected       : {len(weeds)}")

    # Sort weeds by distance from image centre (closest = primary target)
    img_cx, img_cy = w // 2, h // 2
    weeds.sort(key=lambda weed: (
        (weed["centroid_px"]["x"] - img_cx)**2 +
        (weed["centroid_px"]["y"] - img_cy)**2
    ))

    for weed in weeds:
        print(f"     #{weed['id']}  centroid=({weed['centroid_px']['x']}, "
              f"{weed['centroid_px']['y']})  Z={Z_FIXED_MM}mm  "
              f"area={weed['mask_area_px']}px²")

    # ── Step 4: Render annotated image ───────────────────────────────────────
    annotated = render_annotated(frame, crops, weeds, weed_mask, cauli_mask)

    # ── Save outputs ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    # Annotated image
    out_img = output_dir / f"{stem}_result.jpg"
    cv2.imwrite(str(out_img), annotated)

    # ExG debug image (grayscale heatmap)
    exg_norm = cv2.normalize(exg_map, None, 0, 255, cv2.NORM_MINMAX)
    exg_vis  = cv2.applyColorMap(exg_norm.astype(np.uint8), cv2.COLORMAP_SUMMER)
    out_exg  = output_dir / f"{stem}_exg_map.jpg"
    cv2.imwrite(str(out_exg), exg_vis)

    # JSON results
    def clean(lst):
        """Remove internal _mask field before JSON serialisation."""
        return [{k: v for k, v in d.items() if not k.startswith('_')} for d in lst]

    result = {
        "image":          img_path.name,
        "size_px":        {"w": w, "h": h},
        "Z_fixed_mm":     Z_FIXED_MM,
        "cauliflower":    clean(crops),
        "weeds":          clean(weeds),
        "summary": {
            "n_cauliflower": len(crops),
            "n_weeds":       len(weeds),
            "robot_action":  "SCAN" if not weeds else
                             (f"MOVE_TO ({weeds[0]['centroid_px']['x']},"
                              f"{weeds[0]['centroid_px']['y']},Z={Z_FIXED_MM}mm)")
        }
    }

    out_json = output_dir / f"{stem}_result.json"
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  📁 Saved:")
    print(f"     {out_img}")
    print(f"     {out_exg}  (ExG vegetation heatmap)")
    print(f"     {out_json}")
    print(f"\n  🤖 Robot action : {result['summary']['robot_action']}")
    print(f"{'═'*65}")

    if show:
        try:
            cv2.imshow(f"Result — {img_path.name}", annotated)
            cv2.imshow(f"ExG Map — {img_path.name}", exg_vis)
            print("  (Press any key to continue...)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass  # headless environment

    return result


def main():

    global EXG_THRESHOLD, CONF_THRESH, Z_FIXED_MM

    parser = argparse.ArgumentParser(
        description="Laptop test: Hybrid weed detector (YOLOv8 + ExG)"
    )

    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', default='./results')
    parser.add_argument('--exg', type=float, default=EXG_THRESHOLD)
    parser.add_argument('--conf', type=float, default=CONF_THRESH)
    parser.add_argument('--z', type=int, default=Z_FIXED_MM)
    parser.add_argument('--no-show', action='store_true')

    args = parser.parse_args()

    # Apply CLI overrides
    EXG_THRESHOLD = args.exg
    CONF_THRESH   = args.conf
    Z_FIXED_MM    = args.z

    # ── Load model ─────────────────────────────────────────────────────────
    from ultralytics import YOLO
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"\n{'='*65}")
    print(f"  HYBRID WEED DETECTOR — Laptop Test")
    print(f"{'='*65}")
    print(f"  Model     : {model_path}")
    print(f"  ExG thresh: {EXG_THRESHOLD}  (tune with --exg)")
    print(f"  CONF      : {CONF_THRESH}")
    print(f"  Z fixed   : {Z_FIXED_MM} mm")
    print(f"{'='*65}")

    model = YOLO(str(model_path))
    print(f"  ✅ Model loaded\n")

    # ── Find images ─────────────────────────────────────────────────────────
    src  = Path(args.source)
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    if src.is_dir():
        images = sorted([p for p in src.iterdir() if p.suffix.lower() in exts])
    else:
        images = [src] if src.suffix.lower() in exts else []

    if not images:
        print(f"❌ No images found at: {args.source}")
        return

    print(f"  Found {len(images)} image(s) to process\n")

    output_dir = Path(args.output)
    all_results = []

    for img_path in images:
        result = process_image(model, img_path, output_dir,
                               show=not args.no_show)
        if result:
            all_results.append(result)

    # ── Summary across all images ───────────────────────────────────────────
    if len(all_results) > 1:
        total_cauli = sum(r["summary"]["n_cauliflower"] for r in all_results)
        total_weeds = sum(r["summary"]["n_weeds"] for r in all_results)
        print(f"\n{'═'*65}")
        print(f"  BATCH SUMMARY")
        print(f"  Images processed : {len(all_results)}")
        print(f"  Total cauliflower: {total_cauli}")
        print(f"  Total weeds      : {total_weeds}")
        print(f"  Results folder   : {output_dir.resolve()}")
        print(f"{'═'*65}")

    print(f"\n  Done! Results saved to: {output_dir.resolve()}")


if __name__ == '__main__':
    main()
