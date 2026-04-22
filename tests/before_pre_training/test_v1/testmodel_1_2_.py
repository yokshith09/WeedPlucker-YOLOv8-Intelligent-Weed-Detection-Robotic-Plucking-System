"""
WEED ROBOT INFERENCE — Clean, Final Script
==========================================
Uses ONLY the trained YOLOv8-seg model output.
  class 0 = cauliflower (crop)
  class 1 = weed

NO ExG. NO vegetation index. The model already knows both classes.

What this script gives you:
  ✅ Coloured segmentation mask per plant (drawn ON the image)
  ✅ Centroid (cx, cy) marked with a crosshair ON each plant
  ✅ Coordinates printed as text ON the plant
  ✅ Z coordinate is fixed (you set it below)
  ✅ One JSON file with clean robot-ready coordinates
  ✅ Only one output coordinate per plant
  ✅ False positive removal: overlap filter + tiny blob filter

WHY WEEDS NOT DETECTED ON NEW IMAGES — read this:
  Your model was trained on close-up top-down or 45° field shots.
  New eye-level images with wide backgrounds are a different distribution.
  Fix: use conf=0.15 (very sensitive). The model WILL detect if the
  plant is visible. If still nothing: your new image angles need more
  training data collected from that same viewpoint.

USAGE:
    # Single image — shows window + saves result
    python detect.py --model best.pt --source image.jpg

    # Folder of images
    python detect.py --model best.pt --source ./test_images/

    # ONNX model (for Pi testing on laptop)
    python detect.py --model best.onnx --source ./test_images/

    # Set your actual camera height
    python detect.py --model best.pt --source image.jpg --z 450

    # If weed STILL not detected → lower conf more
    python detect.py --model best.pt --source image.jpg --conf 0.12

    # If too many false positives → raise conf
    python detect.py --model best.pt --source image.jpg --conf 0.35

OUTPUT FILES (saved to ./results/):
    image_result.jpg   — annotated image with masks + coordinates
    image_result.json  — robot coordinates JSON
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path

# ─── SET THESE ─────────────────────────────────────────────────────────────────
Z_MM         = 500     # ← your camera height above ground in mm
CONF         = 0.15    # ← confidence threshold (0.15 = sensitive, 0.40 = strict)
IMGSZ        = 640     # ← MUST match your training imgsz (you trained at 640)
IOU          = 0.45    # NMS IoU threshold

# False positive filters (these are safe and correct)
MIN_AREA_PX  = 400     # ignore masks smaller than 400px² (drip tubes, stones)
OVERLAP_THR  = 0.40    # discard weed if >40% of its mask overlaps a crop mask

# Colours
COL_CROP     = (0, 220, 0)     # green  — cauliflower
COL_WEED_PRI = (0, 0, 255)     # red    — primary weed target
COL_WEED     = (0, 100, 220)   # orange — other weeds
COL_TEXT     = (255, 255, 255) # white  — text
# ───────────────────────────────────────────────────────────────────────────────


def run(model, frame):
    """YOLOv8 inference → list of detections."""
    h, w = frame.shape[:2]
    res   = model(frame, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)[0]

    dets = []
    if res.masks is None or res.boxes is None:
        return dets

    masks  = res.masks.data.cpu().numpy()
    boxes  = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs  = res.boxes.conf.cpu().numpy()

    for mask_raw, box, cls_id, conf in zip(masks, boxes, cls_ids, confs):

        # Full-resolution binary mask
        m    = cv2.resize(mask_raw, (w, h))
        m    = (m > 0.5).astype(np.uint8) * 255

        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue

        # Centroid from mask pixels (accurate centre of plant, not bbox centre)
        cx   = int(xs.mean())
        cy   = int(ys.mean())
        area = int(len(xs))

        x1, y1, x2, y2 = map(int, box)

        dets.append({
            "cls":      int(cls_id),          # 0=crop, 1=weed
            "name":     "crop" if cls_id == 0 else "weed",
            "conf":     round(float(conf), 3),
            "cx":       cx,                   # ← centroid x pixel
            "cy":       cy,                   # ← centroid y pixel
            "z":        Z_MM,                 # ← fixed Z (camera height)
            "cx_norm":  round(cx / w, 4),
            "cy_norm":  round(cy / h, 4),
            "bbox":     [x1, y1, x2, y2],
            "area":     area,
            "_m":       m,                    # mask (stripped before JSON)
        })

    return dets


def filter_fp(dets, img_h, img_w):
    """
    Two safe filters — does NOT use sky zone (that was removing real weeds).

    Filter 1: Remove tiny masks (noise, stones, drip tube ends)
    Filter 2: Remove weed detections that substantially overlap crop masks
              (model occasionally fires class 1 on cauliflower leaf edges)
    """
    crops = [d for d in dets if d["cls"] == 0 and d["area"] >= MIN_AREA_PX]
    weeds = [d for d in dets if d["cls"] == 1 and d["area"] >= MIN_AREA_PX]

    if not crops:
        return crops, weeds  # no crops → keep all weeds as-is

    # Build combined dilated crop mask
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    for c in crops:
        combined = cv2.bitwise_or(combined, c["_m"])
    buf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    combined = cv2.dilate(combined, buf, iterations=1)

    clean_weeds = []
    for w in weeds:
        m       = w["_m"]
        overlap = int(cv2.bitwise_and(m, combined).sum() // 255)
        frac    = overlap / max(int(m.sum() // 255), 1)
        if frac < OVERLAP_THR:
            clean_weeds.append(w)

    return crops, clean_weeds


def draw(frame, crops, weeds):
    """
    Draw segmentation masks, centroids, crosshairs, and coordinate labels
    directly ON the image.
    """
    out     = frame.copy()
    overlay = out.copy()
    h, w    = out.shape[:2]
    img_cx  = w // 2
    img_cy  = h // 2

    # Sort weeds: closest to image centre = primary target
    weeds = sorted(weeds, key=lambda d: (
        (d["cx"] - img_cx) ** 2 + (d["cy"] - img_cy) ** 2
    ))

    # ── Segmentation fills ────────────────────────────────────────────────────
    for c in crops:
        overlay[c["_m"] > 0] = np.clip(
            overlay[c["_m"] > 0] * 0.3 + np.array([20, 200, 20]) * 0.7, 0, 255
        ).astype(np.uint8)

    for i, weed in enumerate(weeds):
        col   = [20, 20, 200] if i == 0 else [30, 100, 200]
        alpha = 0.70 if i == 0 else 0.55
        overlay[weed["_m"] > 0] = np.clip(
            overlay[weed["_m"] > 0] * (1 - alpha) + np.array(col) * alpha,
            0, 255
        ).astype(np.uint8)

    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # ── Draw segmentation CONTOURS (outline per plant) ────────────────────────
    for c in crops:
        cnts, _ = cv2.findContours(c["_m"], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COL_CROP, 2)

    for i, weed in enumerate(weeds):
        col  = COL_WEED_PRI if i == 0 else COL_WEED
        cnts, _ = cv2.findContours(weed["_m"], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 2)

    # ── Crop centroid + coordinates ON the plant ──────────────────────────────
    for c in crops:
        cx, cy = c["cx"], c["cy"]

        # Filled circle at centroid
        cv2.circle(out, (cx, cy), 10, COL_CROP, -1)
        cv2.circle(out, (cx, cy), 10, (0, 0, 0), 1)  # thin black ring

        # Coordinate text directly on the plant
        label = f"CROP  x={cx}  y={cy}  z={Z_MM}mm"
        conf_label = f"conf={c['conf']:.2f}"

        # Background pill for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (cx - 4, cy + 12), (cx + tw + 8, cy + 14 + th + 4),
                      (0, 0, 0), -1)
        cv2.putText(out, label, (cx, cy + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_CROP, 1)
        cv2.putText(out, conf_label, (cx, cy + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)

    # ── Weed centroid + coordinates ON the plant ──────────────────────────────
    for i, weed in enumerate(weeds):
        cx, cy = weed["cx"], weed["cy"]
        col    = COL_WEED_PRI if i == 0 else COL_WEED
        is_pri = (i == 0)

        # Centroid dot
        cv2.circle(out, (cx, cy), 12 if is_pri else 8, col, -1)
        cv2.circle(out, (cx, cy), 12 if is_pri else 8, (0, 0, 0), 1)

        # Crosshair on primary target
        if is_pri:
            arm = 30
            cv2.line(out, (cx - arm, cy), (cx + arm, cy), col, 2)
            cv2.line(out, (cx, cy - arm), (cx, cy + arm), col, 2)
            # Offset arrows to image centre
            ox = cx - img_cx
            oy = cy - img_cy
            offset_txt = f"offset x={ox:+d}  y={oy:+d}px"
            cv2.putText(out, offset_txt, (cx, cy - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1)

        # Coordinate label background + text
        tag   = "TARGET" if is_pri else f"WEED #{i}"
        label = f"{tag}  x={cx}  y={cy}  z={Z_MM}mm"
        conf_label = f"conf={weed['conf']:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (cx - 4, cy + 12), (cx + tw + 8, cy + 14 + th + 4),
                      (0, 0, 0), -1)
        cv2.putText(out, label, (cx, cy + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
        cv2.putText(out, conf_label, (cx, cy + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 160, 160) if is_pri else (180, 180, 255), 1)

    # ── Summary banner ────────────────────────────────────────────────────────
    if weeds:
        tw = weeds[0]
        action = f"MOVE_TO  x={tw['cx']}  y={tw['cy']}  z={Z_MM}mm"
        col_banner = (0, 0, 200)
    else:
        action = "SCAN — no weeds detected"
        col_banner = (100, 100, 0)

    cv2.rectangle(out, (0, 0), (w, 50), (10, 10, 10), -1)
    cv2.putText(out,
                f"crops={len(crops)}  weeds={len(weeds)}  |  {action}",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    # ── Legend (bottom) ───────────────────────────────────────────────────────
    lx, ly = 10, h - 66
    cv2.rectangle(out, (lx, ly), (lx + 18, ly + 14), COL_CROP, -1)
    cv2.putText(out, "cauliflower (protect)",
                (lx + 22, ly + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_CROP, 1)
    cv2.rectangle(out, (lx, ly + 20), (lx + 18, ly + 34), COL_WEED_PRI, -1)
    cv2.putText(out, "weed — robot target",
                (lx + 22, ly + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_WEED_PRI, 1)
    cv2.rectangle(out, (lx, ly + 40), (lx + 18, ly + 54), COL_WEED, -1)
    cv2.putText(out, "weed — secondary",
                (lx + 22, ly + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_WEED, 1)

    return out, weeds


def process_one(model, img_path, out_dir, show):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  ❌ Cannot read: {img_path}")
        return None

    h, w = frame.shape[:2]

    print(f"\n{'═' * 65}")
    print(f"  {img_path.name}  ({w}×{h})")
    print(f"  conf={CONF}  imgsz={IMGSZ}  z={Z_MM}mm")

    # ── Detect ────────────────────────────────────────────────────────────────
    dets = run(model, frame)

    raw_crops = sum(1 for d in dets if d["cls"] == 0)
    raw_weeds = sum(1 for d in dets if d["cls"] == 1)
    print(f"  Model output: {raw_crops} crops  {raw_weeds} weeds (before filter)")

    # ── Filter ────────────────────────────────────────────────────────────────
    crops, weeds = filter_fp(dets, h, w)
    print(f"  After filter: {len(crops)} crops  {len(weeds)} weeds")

    if not crops and not weeds:
        print(f"\n  ⚠️  Nothing detected at conf={CONF}")
        print(f"  → Try: python detect.py --model best.pt --source {img_path.name} --conf 0.10")
        print(f"  → Or:  python detect.py --model best.pt --source {img_path.name} --imgsz 1280")

    # ── Draw ─────────────────────────────────────────────────────────────────
    annotated, weeds_sorted = draw(frame, crops, weeds)

    # ── Print per-plant results ───────────────────────────────────────────────
    img_cx = w // 2
    img_cy = h // 2

    print(f"\n  {'─'*55}")
    print(f"  CROPS  (class 0 — protect these)")
    for i, c in enumerate(crops):
        print(f"    #{i}  centroid=({c['cx']}, {c['cy']})  z={Z_MM}mm"
              f"  conf={c['conf']}  area={c['area']}px²")

    print(f"\n  WEEDS  (class 1 — robot targets, sorted by distance to centre)")
    for i, weed in enumerate(weeds_sorted):
        ox = weed["cx"] - img_cx
        oy = weed["cy"] - img_cy
        tag = "← PRIMARY TARGET" if i == 0 else ""
        print(f"    #{i}  centroid=({weed['cx']}, {weed['cy']})  z={Z_MM}mm"
              f"  conf={weed['conf']}  offset=({ox:+d},{oy:+d})  {tag}")

    # ── Robot command ─────────────────────────────────────────────────────────
    if weeds_sorted:
        t = weeds_sorted[0]
        robot = {
            "action":      "PLUCK" if abs(t["cx"] - img_cx) < 30 and
                                      abs(t["cy"] - img_cy) < 30 else "MOVE_TO_WEED",
            "x":           t["cx"],
            "y":           t["cy"],
            "z":           Z_MM,
            "offset_x":    t["cx"] - img_cx,
            "offset_y":    t["cy"] - img_cy,
            "confidence":  t["conf"],
            "n_weeds":     len(weeds_sorted),
            "n_crops":     len(crops),
            "all_weeds": [{"x": w["cx"], "y": w["cy"], "z": Z_MM,
                           "conf": w["conf"]} for w in weeds_sorted],
            "all_crops": [{"x": c["cx"], "y": c["cy"], "z": Z_MM,
                           "conf": c["conf"]} for c in crops],
        }
    else:
        robot = {
            "action": "SCAN", "x": None, "y": None, "z": Z_MM,
            "n_weeds": 0, "n_crops": len(crops),
        }

    print(f"\n  ROBOT: {robot['action']}", end="")
    if robot.get("x") is not None:
        print(f"  x={robot['x']}  y={robot['y']}  z={Z_MM}mm"
              f"  offset=({robot['offset_x']:+d},{robot['offset_y']:+d})")
    else:
        print()
    print(f"  {'─'*55}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = img_path.stem
    out_img  = out_dir / f"{stem}_result.jpg"
    out_json = out_dir / f"{stem}_result.json"

    cv2.imwrite(str(out_img), annotated)

    def _clean(lst):
        return [{k: v for k, v in d.items() if k != "_m"} for d in lst]

    result = {
        "image":      img_path.name,
        "size":       {"w": w, "h": h},
        "z_mm":       Z_MM,
        "conf_used":  CONF,
        "crops":      _clean(crops),
        "weeds":      _clean(weeds_sorted),
        "robot":      robot,
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved → {out_img.name}")
    print(f"  Saved → {out_json.name}")

    if show:
        scale = min(1.0, 900 / max(w, h))
        disp  = cv2.resize(annotated, (int(w * scale), int(h * scale)))
        cv2.imshow(f"{img_path.name}", disp)
        print(f"  (press any key to continue)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def main():
    global CONF, IMGSZ, Z_MM, MIN_AREA_PX, OVERLAP_THR

    p = argparse.ArgumentParser(
        description="YOLOv8 weed+crop inference — clean output"
    )
    p.add_argument("--model",   required=True, help="best.pt or best.onnx")
    p.add_argument("--source",  required=True, help="image file or folder")
    p.add_argument("--output",  default="./results")
    p.add_argument("--conf",    type=float, default=CONF,
                   help=f"confidence (default {CONF}). Lower=more detections")
    p.add_argument("--imgsz",   type=int,   default=IMGSZ,
                   help=f"inference size (default {IMGSZ}, match training)")
    p.add_argument("--z",       type=int,   default=Z_MM,
                   help=f"fixed Z in mm (default {Z_MM})")
    p.add_argument("--min_px",  type=int,   default=MIN_AREA_PX,
                   help=f"min mask area px (default {MIN_AREA_PX})")
    p.add_argument("--no_show", action="store_true")
    args = p.parse_args()

    CONF        = args.conf
    IMGSZ       = args.imgsz
    Z_MM        = args.z
    MIN_AREA_PX = args.min_px

    from ultralytics import YOLO
    mp = Path(args.model)
    if not mp.exists():
        print(f"❌ Model not found: {mp}")
        return

    print(f"\n{'='*65}")
    print(f"  WEED ROBOT INFERENCE")
    print(f"{'='*65}")
    print(f"  Model  : {mp.name}")
    print(f"  conf   : {CONF}  |  imgsz : {IMGSZ}  |  Z : {Z_MM}mm")
    print(f"  min_px : {MIN_AREA_PX}  |  overlap_thr : {OVERLAP_THR}")
    print(f"{'='*65}")

    model = YOLO(str(mp))

    src  = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in exts
    )

    if not imgs:
        print(f"❌ No images found: {args.source}")
        return

    out_dir = Path(args.output)
    results = []
    for img_path in imgs:
        r = process_one(model, img_path, out_dir, show=not args.no_show)
        if r:
            results.append(r)

    if len(results) > 1:
        tc = sum(r["robot"]["n_crops"] for r in results)
        tw = sum(r["robot"]["n_weeds"] for r in results)
        print(f"\n{'='*65}")
        print(f"  DONE — {len(results)} images")
        print(f"  Total crops detected : {tc}")
        print(f"  Total weeds detected : {tw}")
        print(f"  Results in           : {out_dir.resolve()}")
        print(f"{'='*65}")
    else:
        print(f"\n  Results in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()