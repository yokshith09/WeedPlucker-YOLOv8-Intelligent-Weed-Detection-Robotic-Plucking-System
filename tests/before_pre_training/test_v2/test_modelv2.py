"""
WEED ROBOT INFERENCE  — Fixed Version
======================================
Fixes vs previous infer.py:
  ✅ Separate confidence thresholds for crop vs weed
  ✅ Size filter uses MASK area (not bbox) — close-up plants no longer removed
  ✅ Overlap filter uses strict containment only — weeds between rows kept
  ✅ --debug flag prints exactly why each detection was filtered
  ✅ Sky filter made optional (--sky 0 to disable for close-up shots)

Usage:
    python infer_fixed.py --model best.pt --source img_259.jpg
    python infer_fixed.py --model best.pt --source ./images/ --conf_crop 0.40 --conf_weed 0.20
    python infer_fixed.py --model best.pt --source img_259.jpg --debug
    python infer_fixed.py --model best.pt --source ./images/ --sky 0    # disable sky filter
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path

# ─── DEFAULT SETTINGS (all overridable via CLI) ────────────────────────────────

# ✅ FIX 1: Separate thresholds per class
CONF_CROP       = 0.40   # crops are large → model is confident → can afford higher
CONF_WEED       = 0.20   # weeds are small → keep lower to catch more

IOU_THRESH      = 0.45
IMGSZ           = 1280   # keep high for 4K images; use 640 for already-cropped patches

Z_FIXED_MM      = 500

# ✅ FIX 2: Size filter now uses MASK pixel fraction, not bbox
#   Old: bbox_area/total_px > 0.12 → killed close-up plants
#   New: mask_px/total_px  > 0.40 → only truly full-frame background blobs removed
MAX_MASK_AREA_FRAC  = 0.40   # >40% of image pixels = background blob, discard
MIN_MASK_AREA_PX    = 150    # ignore tiny noise detections (< 150px)

# Sky zone (top N% of image ignored) — set to 0.0 to fully disable
SKY_FRAC        = 0.30

# ✅ FIX 3: Overlap filter now only removes weeds that are STRICTLY INSIDE crops
#   Old: crop mask dilated by 15px → weed anywhere near crop row = discarded
#   New: no dilation, strict overlap >60% → only genuine leaf-edge FP removed
CROP_OVERLAP_THRESH = 0.60   # weed must be >60% inside crop mask to be discarded
                              # (was 0.25 with 15px buffer — way too aggressive)

DEBUG = False   # set True via --debug to print filter reasons

# ─── Colours ──────────────────────────────────────────────────────────────────
CROP_COLOR  = (30, 220, 30)    # green
WEED_COLOR  = (30,  30, 220)   # red (BGR)
TEXT_COLOR  = (255, 255, 255)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, frame, img_h, img_w):
    """
    Run YOLOv8-seg at the LOWER of the two conf thresholds.
    Per-class filtering happens in apply_filters().
    """
    min_conf = min(CONF_CROP, CONF_WEED)

    results = model(
        frame,
        conf=min_conf,      # pass minimum — we filter per class below
        iou=IOU_THRESH,
        imgsz=IMGSZ,
        verbose=False,
        retina_masks=True,  # full-res masks — critical for large images
    )[0]

    dets = []
    if results.masks is None or results.boxes is None:
        return dets

    masks_data = results.masks.data.cpu().numpy()
    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
    classes    = results.boxes.cls.cpu().numpy().astype(int)
    confs      = results.boxes.conf.cpu().numpy()

    for mask_raw, box, cls_id, conf in zip(masks_data, boxes_xyxy, classes, confs):
        mask = cv2.resize(mask_raw, (img_w, img_h))
        mask = (mask > 0.5).astype(np.uint8) * 255

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        cx = int(xs.mean())
        cy = int(ys.mean())
        x1, y1, x2, y2 = map(int, box)

        dets.append({
            "class_id":   int(cls_id),
            "class_name": "cauliflower" if cls_id == 0 else "weed",
            "conf":       round(float(conf), 3),
            "centroid":   {"x": cx, "y": cy},
            "Z_mm":       Z_FIXED_MM,
            "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "mask_area":  int(len(xs)),
            "_mask":      mask,
        })

    return dets


# ══════════════════════════════════════════════════════════════════════════════
# FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def _dbg(msg):
    if DEBUG:
        print(f"    [FILTER] {msg}")


def apply_filters(dets, img_h, img_w):
    """
    Returns (crops, weeds) after three filters.

    Filter A — per-class confidence threshold (NEW)
    Filter B — mask size: too small (noise) or too large (background)
    Filter C — sky zone: centroid above SKY_FRAC
    Filter D — weed strictly inside crop mask (strict overlap only)
    """
    total_px = img_h * img_w
    sky_y    = int(img_h * SKY_FRAC)

    def base_filters(d):
        """Filters A, B, C — applied to both crops and weeds."""
        cls   = d["class_name"]
        conf  = d["conf"]
        cy    = d["centroid"]["y"]
        area  = d["mask_area"]

        # Filter A: per-class confidence
        threshold = CONF_CROP if d["class_id"] == 0 else CONF_WEED
        if conf < threshold:
            _dbg(f"REMOVE {cls} conf={conf} < threshold={threshold}")
            return False

        # Filter B: mask too small (noise pixel)
        if area < MIN_MASK_AREA_PX:
            _dbg(f"REMOVE {cls} mask_area={area}px < {MIN_MASK_AREA_PX}px (noise)")
            return False

        # Filter B: mask too large (background tree / entire frame)
        mask_frac = area / total_px
        if mask_frac > MAX_MASK_AREA_FRAC:
            _dbg(f"REMOVE {cls} mask={mask_frac*100:.1f}% of image > {MAX_MASK_AREA_FRAC*100:.0f}% (background)")
            return False

        # Filter C: sky zone
        if SKY_FRAC > 0 and cy < sky_y:
            _dbg(f"REMOVE {cls} cy={cy} < sky_y={sky_y} (sky zone)")
            return False

        return True

    crops = [d for d in dets if d["class_id"] == 0 and base_filters(d)]
    weeds_raw = [d for d in dets if d["class_id"] == 1 and base_filters(d)]

    # Filter D: weed strictly inside crop mask — NO dilation buffer
    # Only remove if >CROP_OVERLAP_THRESH of the weed's pixels are inside a crop mask
    if crops and weeds_raw:
        crop_combined = np.zeros((img_h, img_w), dtype=np.uint8)
        for c in crops:
            crop_combined = cv2.bitwise_or(crop_combined, c["_mask"])

        weeds = []
        for w in weeds_raw:
            m           = w["_mask"]
            weed_px     = max(int(m.sum() // 255), 1)
            overlap_px  = int(cv2.bitwise_and(m, crop_combined).sum() // 255)
            overlap_frac = overlap_px / weed_px

            if overlap_frac > CROP_OVERLAP_THRESH:
                _dbg(f"REMOVE weed x={w['centroid']['x']} overlap={overlap_frac*100:.0f}% inside crop mask (leaf-edge FP)")
            else:
                _dbg(f"KEEP   weed x={w['centroid']['x']} overlap={overlap_frac*100:.0f}% (not inside crop)")
                weeds.append(w)
    else:
        weeds = weeds_raw

    return crops, weeds


# ══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def put_label(img, x, y, line1, line2=None, bg_color=(20, 20, 20)):
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.50
    thick  = 1
    pad    = 5

    (w1, h1), _ = cv2.getTextSize(line1, font, scale, thick)
    bw = w1 + pad * 2
    bh = h1 + pad * 2

    if line2:
        (w2, h2), _ = cv2.getTextSize(line2, font, scale, thick)
        bw = max(bw, w2 + pad * 2)
        bh += h2 + 4

    ih, iw = img.shape[:2]
    bx1 = max(0, min(x, iw - bw - 1))
    by1 = max(0, min(y - bh - 6, ih - bh - 1))
    bx2 = bx1 + bw
    by2 = by1 + bh

    cv2.rectangle(img, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.putText(img, line1, (bx1 + pad, by1 + pad + h1),
                font, scale, TEXT_COLOR, thick, cv2.LINE_AA)
    if line2:
        (_, h2), _ = cv2.getTextSize(line2, font, scale, thick)
        cv2.putText(img, line2, (bx1 + pad, by2 - pad),
                    font, scale, TEXT_COLOR, thick, cv2.LINE_AA)


def render_output(frame, crops, weeds, img_h, img_w):
    out     = frame.copy()
    overlay = out.copy()

    for c in crops:
        overlay[c["_mask"] > 0] = (
            overlay[c["_mask"] > 0] * 0.35 + np.array([30, 220, 30]) * 0.65
        ).astype(np.uint8)

    for w in weeds:
        overlay[w["_mask"] > 0] = (
            overlay[w["_mask"] > 0] * 0.35 + np.array([30, 30, 220]) * 0.65
        ).astype(np.uint8)

    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    # Contours
    for c in crops:
        cnts, _ = cv2.findContours(c["_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, CROP_COLOR, 3)

    for w in weeds:
        cnts, _ = cv2.findContours(w["_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, WEED_COLOR, 3)

    # Crop labels
    for idx, c in enumerate(crops):
        cx, cy = c["centroid"]["x"], c["centroid"]["y"]
        cv2.circle(out, (cx, cy), 8, CROP_COLOR, -1)
        cv2.circle(out, (cx, cy), 8, (255, 255, 255), 2)
        arm = 18
        cv2.line(out, (cx - arm, cy), (cx + arm, cy), CROP_COLOR, 2)
        cv2.line(out, (cx, cy - arm), (cx, cy + arm), CROP_COLOR, 2)
        put_label(out, cx, cy,
                  f"CROP #{idx}  conf={c['conf']}",
                  f"x={cx}  y={cy}  Z={Z_FIXED_MM}mm",
                  bg_color=(0, 80, 0))

    # Weed labels
    for idx, w in enumerate(weeds):
        cx, cy = w["centroid"]["x"], w["centroid"]["y"]
        is_primary = (idx == 0)
        color  = (0, 0, 255) if is_primary else (60, 60, 200)
        dot_r  = 12 if is_primary else 8
        arm    = 28 if is_primary else 16

        cv2.circle(out, (cx, cy), dot_r, color, -1)
        cv2.circle(out, (cx, cy), dot_r, (255, 255, 255), 2)
        cv2.line(out, (cx - arm, cy), (cx + arm, cy), color, 2)
        cv2.line(out, (cx, cy - arm), (cx, cy + arm), color, 2)

        tag = "TARGET" if is_primary else f"WEED #{idx}"
        put_label(out, cx, cy,
                  f"{tag}  conf={w['conf']}",
                  f"x={cx}  y={cy}  Z={Z_FIXED_MM}mm",
                  bg_color=(130, 0, 0))

    # Banner
    banner_h = 52
    cv2.rectangle(out, (0, 0), (img_w, banner_h), (15, 15, 15), -1)
    if weeds:
        tw     = weeds[0]
        action = f"MOVE_TO  x={tw['centroid']['x']}  y={tw['centroid']['y']}  Z={Z_FIXED_MM}mm"
        col    = (0, 100, 255)
    else:
        action = "SCAN — no weed detected"
        col    = (0, 200, 200)

    cv2.putText(out,
                f"Crops:{len(crops)}  Weeds:{len(weeds)}  |  {action}",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)

    # Legend
    lx, ly = 10, img_h - 70
    cv2.rectangle(out, (lx, ly),      (lx+14, ly+12), CROP_COLOR, -1)
    cv2.putText(out, f"Cauliflower (class 0, conf>{CONF_CROP})",
                (lx+20, ly+11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CROP_COLOR, 1)
    cv2.rectangle(out, (lx, ly+20), (lx+14, ly+32), WEED_COLOR, -1)
    cv2.putText(out, f"Weed (class 1, conf>{CONF_WEED})  \u2190 robot target",
                (lx+20, ly+31), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WEED_COLOR, 1)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# PER-IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_image(model, img_path, output_dir):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  ❌ Cannot read: {img_path}")
        return None

    img_h, img_w = frame.shape[:2]
    print(f"\n{'═'*70}")
    print(f"  {img_path.name}   ({img_w} × {img_h})")
    print(f"  imgsz={IMGSZ}  conf_crop={CONF_CROP}  conf_weed={CONF_WEED}  sky_cut={int(img_h*SKY_FRAC)}px")
    print(f"  size_filter: mask>{MAX_MASK_AREA_FRAC*100:.0f}% discarded | mask<{MIN_MASK_AREA_PX}px discarded")
    print(f"  overlap_filter: weed>{CROP_OVERLAP_THRESH*100:.0f}% inside crop mask discarded")

    raw      = run_inference(model, frame, img_h, img_w)
    raw_crop = [d for d in raw if d["class_id"] == 0]
    raw_weed = [d for d in raw if d["class_id"] == 1]

    print(f"\n  Raw model output (conf>={min(CONF_CROP,CONF_WEED)}):")
    print(f"    crops={len(raw_crop)}  weeds={len(raw_weed)}")

    if DEBUG and raw_crop:
        print(f"  Raw crops:")
        for d in raw_crop:
            frac = d["mask_area"] / (img_h * img_w)
            print(f"    conf={d['conf']}  mask={frac*100:.1f}%  cx={d['centroid']['x']}  cy={d['centroid']['y']}")
    if DEBUG and raw_weed:
        print(f"  Raw weeds:")
        for d in raw_weed:
            frac = d["mask_area"] / (img_h * img_w)
            print(f"    conf={d['conf']}  mask={frac*100:.1f}%  cx={d['centroid']['x']}  cy={d['centroid']['y']}")

    crops, weeds = apply_filters(raw, img_h, img_w)
    rc = len(raw_crop) - len(crops)
    rw = len(raw_weed) - len(weeds)
    print(f"\n  After filtering:")
    print(f"    crops={len(crops)}  weeds={len(weeds)}"
          f"  (removed {rc} crop FP, {rw} weed FP)")

    # Sort weeds: closest to image centre = primary target
    img_cx, img_cy = img_w // 2, img_h // 2
    weeds = sorted(weeds, key=lambda d: (
        (d["centroid"]["x"] - img_cx)**2 + (d["centroid"]["y"] - img_cy)**2
    ))

    print(f"\n  CAULIFLOWER ({len(crops)}):")
    for c in crops:
        print(f"    x={c['centroid']['x']:5d}  y={c['centroid']['y']:5d}"
              f"  Z={Z_FIXED_MM}mm  conf={c['conf']}  area={c['mask_area']}px")

    print(f"\n  WEEDS ({len(weeds)}):")
    if weeds:
        for i, w in enumerate(weeds):
            tag = "  ← PRIMARY TARGET" if i == 0 else ""
            print(f"    x={w['centroid']['x']:5d}  y={w['centroid']['y']:5d}"
                  f"  Z={Z_FIXED_MM}mm  conf={w['conf']}  area={w['mask_area']}px{tag}")
    else:
        print("    (none detected)")

    # Robot command
    if weeds:
        tw  = weeds[0]
        dx  = tw["centroid"]["x"] - img_cx
        dy  = tw["centroid"]["y"] - img_cy
        cmd = {
            "action": "PLUCK" if abs(dx) < 30 and abs(dy) < 30 else "MOVE_TO_WEED",
            "target": {"x": tw["centroid"]["x"], "y": tw["centroid"]["y"], "Z_mm": Z_FIXED_MM},
            "offset_x_px": dx,
            "offset_y_px": dy,
        }
    else:
        cmd = {"action": "SCAN", "target": None}

    print(f"\n  Robot: {cmd['action']}", end="")
    if cmd["target"]:
        t = cmd["target"]
        print(f"  →  x={t['x']}  y={t['y']}  Z={Z_FIXED_MM}mm"
              f"  offset=({cmd['offset_x_px']:+d}, {cmd['offset_y_px']:+d})")
    else:
        print()

    annotated = render_output(frame, crops, weeds, img_h, img_w)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    out_img = output_dir / f"{stem}_result.jpg"
    cv2.imwrite(str(out_img), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    def clean(lst):
        return [{k: v for k, v in d.items() if not k.startswith("_")} for d in lst]

    result = {
        "image":      img_path.name,
        "size_px":    {"w": img_w, "h": img_h},
        "Z_fixed_mm": Z_FIXED_MM,
        "settings": {
            "conf_crop":  CONF_CROP,
            "conf_weed":  CONF_WEED,
            "imgsz":      IMGSZ,
            "sky_frac":   SKY_FRAC,
            "max_mask_frac": MAX_MASK_AREA_FRAC,
            "overlap_thresh": CROP_OVERLAP_THRESH,
        },
        "cauliflower": clean(crops),
        "weeds":       clean(weeds),
        "robot":       cmd,
        "stats": {
            "n_crop": len(crops), "n_weed": len(weeds),
            "raw_crop": len(raw_crop), "raw_weed": len(raw_weed),
        },
    }

    out_json = output_dir / f"{stem}_result.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved: {out_img.name}  +  {out_json.name}")
    print(f"{'═'*70}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global CONF_CROP, CONF_WEED, IOU_THRESH, IMGSZ, Z_FIXED_MM
    global SKY_FRAC, MAX_MASK_AREA_FRAC, MIN_MASK_AREA_PX
    global CROP_OVERLAP_THRESH, DEBUG

    parser = argparse.ArgumentParser(
        description="YOLOv8 weed robot inference — fixed confidence thresholds + filters"
    )
    parser.add_argument("--model",       required=True,  help="best.pt or best.onnx")
    parser.add_argument("--source",      required=True,  help="Image file or folder")
    parser.add_argument("--output",      default="./results", help="Output folder")
    parser.add_argument("--conf_crop",   type=float, default=CONF_CROP,
                        help=f"Crop confidence threshold (default {CONF_CROP})")
    parser.add_argument("--conf_weed",   type=float, default=CONF_WEED,
                        help=f"Weed confidence threshold (default {CONF_WEED})")
    parser.add_argument("--imgsz",       type=int,   default=IMGSZ,
                        help=f"Inference size (default {IMGSZ})")
    parser.add_argument("--z",           type=int,   default=Z_FIXED_MM,
                        help=f"Camera height mm (default {Z_FIXED_MM})")
    parser.add_argument("--sky",         type=float, default=SKY_FRAC,
                        help=f"Sky exclusion top fraction (default {SKY_FRAC}, 0=disable)")
    parser.add_argument("--max_mask",    type=float, default=MAX_MASK_AREA_FRAC,
                        help=f"Max mask fraction before discarding as background (default {MAX_MASK_AREA_FRAC})")
    parser.add_argument("--overlap",     type=float, default=CROP_OVERLAP_THRESH,
                        help=f"Weed-in-crop overlap threshold (default {CROP_OVERLAP_THRESH})")
    parser.add_argument("--no_show",     action="store_true", help="Skip display window")
    parser.add_argument("--debug",       action="store_true", help="Print filter decisions")
    args = parser.parse_args()

    CONF_CROP           = args.conf_crop
    CONF_WEED           = args.conf_weed
    IMGSZ               = args.imgsz
    Z_FIXED_MM          = args.z
    SKY_FRAC            = args.sky
    MAX_MASK_AREA_FRAC  = args.max_mask
    CROP_OVERLAP_THRESH = args.overlap
    DEBUG               = args.debug

    from ultralytics import YOLO
    mp = Path(args.model)
    if not mp.exists():
        print(f"❌ Model not found: {mp}"); return

    print(f"\n{'='*70}")
    print(f"  WEED ROBOT INFERENCE  (fixed)")
    print(f"{'='*70}")
    print(f"  Model       : {mp}")
    print(f"  conf_crop   : {CONF_CROP}   conf_weed : {CONF_WEED}")
    print(f"  imgsz       : {IMGSZ}")
    print(f"  Sky cutoff  : top {SKY_FRAC*100:.0f}% excluded  (0 = disabled)")
    print(f"  Max mask    : {MAX_MASK_AREA_FRAC*100:.0f}% of image  (was 12% bbox — caused close-up bug)")
    print(f"  Overlap     : {CROP_OVERLAP_THRESH*100:.0f}% weed inside crop = discard  (was 25%+buffer — caused row bug)")
    print(f"{'='*70}")

    model = YOLO(str(mp))
    print(f"  ✅ Model loaded")

    src  = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = ([src] if src.is_file()
            else sorted(p for p in src.iterdir() if p.suffix.lower() in exts))

    if not imgs:
        print(f"❌ No images at: {args.source}"); return

    print(f"  {len(imgs)} image(s) to process\n")
    out_dir = Path(args.output)
    results = []

    for img_path in imgs:
        r = process_image(model, img_path, out_dir)
        if r:
            results.append(r)
            if not args.no_show and len(imgs) == 1:
                res_img = cv2.imread(str(out_dir / f"{img_path.stem}_result.jpg"))
                if res_img is not None:
                    dh, dw = res_img.shape[:2]
                    if dh > 1200:
                        s = 1200 / dh
                        res_img = cv2.resize(res_img, (int(dw*s), int(dh*s)))
                    cv2.imshow("Result — press any key", res_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    if len(results) > 1:
        tc = sum(r["stats"]["n_crop"] for r in results)
        tw = sum(r["stats"]["n_weed"] for r in results)
        rw = sum(r["stats"]["raw_weed"] for r in results)
        print(f"\n{'='*70}")
        print(f"  BATCH DONE")
        print(f"  Images   : {len(results)}")
        print(f"  Crops    : {tc}  total")
        print(f"  Weeds    : {tw}  (model raw: {rw})")
        print(f"  Output   : {out_dir.resolve()}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()