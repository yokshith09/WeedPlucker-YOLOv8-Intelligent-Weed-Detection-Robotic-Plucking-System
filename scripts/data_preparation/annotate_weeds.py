"""
CAULIFLOWER ANNOTATOR v3 — Fixed & Tested on Your Images
=========================================================
Problems fixed from the previous version:
  ✗ Was detecting background trees → ✅ Sky/background zone excluded
  ✗ Was detecting soil             → ✅ Size filter removes huge blobs
  ✗ Was missing actual plants      → ✅ Dual ExG + HSV catches all green tones
  ✗ Plants merging into one blob   → ✅ Watershed splits individual plants

Strategy (same simple logic as annotate_weeds.py):
  Step 1  ExG + HSV mask  → find ALL green in the image
  Step 2  Sky exclusion   → ignore everything in the top SKY_ZONE_FRAC of image
  Step 3  Size filter     → keep blobs sized like individual plants (not huge trees)
  Step 4  Watershed       → split touching plants into separate instances
  Step 5  YOLO labels     → segmentation polygons (or bbox with --bbox flag)

Output per image:
  labels/  →  "0 x1 y1 x2 y2 ... xn yn"  (YOLO-seg format, class 0 = cauliflower)
  images/  →  copy of original image
  preview/ →  annotated image for visual checking

USAGE:
    # 1. Preview first — always tune before writing labels
    python annotate_cauliflower.py --source ./dataset/annotated/images/train --preview_only

    # 2. Tune if needed:
    #    Too many background detections → raise --sky 0.40
    #    Plants missed → lower --sky 0.25 or lower --exg 0.03
    #    Drip tubes detected → raise --min_area 0.001
    python annotate_cauliflower.py --source ./images/train --sky 0.35 --preview_only

    # 3. Run on train split
    python annotate_cauliflower.py \\
        --source ./dataset/annotated/images/train \\
        --output ./dataset/reannotated \\
        --split  train

    # 4. Run on val split
    python annotate_cauliflower.py \\
        --source ./dataset/annotated/images/val \\
        --output ./dataset/reannotated \\
        --split  val

    # 5. Use bounding boxes instead of polygons
    python annotate_cauliflower.py --source ./images/train --bbox
"""

import cv2
import numpy as np
import shutil
import argparse
from pathlib import Path

# ─── CONFIG — tested on your field images ──────────────────────────────────────
SKY_ZONE_FRAC = 0.32    # Ignore top 32% of image (background trees / sky)
                         # Raise to 0.40 if trees still appear
                         # Lower to 0.25 if upper plants are missed

EXG_THRESH    = 0.05    # Excess Green threshold
                         # Lower (0.03) = catches pale/dusty leaves too
                         # Higher (0.08) = only bright green

# HSV range for cauliflower (blue-green to bright green, catches waxy leaves)
HSV_LOW  = np.array([45,  20, 40])    # H:45=yellow-green  S:20  V:40
HSV_HIGH = np.array([135, 255, 225])  # H:135=blue-green   S:max V:225

MIN_AREA_FRAC = 0.0008  # Min blob size = 0.08% of image (removes noise, drip tubes)
MAX_AREA_FRAC = 0.12    # Max blob size = 12% of image per plant instance
                         # Raise to 0.18 if large foreground plants are missed

DIST_THRESH   = 0.30    # Watershed distance threshold
                         # Lower (0.20) = more splits (separate touching plants)
                         # Higher (0.45) = fewer splits (treat cluster as one)

POLY_EPSILON  = 0.006   # Polygon simplification
MORPH_K       = 9       # Morphology kernel size
PROCESS_SCALE = 0.5     # Internal processing scale
PREVIEW_MAX   = 30      # Max preview images to save
CLASS_ID      = 0       # cauliflower = class 0
# ───────────────────────────────────────────────────────────────────────────────

COLORS = [
    (0, 255, 0), (0, 160, 255), (0, 220, 220),
    (200, 0, 255), (255, 200, 0), (255, 80, 0),
    (180, 255, 0), (255, 0, 150),
]


def make_plant_mask(roi_bgr):
    """ExG + HSV combined green mask."""
    f = roi_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(f)
    exg   = 2.0 * g - r - b
    exg_m = (exg > EXG_THRESH).astype(np.uint8) * 255
    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hsv_m = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    plant = cv2.bitwise_or(exg_m, hsv_m)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    plant = cv2.morphologyEx(plant, cv2.MORPH_OPEN,  k)
    plant = cv2.morphologyEx(plant, cv2.MORPH_CLOSE, k)
    return plant


def size_filter(mask, min_frac, max_frac):
    """Keep only blobs in the plant size range."""
    h, w = mask.shape[:2]
    total = h * w
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for cnt in cnts:
        pct = cv2.contourArea(cnt) / total
        if min_frac <= pct <= max_frac:
            cv2.drawContours(out, [cnt], -1, 255, -1)
    return out


def watershed_split(filtered_mask, roi_bgr):
    """Distance transform + watershed to separate individual plants."""
    dist   = cv2.distanceTransform(filtered_mask, cv2.DIST_L2, 5)
    dn     = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    _, sfg = cv2.threshold(dn, DIST_THRESH, 1.0, cv2.THRESH_BINARY)
    sfg    = sfg.astype(np.uint8)
    n_labels, markers = cv2.connectedComponents(sfg)
    if n_labels < 2:
        return [filtered_mask]
    kb  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    sbg = cv2.dilate(filtered_mask, kb, iterations=1)
    unk = cv2.subtract(sbg, sfg)
    mw              = markers + 1
    mw[unk == 255]  = 0
    mw              = cv2.watershed(roi_bgr.copy(), mw)
    instances = []
    for lbl in range(2, n_labels + 1):
        m = (mw == lbl).astype(np.uint8) * 255
        if m.sum() // 255 > 50:
            instances.append(m)
    return instances if instances else [filtered_mask]


def mask_to_seg_label(mask_full, orig_h, orig_w):
    """Binary mask → YOLO-seg label string."""
    cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    eps    = POLY_EPSILON * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 3:
        return None
    pts = []
    for pt in approx:
        x = max(0.0, min(1.0, float(pt[0][0]) / orig_w))
        y = max(0.0, min(1.0, float(pt[0][1]) / orig_h))
        pts.extend([round(x, 5), round(y, 5)])
    return f"{CLASS_ID} {' '.join(str(v) for v in pts)}"


def mask_to_bbox_label(mask_full, orig_h, orig_w):
    """Binary mask → YOLO-bbox label string."""
    ys, xs = np.where(mask_full > 0)
    if len(xs) == 0:
        return None
    cx = float((xs.min() + xs.max()) / 2) / orig_w
    cy = float((ys.min() + ys.max()) / 2) / orig_h
    bw = float(xs.max() - xs.min()) / orig_w
    bh = float(ys.max() - ys.min()) / orig_h
    return (f"{CLASS_ID} {max(0,min(1,cx)):.5f} {max(0,min(1,cy)):.5f} "
            f"{max(0,min(1,bw)):.5f} {max(0,min(1,bh)):.5f}")


def annotate_image(img_path, seg_mode=True):
    """Full pipeline for one image. Returns (labels, preview, n_plants)."""
    orig = cv2.imread(str(img_path))
    if orig is None:
        return [], None, 0

    oh, ow = orig.shape[:2]
    sc  = PROCESS_SCALE
    img = cv2.resize(orig, (int(ow * sc), int(oh * sc)))
    sh, sw = img.shape[:2]

    # Exclude sky/background zone
    sky_y = int(sh * SKY_ZONE_FRAC)
    roi   = img[sky_y:, :]
    rh, rw = roi.shape[:2]
    if rh < 20:
        return [], None, 0

    # Step 1: plant mask
    plant    = make_plant_mask(roi)

    # Step 2: size filter
    filtered = size_filter(plant, MIN_AREA_FRAC, MAX_AREA_FRAC)
    if filtered.sum() == 0:
        return [], None, 0

    # Step 3: watershed instance split
    instances = watershed_split(filtered, roi)

    # Step 4: generate labels
    labels       = []
    contours_vis = []
    min_inst_px  = MIN_AREA_FRAC * rh * rw * 0.5

    for inst_roi in instances:
        if inst_roi.sum() // 255 < min_inst_px:
            continue

        # Map back to full-image coords (add sky offset, upscale)
        inst_sc = np.zeros((sh, sw), dtype=np.uint8)
        inst_sc[sky_y:, :] = inst_roi
        inst_full = cv2.resize(inst_sc, (ow, oh), interpolation=cv2.INTER_NEAREST)
        inst_full = (inst_full > 127).astype(np.uint8) * 255

        label = mask_to_seg_label(inst_full, oh, ow) if seg_mode \
                else mask_to_bbox_label(inst_full, oh, ow)
        if label:
            labels.append(label)
            cnts, _ = cv2.findContours(inst_sc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                contours_vis.append(max(cnts, key=cv2.contourArea))

    # Step 5: preview
    psc     = 0.38
    pw, ph  = int(ow * psc), int(oh * psc)
    preview = cv2.resize(orig, (pw, ph))
    overlay = preview.copy()

    for idx, cnt_sc in enumerate(contours_vis):
        cnt_prev = (cnt_sc * psc).astype(np.int32)
        color    = COLORS[idx % len(COLORS)]
        cv2.drawContours(overlay, [cnt_prev], -1, color, -1)
        cv2.drawContours(preview, [cnt_prev], -1, color, 2)
        M = cv2.moments(cnt_prev)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(preview, (cx, cy), 5, color, -1)
            cv2.putText(preview, f"#{idx+1}",
                        (cx + 7, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    cv2.addWeighted(overlay, 0.28, preview, 0.72, 0, preview)

    # Sky zone indicator line
    sky_y_prev = int(ph * SKY_ZONE_FRAC)
    cv2.line(preview, (0, sky_y_prev), (pw, sky_y_prev), (0, 80, 255), 1)
    cv2.putText(preview, f"sky cutoff ({SKY_ZONE_FRAC*100:.0f}%)",
                (4, sky_y_prev - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 80, 255), 1)

    # Banner
    cv2.rectangle(preview, (0, 0), (pw, 28), (20, 20, 20), -1)
    cv2.putText(preview,
                f"{img_path.name}  plants={len(labels)}  "
                f"{'SEG' if seg_mode else 'BBOX'}",
                (5, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    return labels, preview, len(labels)


def run(source, output, split, preview_only, seg_mode):
    source = Path(source)
    output = Path(output)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp",
                ".JPG", ".JPEG", ".PNG", ".BMP"}
    images   = sorted([p for p in source.iterdir() if p.suffix in img_exts])

    if not images:
        print(f"❌  No images found in {source}")
        return

    out_img = output / "images" / split
    out_lbl = output / "labels" / split
    prv_dir = output / "preview" / split

    prv_dir.mkdir(parents=True, exist_ok=True)
    if not preview_only:
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  CAULIFLOWER ANNOTATOR v3")
    print(f"{'='*65}")
    print(f"  Source     : {source}  ({len(images)} images)")
    print(f"  Split      : {split}")
    print(f"  Mode       : {'Segmentation' if seg_mode else 'Bounding box'}")
    print(f"  Sky cutoff : top {SKY_ZONE_FRAC*100:.0f}% excluded")
    print(f"  ExG thresh : {EXG_THRESH}")
    print(f"  Preview    : {prv_dir}")
    print(f"{'='*65}\n")

    ok = empty = skipped = total_pl = n_prev = 0

    for i, img_path in enumerate(images):
        labels, preview, n_plants = annotate_image(img_path, seg_mode)

        if preview is None:
            skipped += 1
            continue

        if n_prev < PREVIEW_MAX:
            cv2.imwrite(str(prv_dir / f"prev_{img_path.stem}.jpg"), preview)
            n_prev += 1

        if not labels:
            empty += 1
            if not preview_only:
                shutil.copy2(img_path, out_img / img_path.name)
                (out_lbl / (img_path.stem + ".txt")).write_text("")
            continue

        if not preview_only:
            shutil.copy2(img_path, out_img / img_path.name)
            (out_lbl / (img_path.stem + ".txt")).write_text(
                "\n".join(labels) + "\n")

        ok       += 1
        total_pl += n_plants

        if (i + 1) % 100 == 0 or i == len(images) - 1:
            print(f"  [{i+1:4d}/{len(images)}]  "
                  f"ok={ok}  empty={empty}  "
                  f"avg_plants={total_pl/max(ok,1):.1f}")

    print(f"\n{'='*65}")
    print(f"  Done  — {split} split")
    print(f"  Annotated : {ok} images  |  {total_pl} plant instances")
    print(f"  Empty     : {empty} images (no plant detected)")
    print(f"  Avg       : {total_pl/max(ok,1):.1f} plants/image")
    if not preview_only:
        print(f"\n  Labels  → {out_lbl}")
        print(f"  Images  → {out_img}")
    print(f"  Preview → {prv_dir}")
    print(f"\n  TUNE FLAGS (add to command if needed):")
    print(f"    Still seeing trees/sky   → --sky 0.40")
    print(f"    Missing plants at top    → --sky 0.25")
    print(f"    Missing pale/dark plants → --exg 0.03")
    print(f"    Drip tubes annotated     → --min_area 0.001")
    print(f"    Plants fused together    → --dist 0.20")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   default="./dataset/annotated/images/train")
    parser.add_argument("--output",   default="./dataset/reannotated")
    parser.add_argument("--split",    default="train", choices=["train","val"])
    parser.add_argument("--sky",      type=float, default=SKY_ZONE_FRAC,
                        help="Top fraction excluded as background (default 0.32)")
    parser.add_argument("--exg",      type=float, default=EXG_THRESH,
                        help="ExG sensitivity (default 0.05)")
    parser.add_argument("--min_area", type=float, default=MIN_AREA_FRAC,
                        help="Min plant fraction (default 0.0008)")
    parser.add_argument("--max_area", type=float, default=MAX_AREA_FRAC,
                        help="Max plant fraction (default 0.12)")
    parser.add_argument("--dist",     type=float, default=DIST_THRESH,
                        help="Watershed threshold (default 0.30)")
    parser.add_argument("--bbox",     action="store_true",
                        help="Output bounding boxes instead of seg polygons")
    parser.add_argument("--preview_only", action="store_true",
                        help="Only save previews, don't write labels")
    args = parser.parse_args()

    SKY_ZONE_FRAC = args.sky
    EXG_THRESH    = args.exg
    MIN_AREA_FRAC = args.min_area
    MAX_AREA_FRAC = args.max_area
    DIST_THRESH   = args.dist

    run(args.source, args.output, args.split,
        args.preview_only, not args.bbox)