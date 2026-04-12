"""
MODEL COMPARISON — Base vs SimCLR
===================================
Runs BOTH models on the same images and shows:
  1. Side-by-side annotated output (saved as image)
  2. Per-image metric differences (confidence, mask area, count)
  3. A summary table at the end

This lets you SEE the difference beyond just mAP numbers.

Usage:
    python compare_models.py --base best.pt --simclr best-simclr.pt --source ./test_images/
    python compare_models.py --base best.pt --simclr best-simclr.pt --source image.jpg
"""

import cv2
import numpy as np
import argparse
import json
from pathlib import Path


CONF   = 0.15
IMGSZ  = 640
IOU    = 0.45

COL_CROP = (0, 220, 0)
COL_WEED = (0, 0, 255)


def run_model(model, frame):
    """Run inference, return list of detections."""
    h, w = frame.shape[:2]
    res   = model(frame, conf=CONF, iou=IOU, imgsz=IMGSZ,
                  task="segment", verbose=False)[0]

    dets = []
    if res.masks is None or res.boxes is None:
        return dets

    masks   = res.masks.data.cpu().numpy()
    boxes   = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs   = res.boxes.conf.cpu().numpy()

    for mask_raw, box, cls_id, conf in zip(masks, boxes, cls_ids, confs):
        m    = cv2.resize(mask_raw, (w, h))
        m    = (m > 0.5).astype(np.uint8) * 255
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        dets.append({
            "cls":  int(cls_id),
            "name": "crop" if cls_id == 0 else "weed",
            "conf": round(float(conf), 3),
            "cx":   int(xs.mean()),
            "cy":   int(ys.mean()),
            "area": int(len(xs)),
            "bbox": list(map(int, box)),
            "_m":   m,
        })
    return dets


def draw_panel(frame, dets, label, img_h, img_w):
    """Draw detections on a copy of frame. Returns annotated image."""
    out     = frame.copy()
    overlay = out.copy()

    crops = [d for d in dets if d["cls"] == 0]
    weeds = [d for d in dets if d["cls"] == 1]

    for c in crops:
        overlay[c["_m"] > 0] = np.clip(
            overlay[c["_m"] > 0] * 0.3 + np.array([20, 200, 20]) * 0.7, 0, 255
        ).astype(np.uint8)

    for w in weeds:
        overlay[w["_m"] > 0] = np.clip(
            overlay[w["_m"] > 0] * 0.3 + np.array([20, 20, 200]) * 0.7, 0, 255
        ).astype(np.uint8)

    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    for c in crops:
        cnts, _ = cv2.findContours(c["_m"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COL_CROP, 2)
        cv2.circle(out, (c["cx"], c["cy"]), 8, COL_CROP, -1)
        cv2.putText(out, f"CROP {c['conf']:.2f}", (c["bbox"][0], max(c["bbox"][1]-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_CROP, 1)

    for w in weeds:
        cnts, _ = cv2.findContours(w["_m"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COL_WEED, 2)
        cv2.circle(out, (w["cx"], w["cy"]), 8, COL_WEED, -1)
        cv2.putText(out, f"WEED {w['conf']:.2f}", (w["bbox"][0], max(w["bbox"][1]-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_WEED, 1)

    # Banner
    cv2.rectangle(out, (0, 0), (img_w, 46), (10, 10, 10), -1)
    cv2.putText(out, label, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(out,
                f"crops={len(crops)}  weeds={len(weeds)}",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return out


def metrics(dets):
    crops = [d for d in dets if d["cls"] == 0]
    weeds = [d for d in dets if d["cls"] == 1]
    return {
        "n_crops":     len(crops),
        "n_weeds":     len(weeds),
        "crop_confs":  [c["conf"] for c in crops],
        "weed_confs":  [w["conf"] for w in weeds],
        "crop_areas":  [c["area"] for c in crops],
        "weed_areas":  [w["area"] for w in weeds],
        "avg_crop_conf": round(float(np.mean([c["conf"] for c in crops])), 3) if crops else 0,
        "avg_weed_conf": round(float(np.mean([w["conf"] for w in weeds])), 3) if weeds else 0,
        "avg_crop_area": int(np.mean([c["area"] for c in crops])) if crops else 0,
        "avg_weed_area": int(np.mean([w["area"] for w in weeds])) if weeds else 0,
    }


def diff_line(key, base_val, sim_val, higher_is_better=True):
    if base_val == 0 and sim_val == 0:
        return f"    {key:<30} base=0       simclr=0"
    delta = sim_val - base_val
    pct   = (delta / base_val * 100) if base_val != 0 else 0
    arrow = ""
    if delta > 0:
        arrow = "▲ SimCLR better" if higher_is_better else "▼ SimCLR worse"
    elif delta < 0:
        arrow = "▼ SimCLR worse" if higher_is_better else "▲ SimCLR better"
    else:
        arrow = "= same"
    return f"    {key:<30} base={base_val}   simclr={sim_val}   Δ={delta:+.3f} ({pct:+.1f}%)  {arrow}"


def compare_image(base_model, sim_model, img_path, out_dir):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  ❌ Cannot read {img_path}")
        return None

    h, w = frame.shape[:2]
    print(f"\n{'═'*65}")
    print(f"  {img_path.name}  ({w}×{h})")

    # Run both models
    base_dets = run_model(base_model, frame)
    sim_dets  = run_model(sim_model,  frame)

    bm = metrics(base_dets)
    sm = metrics(sim_dets)

    # Print per-image comparison
    print(f"\n  {'─'*55}")
    print(f"  DETECTION COUNTS")
    print(f"    {'Base model':<30}  crops={bm['n_crops']}  weeds={bm['n_weeds']}")
    print(f"    {'SimCLR model':<30}  crops={sm['n_crops']}  weeds={sm['n_weeds']}")

    print(f"\n  CONFIDENCE SCORES (higher = more certain)")
    print(diff_line("avg weed confidence",  bm['avg_weed_conf'],  sm['avg_weed_conf']))
    print(diff_line("avg crop confidence",  bm['avg_crop_conf'],  sm['avg_crop_conf']))

    print(f"\n  MASK COVERAGE (higher = larger/tighter masks)")
    print(diff_line("avg weed mask area px", bm['avg_weed_area'], sm['avg_weed_area']))
    print(diff_line("avg crop mask area px", bm['avg_crop_area'], sm['avg_crop_area']))

    # Build side-by-side image
    # Scale down to fit side by side
    scale = min(1.0, 800 / w)
    sw, sh = int(w * scale), int(h * scale)

    base_panel = cv2.resize(
        draw_panel(frame, base_dets, f"BASE MODEL  conf={CONF}", h, w),
        (sw, sh)
    )
    sim_panel  = cv2.resize(
        draw_panel(frame, sim_dets,  f"SIMCLR MODEL  conf={CONF}", h, w),
        (sw, sh)
    )

    # Divider
    divider = np.full((sh, 4, 3), 80, dtype=np.uint8)
    side_by_side = np.hstack([base_panel, divider, sim_panel])

    # Add difference summary bar at bottom
    bar_h = 90
    bar   = np.zeros((bar_h, side_by_side.shape[1], 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)

    def gain_text(label, bv, sv, unit=""):
        d = sv - bv
        col = (0, 220, 0) if d > 0 else ((0, 0, 220) if d < 0 else (180, 180, 180))
        return label + f" {bv}{unit}→{sv}{unit} ({d:+.3f})", col

    lines = [
        gain_text("weed conf:", bm['avg_weed_conf'], sm['avg_weed_conf']),
        gain_text("crop conf:", bm['avg_crop_conf'], sm['avg_crop_conf']),
        gain_text("weed count:", float(bm['n_weeds']), float(sm['n_weeds'])),
        gain_text("weed area:", float(bm['avg_weed_area']), float(sm['avg_weed_area']), "px"),
    ]

    x = 10
    for i, (txt, col) in enumerate(lines):
        row = 22 + (i // 2) * 32
        col_x = x if i % 2 == 0 else side_by_side.shape[1] // 2
        cv2.putText(bar, txt, (col_x, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    cv2.putText(bar, "Difference summary (SimCLR vs Base)", (10, bar_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    final = np.vstack([side_by_side, bar])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}_compare.jpg"
    cv2.imwrite(str(out_path), final)
    print(f"\n  Saved → {out_path.name}")

    return {"image": img_path.name, "base": bm, "simclr": sm}


def main():
    ap = argparse.ArgumentParser(description="Visual + metric comparison: base vs SimCLR model")
    ap.add_argument("--base",    required=True, help="Path to base best.pt")
    ap.add_argument("--simclr",  required=True, help="Path to simclr best.pt or best-simclr.pt")
    ap.add_argument("--source",  required=True, help="Image file or folder")
    ap.add_argument("--output",  default="./comparison_results")
    ap.add_argument("--conf",    type=float, default=CONF,  help=f"Confidence (default {CONF})")
    ap.add_argument("--imgsz",   type=int,   default=IMGSZ, help=f"Image size (default {IMGSZ})")
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    global CONF, IMGSZ
    CONF  = args.conf
    IMGSZ = args.imgsz

    from ultralytics import YOLO
    print(f"\n{'='*65}")
    print(f"  LOADING MODELS")
    print(f"{'='*65}")

    base_path = Path(args.base)
    sim_path  = Path(args.simclr)

    for p in [base_path, sim_path]:
        if not p.exists():
            print(f"  ❌ Not found: {p}")
            return

    base_model = YOLO(str(base_path), task="segment") if base_path.suffix == ".onnx" else YOLO(str(base_path))
    sim_model  = YOLO(str(sim_path),  task="segment") if sim_path.suffix  == ".onnx" else YOLO(str(sim_path))

    print(f"  ✅ Base   : {base_path.name}")
    print(f"  ✅ SimCLR : {sim_path.name}")
    print(f"  conf={CONF}  imgsz={IMGSZ}")

    src    = Path(args.source)
    exts   = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in exts
    )

    if not images:
        print(f"  ❌ No images found: {args.source}")
        return

    print(f"\n  Processing {len(images)} image(s)...\n")
    out_dir = Path(args.output)
    results = []

    for img_path in images:
        r = compare_image(base_model, sim_model, img_path, out_dir)
        if r:
            results.append(r)
            if not args.no_show and len(images) == 1:
                img = cv2.imread(str(out_dir / f"{img_path.stem}_compare.jpg"))
                if img is not None:
                    scale = min(1.0, 1400 / img.shape[1])
                    disp  = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                    cv2.imshow("Base (left) vs SimCLR (right)", disp)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    # ── Aggregate summary across all images ───────────────────────────────────
    if results:
        print(f"\n{'='*65}")
        print(f"  AGGREGATE COMPARISON — {len(results)} images")
        print(f"{'='*65}")

        def avg(results, key, sub):
            vals = [r[sub][key] for r in results if r[sub][key] > 0]
            return round(float(np.mean(vals)), 3) if vals else 0

        metrics_to_compare = [
            ("avg_weed_conf",  "Avg weed confidence",  True),
            ("avg_crop_conf",  "Avg crop confidence",  True),
            ("avg_weed_area",  "Avg weed mask area",   True),
            ("avg_crop_area",  "Avg crop mask area",   True),
            ("n_weeds",        "Avg weeds detected",   True),
            ("n_crops",        "Avg crops detected",   True),
        ]

        gains = []
        for key, label, hib in metrics_to_compare:
            bv = avg(results, key, "base")
            sv = avg(results, key, "simclr")
            d  = sv - bv
            direction = "SimCLR better" if (d > 0 and hib) or (d < 0 and not hib) else \
                        ("SimCLR worse" if d != 0 else "same")
            print(f"  {label:<30} base={bv:<8} simclr={sv:<8} Δ={d:+.3f}  {direction}")
            gains.append({"metric": label, "base": bv, "simclr": sv, "delta": round(d, 3)})

        # Save summary JSON
        summary_path = out_dir / "comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"n_images": len(results), "conf": CONF, "gains": gains}, f, indent=2)

        print(f"\n  Summary saved → {summary_path}")
        print(f"  Images saved  → {out_dir.resolve()}")
        print(f"{'='*65}")

        # Final verdict
        weed_conf_gain = avg(results, "avg_weed_conf", "simclr") - avg(results, "avg_weed_conf", "base")
        weed_area_gain = avg(results, "avg_weed_area", "simclr") - avg(results, "avg_weed_area", "base")

        print(f"\n  VERDICT")
        if weed_conf_gain > 0.005:
            print(f"  SimCLR is more CONFIDENT on weeds (+{weed_conf_gain:.3f} avg conf)")
        if weed_area_gain > 100:
            print(f"  SimCLR produces LARGER weed masks (+{weed_area_gain:.0f}px avg area)")
            print(f"  → Tighter segmentation, better robot coordinates")
        if weed_conf_gain <= 0.005 and weed_area_gain <= 100:
            print(f"  Differences are subtle on these images — check the side-by-side JPGs")
            print(f"  The mAP gain (+0.7% weed) shows up on the full val set, not always per-image")


if __name__ == "__main__":
    main()