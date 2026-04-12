"""
WEED ROBOT — Pure ONNX Runtime Inference (no ultralytics, no ROS)
=================================================================
Coordinates will match detect.py exactly.

USAGE:
    python detect_onnx.py --model model.onnx --source image.png
    python detect_onnx.py --model model.onnx --source ./test_images/
    python detect_onnx.py --model model.onnx --source image.png --conf 0.15 --z 450

OUTPUT:
    results/image_result.jpg   — annotated image
    results/image_result.json  — robot-ready coordinates
"""

import cv2
import json
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CONF        = 0.20
IMGSZ       = 640
IOU_THR     = 0.45
MIN_AREA_PX = 400
OVERLAP_THR = 0.40
Z_MM        = 500

CLS_CROP = 0
CLS_WEED = 1

COL_CROP     = (0, 220, 0)
COL_WEED_PRI = (0, 0, 255)
COL_WEED_SEC = (30, 100, 220)
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(frame):
    h, w   = frame.shape[:2]
    scale  = IMGSZ / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    resized  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_top  = (IMGSZ - nh) // 2
    pad_left = (IMGSZ - nw) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,  IMGSZ - nh - pad_top,
        pad_left, IMGSZ - nw - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    rgb  = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[np.newaxis]
    return blob, scale, pad_top, pad_left


def nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1   = np.maximum(x1[i], x1[order[1:]])
        yy1   = np.maximum(y1[i], y1[order[1:]])
        xx2   = np.minimum(x2[i], x2[order[1:]])
        yy2   = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


def decode_masks(coefs, protos, boxes_lb, orig_h, orig_w, scale, pad_top, pad_left):
    nm      = protos.shape[0]
    mh, mw  = protos.shape[1], protos.shape[2]   # 160 × 160

    raw     = (coefs @ protos.reshape(nm, -1)).reshape(-1, mh, mw)
    raw     = 1.0 / (1.0 + np.exp(-raw))          # sigmoid

    scale_p = mh / IMGSZ                           # 0.25

    final = []
    for i, m in enumerate(raw):
        x1p = int(np.clip(boxes_lb[i, 0] * scale_p, 0, mw))
        y1p = int(np.clip(boxes_lb[i, 1] * scale_p, 0, mh))
        x2p = int(np.clip(boxes_lb[i, 2] * scale_p, 0, mw))
        y2p = int(np.clip(boxes_lb[i, 3] * scale_p, 0, mh))

        cropped              = np.zeros_like(m)
        cropped[y1p:y2p, x1p:x2p] = m[y1p:y2p, x1p:x2p]

        m640   = cv2.resize(cropped, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        m_crop = m640[pad_top : pad_top + int(orig_h * scale),
                      pad_left: pad_left + int(orig_w * scale)]
        m_full = cv2.resize(m_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        final.append((m_full > 0.5).astype(np.uint8))
    return final


def filter_fp(crops, weeds, img_h, img_w):
    if not crops:
        return crops, weeds

    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    for c in crops:
        combined = cv2.bitwise_or(combined, c["mask"] * 255)
    buf      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    combined = cv2.dilate(combined, buf, iterations=1)

    clean = []
    for w in weeds:
        m       = w["mask"] * 255
        overlap = int(cv2.bitwise_and(m, combined).sum() // 255)
        frac    = overlap / max(int(m.sum() // 255), 1)
        if frac < OVERLAP_THR:
            clean.append(w)
    return crops, clean


def run_model(sess, input_name, frame):
    h, w         = frame.shape[:2]
    img_cx, img_cy = w // 2, h // 2

    blob, scale, pad_top, pad_left = preprocess(frame)
    out0, out1   = sess.run(None, {input_name: blob})
    # out0: (1, 38, 8400)  for 2-class seg model  [4 box + 2 cls + 32 coef]
    # out1: (1, 32, 160, 160)

    preds  = out0[0].T       # (8400, 38)
    protos = out1[0]         # (32, 160, 160)

    box_xywh   = preds[:, :4]
    cls_scores = preds[:, 4:6]
    coefs      = preds[:, 6:]

    cls_ids = np.argmax(cls_scores, axis=1)
    scores  = cls_scores[np.arange(len(cls_ids)), cls_ids]

    mask = scores > CONF
    if not np.any(mask):
        return [], []

    box_xywh = box_xywh[mask]
    scores   = scores[mask]
    cls_ids  = cls_ids[mask]
    coefs    = coefs[mask]

    boxes = np.stack([
        box_xywh[:, 0] - box_xywh[:, 2] / 2,
        box_xywh[:, 1] - box_xywh[:, 3] / 2,
        box_xywh[:, 0] + box_xywh[:, 2] / 2,
        box_xywh[:, 1] + box_xywh[:, 3] / 2,
    ], axis=1)

    # Per-class NMS (prevents crop suppressing nearby weed)
    keep_all = []
    for cls in [CLS_CROP, CLS_WEED]:
        idx = np.where(cls_ids == cls)[0]
        if len(idx) == 0:
            continue
        k = nms(boxes[idx], scores[idx], IOU_THR)
        keep_all.extend(idx[k].tolist())

    boxes   = boxes[keep_all]
    scores  = scores[keep_all]
    cls_ids = cls_ids[keep_all]
    coefs   = coefs[keep_all]

    masks = decode_masks(coefs, protos, boxes, h, w, scale, pad_top, pad_left)

    raw_crops, raw_weeds = [], []
    for mask_bin, cls_id, conf in zip(masks, cls_ids, scores):
        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0:
            continue
        area = len(xs)
        if area < MIN_AREA_PX:
            continue
        det = {
            "cx":   int(xs.mean()),
            "cy":   int(ys.mean()),
            "conf": round(float(conf), 3),
            "mask": mask_bin,
            "area": area,
        }
        (raw_crops if cls_id == CLS_CROP else raw_weeds).append(det)

    crops, weeds = filter_fp(raw_crops, raw_weeds, h, w)
    weeds.sort(key=lambda d: (d["cx"] - img_cx)**2 + (d["cy"] - img_cy)**2)
    return crops, weeds


def draw(frame, crops, weeds):
    out     = frame.copy()
    overlay = out.copy()
    h, w    = out.shape[:2]
    img_cx, img_cy = w // 2, h // 2

    # Filled masks
    for c in crops:
        overlay[c["mask"] > 0] = np.clip(
            overlay[c["mask"] > 0] * 0.3 + np.array([20, 200, 20]) * 0.7,
            0, 255).astype(np.uint8)

    for i, weed in enumerate(weeds):
        col   = [20, 20, 200] if i == 0 else [30, 100, 200]
        alpha = 0.70 if i == 0 else 0.55
        overlay[weed["mask"] > 0] = np.clip(
            overlay[weed["mask"] > 0] * (1 - alpha) + np.array(col) * alpha,
            0, 255).astype(np.uint8)

    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # Contours
    for c in crops:
        cnts, _ = cv2.findContours(c["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, COL_CROP, 2)

    for i, weed in enumerate(weeds):
        col  = COL_WEED_PRI if i == 0 else COL_WEED_SEC
        cnts, _ = cv2.findContours(weed["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, 2)

    # Crop labels
    for c in crops:
        cx, cy = c["cx"], c["cy"]
        cv2.circle(out, (cx, cy), 10, COL_CROP, -1)
        cv2.circle(out, (cx, cy), 10, (0,0,0), 1)
        label = f"CROP x={cx} y={cy} z={Z_MM}mm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (cx-4, cy+12), (cx+tw+8, cy+14+th+4), (0,0,0), -1)
        cv2.putText(out, label,              (cx, cy+26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_CROP, 1)
        cv2.putText(out, f"conf={c['conf']:.2f}", (cx, cy+44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,255,180), 1)

    # Weed labels
    for i, weed in enumerate(weeds):
        cx, cy   = weed["cx"], weed["cy"]
        col      = COL_WEED_PRI if i == 0 else COL_WEED_SEC
        is_pri   = (i == 0)

        cv2.circle(out, (cx, cy), 12 if is_pri else 8, col, -1)
        cv2.circle(out, (cx, cy), 12 if is_pri else 8, (0,0,0), 1)

        if is_pri:
            arm = 30
            cv2.line(out, (cx-arm, cy), (cx+arm, cy), col, 2)
            cv2.line(out, (cx, cy-arm), (cx, cy+arm), col, 2)
            ox, oy = cx - img_cx, cy - img_cy
            cv2.putText(out, f"offset x={ox:+d} y={oy:+d}px",
                        (cx, cy-38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,200,0), 1)

        tag   = "TARGET" if is_pri else f"WEED #{i}"
        label = f"{tag} x={cx} y={cy} z={Z_MM}mm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (cx-4, cy+12), (cx+tw+8, cy+14+th+4), (0,0,0), -1)
        cv2.putText(out, label, (cx, cy+26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
        cv2.putText(out, f"conf={weed['conf']:.2f}", (cx, cy+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255,160,160) if is_pri else (180,180,255), 1)

    # Banner
    if weeds:
        t      = weeds[0]
        action = f"MOVE_TO x={t['cx']} y={t['cy']} z={Z_MM}mm"
        bcol   = (0, 0, 200)
    else:
        action = "SCAN — no weeds detected"
        bcol   = (100, 100, 0)

    cv2.rectangle(out, (0, 0), (w, 50), (10,10,10), -1)
    cv2.putText(out, f"crops={len(crops)} weeds={len(weeds)} | {action}",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)

    return out


def process_one(sess, input_name, img_path, out_dir, show):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  ❌ Cannot read: {img_path}")
        return None

    h, w   = frame.shape[:2]
    img_cx = w // 2
    img_cy = h // 2

    print(f"\n{'═'*65}")
    print(f"  {img_path.name}  ({w}×{h})")
    print(f"  conf={CONF}  imgsz={IMGSZ}  z={Z_MM}mm")

    crops, weeds = run_model(sess, input_name, frame)

    print(f"  crops={len(crops)}  weeds={len(weeds)}")

    # Console output
    print(f"\n  {'─'*55}")
    print(f"  CROPS")
    for i, c in enumerate(crops):
        print(f"    #{i}  centroid=({c['cx']}, {c['cy']})  conf={c['conf']}  area={c['area']}px²")

    print(f"\n  WEEDS  (sorted closest-to-centre first)")
    for i, weed in enumerate(weeds):
        ox  = weed["cx"] - img_cx
        oy  = weed["cy"] - img_cy
        tag = "← PRIMARY TARGET" if i == 0 else ""
        print(f"    #{i}  centroid=({weed['cx']}, {weed['cy']})  conf={weed['conf']}"
              f"  offset=({ox:+d},{oy:+d})  {tag}")

    if weeds:
        t = weeds[0]
        print(f"\n  ROBOT: MOVE_TO_WEED  x={t['cx']}  y={t['cy']}  z={Z_MM}mm")
    else:
        print(f"\n  ROBOT: SCAN — no weeds")

    print(f"  {'─'*55}")

    # Draw + save
    annotated = draw(frame, crops, weeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img  = out_dir / f"{img_path.stem}_result.jpg"
    out_json = out_dir / f"{img_path.stem}_result.json"
    cv2.imwrite(str(out_img), annotated)

    def _clean(lst):
        return [{"cx": d["cx"], "cy": d["cy"], "conf": d["conf"],
                 "z": Z_MM, "area": d["area"]} for d in lst]

    robot = ({"action": "MOVE_TO_WEED",
              "x": weeds[0]["cx"], "y": weeds[0]["cy"], "z": Z_MM,
              "offset_x": weeds[0]["cx"] - img_cx,
              "offset_y": weeds[0]["cy"] - img_cy,
              "confidence": weeds[0]["conf"],
              "n_weeds": len(weeds), "n_crops": len(crops),
              "all_weeds": _clean(weeds), "all_crops": _clean(crops)}
             if weeds else
             {"action": "SCAN", "x": None, "y": None, "z": Z_MM,
              "n_weeds": 0, "n_crops": len(crops)})

    result = {"image": img_path.name, "size": {"w": w, "h": h},
              "z_mm": Z_MM, "conf_used": CONF,
              "crops": _clean(crops), "weeds": _clean(weeds),
              "robot": robot}

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved → {out_img.name}")
    print(f"  Saved → {out_json.name}")

    if show:
        sc   = min(1.0, 900 / max(w, h))
        disp = cv2.resize(annotated, (int(w*sc), int(h*sc)))
        cv2.imshow(img_path.name, disp)
        print(f"  (press any key)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def main():
    global CONF, IMGSZ, Z_MM, MIN_AREA_PX, OVERLAP_THR

    p = argparse.ArgumentParser(description="Weed robot — pure ONNX inference")
    p.add_argument("--model",   required=True, help="path to model.onnx")
    p.add_argument("--source",  required=True, help="image file or folder")
    p.add_argument("--output",  default="./results")
    p.add_argument("--conf",    type=float, default=CONF)
    p.add_argument("--z",       type=int,   default=Z_MM)
    p.add_argument("--imgsz",   type=int,   default=IMGSZ)
    p.add_argument("--no_show", action="store_true")
    args = p.parse_args()

    CONF  = args.conf
    Z_MM  = args.z
    IMGSZ = args.imgsz

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess       = ort.InferenceSession(str(model_path), providers=providers)
    input_name = sess.get_inputs()[0].name

    print(f"\n{'='*65}")
    print(f"  WEED ROBOT — Pure ONNX Runtime")
    print(f"{'='*65}")
    print(f"  Model  : {model_path.name}")
    print(f"  conf   : {CONF}  |  imgsz : {IMGSZ}  |  Z : {Z_MM}mm")
    print(f"  Device : {sess.get_providers()[0]}")
    print(f"{'='*65}")

    src  = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = ([src] if src.is_file()
            else sorted(p for p in src.iterdir() if p.suffix.lower() in exts))

    if not imgs:
        print(f"❌ No images found: {args.source}")
        return

    out_dir = Path(args.output)
    for img_path in imgs:
        process_one(sess, input_name, img_path, out_dir, show=not args.no_show)


if __name__ == "__main__":
    main()