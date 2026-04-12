import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path

# ================= CONFIG =================
CONF_THRESH   = 0.35
IOU_THRESH    = 0.45
INPUT_SIZE    = 320
PLUCK_ZONE_PX = 25
MIN_WEED_AREA = 200   # ignore tiny detections
# ==========================================


# 🔴 FIXED MODEL PATH (FOR PI)
def find_model():
    ncnn = Path("/home/pi/weed_robot/best_ncnn_model")
    if ncnn.exists():
        return str(ncnn)

    pt = Path("best.pt")  # fallback for testing
    if pt.exists():
        return str(pt)

    return None


def process_frame(model, frame):
    h, w = frame.shape[:2]
    img_cx, img_cy = w // 2, h // 2

    results = model(
        frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=INPUT_SIZE,
        verbose=False
    )[0]

    crops = []
    weeds = []

    # ✅ SEGMENTATION (FIXED)
    if results.masks is not None:
        masks = results.masks.xy
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for seg, cls, conf in zip(masks, classes, confs):
            seg = np.array(seg, dtype=np.int32)

            x1, y1 = np.min(seg[:, 0]), np.min(seg[:, 1])
            x2, y2 = np.max(seg[:, 0]), np.max(seg[:, 1])

            area = (x2 - x1) * (y2 - y1)
            if area < MIN_WEED_AREA:
                continue

            cx = int(np.mean(seg[:, 0]))
            cy = int(np.mean(seg[:, 1]))

            entry = {
                "center_px": [cx, cy],
                "center_norm": [round(cx/w,4), round(cy/h,4)],
                "bbox_px": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(float(conf), 3),
                "class": "crop" if cls == 0 else "weed"
            }

            if cls == 0:
                crops.append(entry)
            else:
                weeds.append(entry)

    # 🎯 TARGET SELECTION (IMPROVED)
    if not weeds:
        cmd = {
            "action": "SCAN",
            "target_px": None,
            "offset_x_px": 0,
            "offset_y_px": 0,
            "total_weeds": 0,
            "total_crops": len(crops),
            "message": "No weeds — scanning"
        }
    else:
        def dist(weed):
            cx, cy = weed["center_px"]
            return ((cx - img_cx)**2 + (cy - img_cy)**2)**0.5 - (cy * 0.5)

        target = min(weeds, key=dist)
        tcx, tcy = target["center_px"]

        offset_x = tcx - img_cx
        offset_y = tcy - img_cy

        at_target = abs(offset_x) < PLUCK_ZONE_PX and abs(offset_y) < PLUCK_ZONE_PX
        action = "PLUCK" if at_target else "MOVE_TO_WEED"

        cmd = {
            "action": action,
            "target_px": [tcx, tcy],
            "target_norm": target["center_norm"],
            "offset_x_px": offset_x,
            "offset_y_px": offset_y,
            "confidence": target["confidence"],
            "total_weeds": len(weeds),
            "total_crops": len(crops),
            "all_weeds": [w["center_px"] for w in weeds],
            "all_crops": [c["center_px"] for c in crops],
            "message": f"{action} weed@({tcx},{tcy}) offset({offset_x:+d},{offset_y:+d})"
        }

    # 🎨 DRAW OUTPUT
    out = frame.copy()

    # crops → green
    for c in crops:
        x1,y1,x2,y2 = c["bbox_px"]
        cx,cy = c["center_px"]
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,200,0), 2)
        cv2.circle(out, (cx,cy), 5, (0,200,0), -1)

    # weeds → red
    for weed in weeds:
        x1,y1,x2,y2 = weed["bbox_px"]
        cx,cy = weed["center_px"]

        is_target = (cmd.get("target_px") == [cx, cy])
        color = (0,0,255) if is_target else (0,120,255)

        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.circle(out, (cx,cy), 6, color, -1)

        if is_target:
            cv2.line(out, (cx-15,cy),(cx+15,cy), color, 2)
            cv2.line(out, (cx,cy-15),(cx,cy+15), color, 2)

    # center marker
    cv2.drawMarker(out, (img_cx, img_cy), (255,255,0), cv2.MARKER_CROSS, 20, 1)

    return cmd, out


def run_camera(model, headless):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not found")
        return

    print("📷 Running... press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ⚡ SPEED BOOST
        frame = cv2.resize(frame, (320, 320))

        t0 = time.time()
        cmd, annotated = process_frame(model, frame)
        fps = 1 / (time.time() - t0)

        if headless:
            print(json.dumps(cmd), flush=True)
        else:
            cv2.putText(annotated, f"FPS:{fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.imshow("Weed Robot", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--source', default='camera')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    model_path = args.model or find_model()

    if not model_path:
        print("❌ No model found")
        return

    from ultralytics import YOLO
    print(f"✅ Loading model: {model_path}")
    model = YOLO(model_path)

    if args.source == "camera":
        run_camera(model, args.headless)


if __name__ == "__main__":
    main()