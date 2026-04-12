import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG =================
INPUT_DIR  = Path(r"C:\NEWDRIVE\Model_train\dataset\annotated\images\train")
OUTPUT_DIR = Path(r"C:\NEWDRIVE\Model_train\dataset\clean_annotated")

CLASS_ID = 0  # crop
VAL_SPLIT = 0.15

MIN_AREA = 500
MAX_AREA_FRAC = 0.6
# ==========================================


def segment_crop(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---- 1. STRICT GREEN DETECTION (crop only) ----
    green = cv2.inRange(hsv, (65, 80, 60), (105, 255, 200))

    # ---- 2. REMOVE SOIL ----
    soil = cv2.inRange(hsv, (0, 30, 30), (25, 255, 200))

    # ---- 3. REMOVE BLACK PIPES ----
    black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))

    # ---- 4. REMOVE BACKGROUND TREES (high saturation) ----
    tree = cv2.inRange(hsv, (60, 150, 50), (120, 255, 255))

    noise = cv2.bitwise_or(soil, black)
    noise = cv2.bitwise_or(noise, tree)

    mask = cv2.bitwise_and(green, cv2.bitwise_not(noise))

    # ---- 5. CLEANUP ----
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    return mask


def mask_to_yolo(mask, w, h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        if area > (w * h * MAX_AREA_FRAC):
            continue

        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2)
        norm = []

        for x, y in pts:
            norm.append(round(x / w, 6))
            norm.append(round(y / h, 6))

        labels.append(f"{CLASS_ID} " + " ".join(map(str, norm)))

    return labels


def run():
    images = list(INPUT_DIR.glob("*.*"))

    train_img_dir = OUTPUT_DIR / "images/train"
    val_img_dir   = OUTPUT_DIR / "images/val"
    train_lbl_dir = OUTPUT_DIR / "labels/train"
    val_lbl_dir   = OUTPUT_DIR / "labels/val"

    for p in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        p.mkdir(parents=True, exist_ok=True)

    import random
    random.shuffle(images)
    split_idx = int(len(images) * (1 - VAL_SPLIT))

    for i, img_path in enumerate(tqdm(images)):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = segment_crop(img)
        labels = mask_to_yolo(mask, w, h)

        split = "train" if i < split_idx else "val"

        if split == "train":
            img_out = train_img_dir / img_path.name
            lbl_out = train_lbl_dir / (img_path.stem + ".txt")
        else:
            img_out = val_img_dir / img_path.name
            lbl_out = val_lbl_dir / (img_path.stem + ".txt")

        shutil.copy(img_path, img_out)

        with open(lbl_out, "w") as f:
            f.write("\n".join(labels))

    print("\n✅ DONE — Clean dataset ready!")


if __name__ == "__main__":
    run()