import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path(r"C:\NEWDRIVE\Model_train\dataset\kaggle_weed")
OUTPUT_DIR = Path(r"C:\NEWDRIVE\Model_train\dataset\kaggle_seg")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = ['.jpg', '.jpeg', '.png']

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    return x1, y1, x2, y2

def mask_to_polygon(mask, img_w, img_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        pts = cnt.reshape(-1, 2)
        norm = []
        for x, y in pts:
            norm.append(x / img_w)
            norm.append(y / img_h)
        polys.append(norm)
    return polys

for img_path in tqdm(list(INPUT_DIR.glob("*.*"))):
    if img_path.suffix.lower() not in IMG_EXTS:
        continue

    label_path = img_path.with_suffix(".txt")
    if not label_path.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    lines = open(label_path).read().strip().splitlines()
    new_labels = []

    for line in lines:
        cls, xc, yc, bw, bh = map(float, line.split())
        x1, y1, x2, y2 = yolo_to_bbox(xc, yc, bw, bh, w, h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # GrabCut
        mask = np.zeros(roi.shape[:2], np.uint8)
        bgd = np.zeros((1,65), np.float64)
        fgd = np.zeros((1,65), np.float64)

        rect = (1, 1, roi.shape[1]-2, roi.shape[0]-2)
        cv2.grabCut(roi, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)

        mask = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')

        polys = mask_to_polygon(mask, roi.shape[1], roi.shape[0])

        for poly in polys:
            # shift back to original image
            shifted = []
            for i in range(0, len(poly), 2):
                px = poly[i] * roi.shape[1] + x1
                py = poly[i+1] * roi.shape[0] + y1
                shifted.append(px / w)
                shifted.append(py / h)

            new_labels.append(f"{int(cls)} " + " ".join(map(str, shifted)))

    # Save outputs
    cv2.imwrite(str(OUTPUT_DIR / img_path.name), img)

    with open(OUTPUT_DIR / (img_path.stem + ".txt"), "w") as f:
        f.write("\n".join(new_labels))