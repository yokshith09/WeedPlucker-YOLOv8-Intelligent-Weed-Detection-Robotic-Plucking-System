# Inference Scripts

## `detect.py` — Full Inference Pipeline

Complete weed detection inference with robot-ready coordinate output.

**Features:**
- Colored segmentation masks drawn directly on image
- Centroid crosshairs on each detected plant
- Pixel + normalized coordinates for robot control
- False positive filtering (overlap + minimum area)
- Primary target selection (closest weed to image centre)
- JSON output with robot action commands

```bash
# Single image
python detect.py --model best.pt --source image.jpg

# Folder of images
python detect.py --model best.pt --source ./test_images/

# ONNX model
python detect.py --model best.onnx --source ./test_images/

# Tune sensitivity
python detect.py --model best.pt --source image.jpg --conf 0.12  # more detections
python detect.py --model best.pt --source image.jpg --conf 0.35  # fewer false positives
```

---

## `detect_onnx.py` — Pure ONNX Runtime Inference

Runs inference using only `onnxruntime` — **no ultralytics dependency required**.
Ideal for Docker/ROS deployments where package conflicts occur.

Implements from scratch:
- Letterbox preprocessing
- Per-class NMS
- Mask prototype decoding
- Mask upsampling to original resolution

```bash
python detect_onnx.py --model model.onnx --source image.png --conf 0.20
```

---

## `robot_pi_inference.py` — Raspberry Pi Camera Loop

Optimized for real-time operation on Raspberry Pi 4/5:
- Camera capture at 320×320
- Segmentation-based detection
- Target selection prioritizing closest + lowest weeds
- `PLUCK` vs `MOVE_TO_WEED` action based on proximity to center
- Headless mode for piping JSON to robot controller

```bash
# With display
python robot_pi_inference.py --model ./best_ncnn_model

# Headless (JSON to stdout)
python robot_pi_inference.py --model ./best_ncnn_model --headless
```

### Robot JSON Output

```json
{
  "action": "MOVE_TO_WEED",
  "target_px": [160, 200],
  "offset_x_px": -40,
  "offset_y_px": +60,
  "confidence": 0.82,
  "total_weeds": 2,
  "total_crops": 3
}
```
