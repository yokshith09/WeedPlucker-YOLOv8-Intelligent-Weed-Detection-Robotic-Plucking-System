"""
EXPORT YOLOv8-seg → ONNX  (Pi 4 / Pi 5 Optimised)
===================================================
Converts best.pt → model.onnx for Raspberry Pi / Docker+ROS deployment.

WHY ONNX over NCNN for Pi 4/5:
  ✅ ONNX keeps BOTH outputs: bbox head + segmentation head (mask prototypes)
  ✅ onnxruntime is pip-installable on Pi — no compile step
  ✅ opset=12 works on ARM64 / armhf without issues
  ⚠️  NCNN export from YOLOv8-seg often SILENTLY DROPS the mask head
  ⚠️  NCNN requires ncnn Python bindings (hard to install on Pi)
  VERDICT: Use ONNX. More reliable for seg models.

DOCKER + ROS INFERENCE NOTE:
  ⚠️  onnxruntime + ROS opencv + numpy often conflict inside Docker.
  ✅  Use detect_ultralytics.py instead — runs model.onnx via ultralytics
      YOLO() which handles all internal deps clea
      nly.
  ✅  Alternatively, pass --source ros to subscribe to a ROS image topic.

USAGE:
    python export_onnx.py                              # exports best.pt at 640
    python export_onnx.py --model best.pt --imgsz 320  # faster on Pi 4
    python export_onnx.py --model best.pt --imgsz 480  # balanced for Pi 4
    python export_onnx.py --model best.pt --imgsz 640  # best accuracy

    # Verify ONNX has segmentation output BEFORE copying to Pi:
    python export_onnx.py --verify model.onnx

    # Run inference (no onnxruntime conflicts):
    python detect_ultralytics.py --model model.onnx --source ./images/
    python detect_ultralytics.py --model model.onnx --source ros   # ROS topic mode

RASPBERRY PI SPEED GUIDE (YOLOv8n-seg, CPU only):
    imgsz=320 → ~0.8–1.2s per frame on Pi 4
    imgsz=480 → ~1.5–2.5s per frame on Pi 4
    imgsz=640 → ~3–5s per frame on Pi 4
    imgsz=320 → ~0.3–0.5s per frame on Pi 5
"""

import argparse
import shutil
from pathlib import Path
import sys


# ─── VERIFY ────────────────────────────────────────────────────────────────────

def verify(onnx_path: Path):
    """
    Check that the exported ONNX has BOTH:
      output0 — bbox + class scores + 32 mask coefficients   [1, 116, 8400]
      output1 — mask prototype tensors                        [1, 32, 160, 160]

    If output1 is missing → segmentation head was dropped → re-export.
    """
    print(f"\n{'='*60}")
    print(f"  VERIFYING: {onnx_path.name}")
    print(f"{'='*60}")

    # ── onnxruntime check ────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        sess    = ort.InferenceSession(str(onnx_path),
                                       providers=["CPUExecutionProvider"])
        inputs  = sess.get_inputs()
        outputs = sess.get_outputs()

        print(f"\n  INPUTS ({len(inputs)}):")
        for inp in inputs:
            print(f"    {inp.name:20s}  shape={inp.shape}  dtype={inp.type}")

        print(f"\n  OUTPUTS ({len(outputs)}):")
        for out in outputs:
            print(f"    {out.name:20s}  shape={out.shape}  dtype={out.type}")

        if len(outputs) >= 2:
            print(f"\n  ✅ Segmentation head present")
            print(f"     output0 = boxes + class scores + mask coefficients")
            print(f"     output1 = mask prototypes  ← segmentation head OK")
            print(f"\n  ✅ ONNX is ready for deployment")
        else:
            print(f"\n  ❌ ONLY {len(outputs)} output found — segmentation head MISSING")
            print(f"     → Re-export with: python export_onnx.py --model best.pt")
            print(f"     → Make sure ultralytics is up to date:")
            print(f"          pip install ultralytics --upgrade")
        return len(outputs) >= 2

    except ImportError:
        print(f"  ⚠️  onnxruntime not installed — trying onnx library instead")

    # ── fallback: onnx library check ─────────────────────────────────────────
    try:
        import onnx
        model   = onnx.load(str(onnx_path))
        outputs = [o.name for o in model.graph.output]

        print(f"  ONNX outputs: {outputs}")
        if len(outputs) >= 2:
            print(f"  ✅ Segmentation output present")
            print(f"     output0 = bounding boxes + class scores")
            print(f"     output1 = mask prototype coefficients  ← seg head OK")
        else:
            print(f"  ❌ Only {len(outputs)} output — seg head may be missing")
            print(f"     Re-export: python export_onnx.py --model best.pt")
        return len(outputs) >= 2

    except ImportError:
        print(f"  Install one of these to verify:")
        print(f"      pip install onnxruntime   (recommended)")
        print(f"      pip install onnx")
        return None


# ─── EXPORT ────────────────────────────────────────────────────────────────────

def export(model_path: Path, imgsz: int, output_name: str = "model.onnx") -> Path:
    """
    Export YOLOv8-seg .pt → .onnx with Pi-safe settings, saved as model.onnx.

    Key settings explained:
      format   = "onnx"   — standard ONNX format
      imgsz    = 640      — MUST match training imgsz; 320/480 for Pi speed
      half     = False    — Pi 4 ARM CPU has NO float16 support → crashes if True
      simplify = True     — removes dead graph nodes → smaller + faster on Pi
      opset    = 12       — opset 12 is safest for onnxruntime on Pi (armhf/arm64)
                            opset 17+ can fail on older onnxruntime builds on Pi
      dynamic  = False    — fixed batch=1; dynamic shapes break mask upsampling
                            in some onnxruntime versions on Pi
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics not installed")
        print("     pip install ultralytics")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  EXPORT: {model_path.name} → ONNX")
    print(f"{'='*60}")
    print(f"  output   : {output_name}")
    print(f"  imgsz    : {imgsz}px   (match your training imgsz!)")
    print(f"  half     : False      (Pi 4 CPU — no float16 support)")
    print(f"  opset    : 12         (safe for onnxruntime on Pi armhf/arm64)")
    print(f"  simplify : True       (removes dead graph nodes)")
    print(f"  dynamic  : False      (fixed batch=1 for Pi inference)")
    print(f"{'='*60}\n")

    model = YOLO(str(model_path))

    model.export(
        format   = "onnx",
        imgsz    = imgsz,
        half     = False,
        simplify = True,
        opset    = 12,
        dynamic  = False,
    )

    # ultralytics saves as best.onnx (same stem as .pt) — rename to model.onnx
    default_onnx = model_path.with_suffix(".onnx")
    output_path  = model_path.parent / output_name

    if default_onnx.exists():
        if default_onnx != output_path:
            shutil.move(str(default_onnx), str(output_path))
            print(f"\n  Renamed: {default_onnx.name} → {output_path.name}")

        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"\n{'='*60}")
        print(f"  ✅ Export complete!")
        print(f"  File : {output_path}")
        print(f"  Size : {size_mb:.1f} MB")
        print(f"{'='*60}")
        _print_next_steps(output_path)
    else:
        print(f"\n  ❌ Export failed — {default_onnx} not found")
        print(f"     Try: pip install ultralytics --upgrade")

    return output_path


def _print_next_steps(onnx_path: Path):
    print(f"""
  NEXT STEPS
  ──────────────────────────────────────────────────────
  1.  Verify seg head is intact (DO THIS BEFORE copying to Pi):
          python export_onnx.py --verify {onnx_path.name}

      You MUST see 2 outputs. If only 1 → re-export.

  2.  Run inference WITHOUT onnxruntime conflicts (Docker/ROS safe):
          python detect_ultralytics.py --model {onnx_path.name} --source image.jpg

      For ROS topic mode (subscribes to /camera/image_raw):
          python detect_ultralytics.py --model {onnx_path.name} --source ros

  3.  Copy to Pi 4:
          scp {onnx_path.name}             pi@raspberrypi.local:/home/pi/weed_robot/
          scp detect_ultralytics.py        pi@raspberrypi.local:/home/pi/weed_robot/

  4.  Install on Pi (no onnxruntime needed separately — ultralytics handles it):
          pip3 install ultralytics

  5.  Run on Pi:
          python3 detect_ultralytics.py --model model.onnx --source ./images/

  6.  If too slow on Pi 4 → re-export at smaller imgsz:
          python export_onnx.py --model best.pt --imgsz 320
  ──────────────────────────────────────────────────────
""")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Export YOLOv8-seg → ONNX for Raspberry Pi / Docker+ROS"
    )
    p.add_argument(
        "--model",  default="best.pt",
        help="Path to best.pt (default: best.pt)"
    )
    p.add_argument(
        "--imgsz",  type=int, default=640,
        help=(
            "Export image size (default: 640). "
            "Use 320 for faster Pi 4 inference (~1s/frame). "
            "MUST match your training imgsz."
        )
    )
    p.add_argument(
        "--output", default="model.onnx",
        help="Output ONNX filename (default: model.onnx)"
    )
    p.add_argument(
        "--verify", default=None,
        metavar="ONNX_PATH",
        help="Path to .onnx file to verify seg outputs (skips export)"
    )
    args = p.parse_args()

    # ── Verify-only mode ──────────────────────────────────────────────────────
    if args.verify:
        vpath = Path(args.verify)
        if not vpath.exists():
            print(f"❌ File not found: {vpath}")
            sys.exit(1)
        ok = verify(vpath)
        sys.exit(0 if ok else 1)

    # ── Export mode ───────────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print(f"   Pass the correct path:")
        print(f"       python export_onnx.py --model /path/to/best.pt")
        sys.exit(1)

    onnx_path = export(model_path, args.imgsz, args.output)

    # Auto-verify after export
    if onnx_path.exists():
        print(f"\n  Auto-verifying exported model...")
        ok = verify(onnx_path)
        if not ok:
            print(f"\n  ❌ Verification failed — do NOT copy to Pi yet")
            print(f"     Try upgrading ultralytics and re-exporting:")
            print(f"         pip install ultralytics --upgrade")
            print(f"         python export_onnx.py --model best.pt")
            sys.exit(1)
        else:
            print(f"\n  ✅ Model verified — safe to copy to Pi")


if __name__ == "__main__":
    main()