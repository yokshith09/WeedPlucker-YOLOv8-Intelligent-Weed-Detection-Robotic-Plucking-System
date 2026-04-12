from pathlib import Path
from ultralytics import YOLO

# 🔴 SET YOUR MODEL PATH HERE
MODEL_PATH = Path(r"C:\NEWDRIVE\Model_train\Yolo_try\runs\weighted_detection\with_pretrained_weights_model\weights\best.pt")


def export():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print("\n" + "="*55)
    print("  STEP 7 — Export for Raspberry Pi 4")
    print("="*55)
    print(f"  Model : {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("\n🔄 Exporting to NCNN (optimized for Pi 4)...")

    model.export(
        format='ncnn',
        imgsz=320,          # ✅ fast inference
        half=True,          # ✅ FP16 (huge speed boost)
        simplify=True
    )

    ncnn_dir = MODEL_PATH.parent / (MODEL_PATH.stem + "_ncnn_model")

    print("\n" + "="*55)
    print("  ✅ Export complete!")
    print(f"  NCNN folder : {ncnn_dir}")
    print("="*55)

    print("\n📋 Copy to Raspberry Pi:")
    print(f'scp -r "{ncnn_dir}" pi@raspberrypi.local:/home/pi/weed_robot/')


if __name__ == '__main__':
    export()