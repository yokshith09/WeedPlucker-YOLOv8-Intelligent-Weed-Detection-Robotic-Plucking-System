from pathlib import Path

label_dir = Path(r"C:\NEWDRIVE\Model_train\dataset\kaggle_seg")

removed = 0

for txt in label_dir.glob("*.txt"):
    if txt.stat().st_size == 0:
        img = txt.with_suffix(".jpeg")

        txt.unlink(missing_ok=True)
        if img.exists():
            img.unlink(missing_ok=True)

        removed += 1

print(f"Removed {removed} empty labels")