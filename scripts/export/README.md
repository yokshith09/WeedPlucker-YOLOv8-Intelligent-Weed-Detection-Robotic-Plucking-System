# Model Export Scripts

Tools for converting trained YOLOv8-segmentation models into deployment-ready formats.

## Key Scripts

### `export_onnx.py`
Exports PyTorch weights (`.pt`) to ONNX format. 
**Crucial for Raspberry Pi:** It includes validation logic to ensure the segmentation prototype heads are preserved (often lost in standard exports).

```bash
# Standard export
python export_onnx.py --model weights/best.pt

# Export at lower resolution for Pi 4 speed
python export_onnx.py --model weights/best.pt --imgsz 320

# Verify an existing ONNX model
python export_onnx.py --verify model.onnx
```

### `export_ncnn_pi.py`
Exports model directly to NCNN format, which is the most optimized format for the Raspberry Pi 4's CPU (ARM).
- Enables FP16 (half-precision) for 2x speed boost.
- Simplifies the computation graph.

```bash
python export_ncnn_pi.py --model weights/best.pt
```

## Comparison: ONNX vs NCNN

| Format | Library | Best For | Typical Latency (Pi 4) |
|--------|---------|----------|------------------------|
| **ONNX** | `onnxruntime` | Pi 5, Generic PC, MacOS | ~0.4s - 0.7s |
| **NCNN** | `ncnn` | Pi 4, Mobile, ARM64 | ~0.6s - 0.9s |
| **PyTorch**| `ultralytics` | Development, GPU Server | ~2.6ms (GPU) / ~2s (CPU) |
