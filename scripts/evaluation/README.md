# Model Evaluation & Comparison

## `compare_models.py` — Base vs SimCLR Visual Comparison

Side-by-side comparison between two trained models on the same test images.

**Outputs per image:**
- Side-by-side annotated panels (base left, SimCLR right)
- Color-coded masks: green = crop, red = weed
- Per-image metrics: confidence, mask area, detection count
- Difference summary bar with gain/loss indicators

**Aggregate outputs:**
- Summary table across all images
- `comparison_summary.json` with per-metric deltas
- Verdict: which model is better for weed detection

```bash
python compare_models.py \
    --base path/to/base/best.pt \
    --simclr path/to/simclr/best.pt \
    --source ./test_images/ \
    --output ./comparison_results/ \
    --conf 0.15
```

### Metrics Compared

| Metric | Higher is Better? |
|--------|:-:|
| Avg weed confidence | Yes |
| Avg crop confidence | Yes |
| Avg weed mask area | Yes (tighter segmentation) |
| Avg crop mask area | Yes |
| Weeds detected | Yes |
| Crops detected | Yes |

---

## `test_hybrid_exg.py` — Hybrid YOLOv8 + ExG Test

Uses YOLOv8 for cauliflower detection, then ExG (Excess Green Index) vegetation
masking to find weeds by exclusion. Useful for comparing pure model detection
vs hybrid approaches.

```bash
python test_hybrid_exg.py --model best.pt --source ./test_images/ --exg 0.08
```
