# AI Retail Shelf Detector — YOLOv8 (Production prototype)

**Live demo:** [Gayan32/Rack_Detector](https://huggingface.co/spaces/Gayan32/Rack_Detector)
## Summary
- **Task:** Object detection (rack / primary shelf / different)
- **Dataset:** ~300 images (annotated) +fine tuned by 37 images 
- **Model:** YOLOv8 small — trained 70 epochs
- **Validation:** mAP50=0.837 | mAP50-95=0.682 | Precision=0.833 | Recall=0.717

## What’s in this repo
- `model/best.pt` (weights) — this is the fine tuned version with 37 images.
- `inference.py` — run predictions on images
- `demo/` — sample inputs & outputs
- `logs/training_plots.png` — training curves
- Live demo on HuggingFace Spaces (Gradio)

## Quick test (local)
```bash
pip install -r requirements.txt
python inference_save.py --weights model/best.pt --source image/folder --save-dir outputs --thickness 3 --font-scale 1.1 --label-padding 6

```

## What's in Notes & Limitations

- Trained on internal retail images; performance may drop on totally different stores/cameras.
- To reach production-grade recall for rack, add 300–500 more labeled examples for that class.

Contact: sasankamaduwantha1@protonmail.com
