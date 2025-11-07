# inference_save.py
# Usage example:
# python inference_save.py --weights model/best.pt --source /path/to/images --save-dir outputs --thickness 3 --font-scale 1.2 --label-padding 6

from ultralytics import YOLO
import cv2
import argparse
import os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--source", required=True, help="Image file or folder")
ap.add_argument("--save-dir", default="outputs")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--thickness", type=int, default=2, help="Box border thickness in px")
ap.add_argument("--font-scale", type=float, default=0.8, help="Font scale for labels (bigger = larger text)")
ap.add_argument("--label-padding", type=int, default=4, help="Padding in px around label text")
ap.add_argument("--font", type=int, default=cv2.FONT_HERSHEY_SIMPLEX)
args = ap.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
model = YOLO(args.weights)

results = model.predict(source=args.source, conf=args.conf, save=False)

for idx, r in enumerate(results):
    img = getattr(r, "orig_img", None)
    if img is None:
        print(f"[WARN] result {idx} has no image, skipping")
        continue

    # Convert RGB -> BGR for OpenCV if needed
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    boxes = getattr(r, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), c, conf in zip(xyxy, cls_ids, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Draw thick rectangle (box)
            color = (0, 200, 0)  # BGR green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=args.thickness, lineType=cv2.LINE_AA)

            # Prepare label text
            label = f"{model.names[int(c)]} {conf:.2f}"
            # Compute text size
            (w, h), baseline = cv2.getTextSize(label, args.font, args.font_scale, args.thickness)
            pad = args.label_padding
            # Background rectangle coordinates (top-left corner above the box if there is room)
            label_x1 = x1
            label_y1 = max(0, y1 - h - 2*pad - baseline)
            label_x2 = x1 + w + 2*pad
            label_y2 = label_y1 + h + 2*pad + baseline

            # If label would go above the image top, draw it inside the box
            if label_y1 < 0:
                label_y1 = y1
                label_y2 = y1 + h + 2*pad + baseline

            # Draw filled rectangle as label background (semi-opaque)
            # For semi-opacity, we blend a colored rectangle with the image
            overlay = img.copy()
            bg_color = (0, 200, 0)  # same as box color
            alpha = 0.8  # 0..1 opacity of the bg (0.8 -> strong)
            cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), bg_color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # Put the white text on top of the bg
            text_org = (label_x1 + pad, label_y2 - baseline - pad)
            cv2.putText(img, label, text_org, args.font, args.font_scale, (255, 255, 255), thickness=max(1, args.thickness//1), lineType=cv2.LINE_AA)

    # Output filename
    src_path = getattr(r, "path", None)
    if src_path:
        name = Path(src_path).name
    else:
        name = f"out_{idx:04d}.jpg"

    out_path = Path(args.save_dir) / name
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        print(f"[ERROR] Failed to write {out_path} (image possibly empty or path issue)")
    else:
        print("Saved:", out_path)

