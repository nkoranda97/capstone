import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

""""
Gemini 3 assisted with converting prediction to image that can be showed
"""


def show_yolo_prediction(
    model_path: str,
    root_dir: str,
    split: str = "val",
    conf: float = 0.25,
    seed: int | None = None,
):
    """
    Display a random YOLO-formatted image from the NEW dataset structure
    (RGB .jpg images) with ground-truth and predicted boxes.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    class_names = ["nodule"]

    img_dir = os.path.join(root_dir, "images", split)
    label_dir = os.path.join(root_dir, "labels", split)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]

    img_name = random.choice(img_files)
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]

    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                cls = int(parts[0])
                xc, yc, bw, bh = parts[1:]

                x0 = (xc - bw / 2) * w
                y0 = (yc - bh / 2) * h
                x1 = (xc + bw / 2) * w
                y1 = (yc + bh / 2) * h
                gt_boxes.append((x0, y0, x1, y1, cls))

    model = YOLO(model_path)
    results = model(img_path, conf=conf, verbose=False)
    result = results[0]

    pred_img_bgr = result.plot(line_width=2)
    pred_img_rgb = pred_img_bgr[..., ::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(img)
    axes[0].set_title(f"Ground Truth (RGB Input)\n{img_name}")

    for x0, y0, x1, y1, cls in gt_boxes:
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="#00FF00", lw=2)
        axes[0].add_patch(rect)
        label_txt = class_names[cls] if cls < len(class_names) else str(cls)
        axes[0].text(
            x0,
            y0 - 5,
            label_txt,
            color="white",
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor="#00FF00", alpha=0.7, pad=1, edgecolor="none"),
        )
    axes[0].axis("off")

    axes[1].imshow(pred_img_rgb)
    axes[1].set_title(f"Prediction (conf={conf})\nDetected: {len(result.boxes)}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
