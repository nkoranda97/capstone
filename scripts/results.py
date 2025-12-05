import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO


def show_yolo_prediction(
    model_path: str,
    root_dir: str,
    split: str = "val",
    conf: float = 0.25,  # <--- Added argument here
    class_names: list[str] | None = None,
):
    """
    Display a random YOLO-formatted image from the NEW dataset structure
    (RGB .jpg images) with ground-truth and predicted boxes.

    Parameters
    ----------
    model_path : str
        Path to YOLO weights (.pt file).
    root_dir : str
        Root directory of the YOLO dataset.
    split : str
        'train' or 'val'.
    conf : float
        Confidence threshold for the prediction (default 0.25).
    class_names : list[str]
        Optional list of class names.
    """

    if class_names is None:
        class_names = ["nodule"]

    # 1. Setup paths
    img_dir = os.path.join(root_dir, "images", split)
    label_dir = os.path.join(root_dir, "labels", split)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
    if not img_files:
        print(f"No .jpg images found in {img_dir}")
        return

    # 2. Select Image
    img_name = random.choice(img_files)
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    # 3. Load Image (RGB)
    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]

    # 4. Load Ground Truth Boxes
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

    # 5. Run Prediction using the custom confidence
    model = YOLO(model_path)
    results = model(img_path, conf=conf, verbose=False)  # <--- Used argument here
    result = results[0]

    # Convert plot to RGB
    pred_img_bgr = result.plot(line_width=2)
    pred_img_rgb = pred_img_bgr[..., ::-1]

    # 6. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --- Left: Ground Truth ---
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

    # --- Right: Prediction ---
    axes[1].imshow(pred_img_rgb)
    # Update title to show which confidence was used
    axes[1].set_title(f"Prediction (conf={conf})\nDetected: {len(result.boxes)}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
