import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_image(csv_path: str, window: tuple[int, int]=(-1000, 400)):
    """
    Input:
    csv_path: path to csv containing image data from data_processing
    window: image dimensions (int) in tuple

    Description:
    Shows random image with all its annotations from our data set made from data processing step
    """
    df = pd.read_csv(csv_path)

    # pick a random image
    img_path = random.choice(df["image_path"].unique())
    img = np.load(img_path)

    # all boxes for that image
    boxes = df[df["image_path"] == img_path][
        ["bbox_i0", "bbox_i1", "bbox_j0", "bbox_j1", "annotation_id"]
    ].to_numpy()

    colors = ["red", "blue", "yellow", "green", "purple", "orange", "cyan", "magenta"]

    # window and plot
    img = np.clip(img, window[0], window[1])
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    for idx, (i0, i1, j0, j1, _) in enumerate(boxes):
        color = colors[idx % len(colors)]
        ax.add_patch(
            Rectangle(
                (j0, i0), j1 - j0, i1 - i0, fill=False, linewidth=2, edgecolor=color
            )
        )

    ax.axis("off")
    plt.show()
