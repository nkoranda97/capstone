import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pylidc as pl
import configparser
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser
import numpy as np
if not hasattr(np, "int"):
    np.int = int
from skimage.feature import graycomatrix, graycoprops


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

def summarize_bboxes(
    csv_path: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["bbox_x_center"] = (df["bbox_j0"] + df["bbox_j1"]) / 2.0
    df["bbox_y_center"] = (df["bbox_i0"] + df["bbox_i1"]) / 2.0

    df["nx_center"] = df["bbox_x_center"] / df["width"]
    df["ny_center"] = df["bbox_y_center"] / df["height"]

    df["bbox_area_px"] = df["bbox_w"] * df["bbox_h"]
    df["rel_area"]     = df["bbox_area_px"] / (df["width"] * df["height"])

    df["aspect_ratio_wh"] = df["bbox_w"] / df["bbox_h"]

    left   = df["bbox_j0"]
    right  = df["width"]  - df["bbox_j1"]
    top    = df["bbox_i0"]
    bottom = df["height"] - df["bbox_i1"]
    df["edge_dist_px"]  = pd.concat([left,right,top,bottom], axis=1).min(axis=1)
    df["edge_dist_rel"] = df["edge_dist_px"] / df[["width","height"]].max(axis=1)

    df["r_center_norm"] = np.sqrt((df["nx_center"] - 0.5)**2 + (df["ny_center"] - 0.5)**2)

    boxes_per_slice = df.groupby("image_path", as_index=False).size().rename(columns={"size":"boxes_per_slice"})
    df = df.merge(boxes_per_slice, on="image_path", how="left")

    return df

def extract_patient_data(annotations: list[pl.Annotation]) -> pd.DataFrame:
    rows = []
    seen_pids = set()

    for ann in annotations:
        scan = ann.scan
        pid = scan.patient_id
        if pid in seen_pids:
            continue

        ds = scan.load_all_dicom_images(verbose=False)[0]

        sex = getattr(ds, "PatientSex", None)
        sex = np.nan if not sex else sex

        age = getattr(ds, "PatientAge", None)
        age = int(age.replace("Y", "")) if age else np.nan

        manufacturer = getattr(ds, "Manufacturer", None)

        rows.append({
            "patient_id": pid,
            "sex": sex,
            "age": age,
            "manufacturer": manufacturer
        })
        seen_pids.add(pid)

    return pd.DataFrame(rows)

def compute_brightness_contrast(csv_path: str, sample: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    paths = df["image_path"].drop_duplicates()
    if sample is not None and sample < len(paths):
        paths = paths.sample(sample, random_state=0)
    paths = paths.reset_index(drop=True)

    rows = []
    for p in paths:
        img = np.load(p).astype(np.float32)
        rows.append({
            "mean": float(np.nanmean(img)),
            "std": float(np.nanstd(img)),
        })


    return pd.DataFrame(rows)


def compute_glcm_metrics(csv_path: str, sample: int = 500, levels: int = 32):
    df = pd.read_csv(csv_path)
    paths = df["image_path"].drop_duplicates()
    if sample is not None and sample < len(paths):
        paths = paths.sample(sample, random_state=0)

    rows = []
    for p in paths:
        img = np.load(p).astype(np.float32)
        img = ((img - img.min()) / (img.max() - img.min()) * (levels - 1)).astype(np.uint8)

        glcm = graycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=levels, symmetric=True, normed=True)
        props = {prop: graycoprops(glcm, prop).mean() for prop in
                 ["homogeneity", "energy", "correlation"]}
        props["image_path"] = p
        rows.append(props)

    return pd.DataFrame(rows)