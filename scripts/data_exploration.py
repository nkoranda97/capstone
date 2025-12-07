import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pylidc as pl
import configparser

if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

if not hasattr(np, "int"):
    np.int = int
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

"""
Gemini 5 used to help display image with bounding box and to create gather data function
"""


def show_random_image(
    root_dir: str,
    split: str = "train",
    class_names: list[str] | None = None,
    seed: int | None = None,
):
    """
    Display a random YOLO-formatted PNG image with its bounding boxes.

    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    img_dir = os.path.join(root_dir, "images", split)
    label_dir = os.path.join(root_dir, "labels", split)

    img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    img_name = random.choice(img_files)
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    img = np.array(Image.open(img_path).convert("L"))
    h, w = img.shape

    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                x0 = (xc - bw / 2) * w
                y0 = (yc - bh / 2) * h
                x1 = (xc + bw / 2) * w
                y1 = (yc + bh / 2) * h
                boxes.append((x0, y0, x1, y1, int(cls)))

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    colors = ["red", "blue", "yellow", "green", "purple", "orange", "cyan", "magenta"]

    for i, (x0, y0, x1, y1, cls) in enumerate(boxes):
        color = colors[i % len(colors)]
        ax.add_patch(
            Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2, edgecolor=color
            )
        )

    ax.axis("off")
    plt.tight_layout()
    plt.show()


def extract_patient_data(annotations: list[pl.Annotation]) -> pd.DataFrame:
    """
    Extracts patient metadata from a pl.Annotation object.
    This is wildly inefficent, as every picture is loaded into memory and never used.
    Could probably be improved.
    """
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

        rows.append(
            {"patient_id": pid, "sex": sex, "age": age, "manufacturer": manufacturer}
        )
        seen_pids.add(pid)

    return pd.DataFrame(rows)


class DatasetStats:
    def __init__(self, root_dir: str):
        """
        Initializes the dataset statistics object by parsing the directory
        structure and metadata. This performs the initial data gathering
        automatically upon instantiation.
        """
        self.root_dir = root_dir
        self.df = self._gather_data()

    def _gather_data(self) -> pd.DataFrame:
        """
        Parses the directory structure, metadata CSV, and YOLO label files
        to construct a single DataFrame containing all bounding box and image
        metadata required for analysis.
        """
        meta_path = os.path.join(self.root_dir, "metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Filter for positive nodules to get bounding boxes
        df_pos = df_meta[df_meta["is_nodule"] == 1].copy()

        data_records = []

        for _, row in df_pos.iterrows():
            clean_uid = row["uid"].replace(".", "")[-10:]
            base_fname = f"{clean_uid}_k{row['k']:04d}"

            split = row["split"]
            img_path = os.path.join(self.root_dir, "images", split, f"{base_fname}.jpg")
            txt_path = os.path.join(self.root_dir, "labels", split, f"{base_fname}.txt")

            # 2. Get Image Dimensions
            with Image.open(img_path) as img:
                width, height = img.size

            # 3. Read YOLO Label file
            with open(txt_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                nx_c, ny_c, nw, nh = map(float, parts[1:])

                # Convert normalized YOLO metrics to pixel coordinates
                w_px = nw * width
                h_px = nh * height
                x_c_px = nx_c * width
                y_c_px = ny_c * height

                # Calculate coordinates
                j0 = x_c_px - (w_px / 2.0)
                j1 = x_c_px + (w_px / 2.0)
                i0 = y_c_px - (h_px / 2.0)
                i1 = y_c_px + (h_px / 2.0)

                data_records.append(
                    {
                        "image_path": img_path,
                        "split": split,
                        "uid": row["uid"],
                        "k": row["k"],
                        "width": width,
                        "height": height,
                        "bbox_w": w_px,
                        "bbox_h": h_px,
                        "bbox_i0": i0,
                        "bbox_i1": i1,
                        "bbox_j0": j0,
                        "bbox_j1": j1,
                    }
                )

        return pd.DataFrame(data_records)

    def get_bbox_stats(self) -> pd.DataFrame:
        """
        Calculates geometric statistics for bounding boxes based on the raw data.
        """

        df = self.df

        df["bbox_x_center"] = (df["bbox_j0"] + df["bbox_j1"]) / 2.0
        df["bbox_y_center"] = (df["bbox_i0"] + df["bbox_i1"]) / 2.0

        df["nx_center"] = df["bbox_x_center"] / df["width"]
        df["ny_center"] = df["bbox_y_center"] / df["height"]

        df["bbox_area_px"] = df["bbox_w"] * df["bbox_h"]
        df["rel_area"] = df["bbox_area_px"] / (df["width"] * df["height"])

        df["aspect_ratio_wh"] = df["bbox_w"] / df["bbox_h"]

        left = df["bbox_j0"]
        right = df["width"] - df["bbox_j1"]
        top = df["bbox_i0"]
        bottom = df["height"] - df["bbox_i1"]

        df["edge_dist_px"] = pd.concat([left, right, top, bottom], axis=1).min(axis=1)
        df["edge_dist_rel"] = df["edge_dist_px"] / df[["width", "height"]].max(axis=1)

        df["r_center_norm"] = np.sqrt(
            (df["nx_center"] - 0.5) ** 2 + (df["ny_center"] - 0.5) ** 2
        )

        boxes_per_slice = (
            df.groupby("image_path", as_index=False)
            .size()
            .rename(columns={"size": "boxes_per_slice"})
        )
        df = df.merge(boxes_per_slice, on="image_path", how="left")

        return df

    def get_brightness_contrast(self, sample: int | None = None) -> pd.DataFrame:
        """
        Computes brightness (mean) and contrast (std) for the images in the dataset.
        """

        paths = self.df["image_path"].drop_duplicates()
        if sample is not None and sample < len(paths):
            paths = paths.sample(sample, random_state=0)

        paths = paths.reset_index(drop=True)

        rows = []
        for p in paths:
            with Image.open(p) as pil_img:
                img = np.array(pil_img.convert("L")).astype(np.float32)

            rows.append(
                {
                    "mean": float(np.nanmean(img)),
                    "std": float(np.nanstd(img)),
                    "image_path": p,
                }
            )

        return pd.DataFrame(rows)

    def get_glcm_metrics(self, sample: int = 500, levels: int = 32) -> pd.DataFrame:
        """
        Computes texture metrics (Homogeneity, Energy, Correlation) using GLCM.
        """

        paths = self.df["image_path"].drop_duplicates()
        if sample is not None and sample < len(paths):
            paths = paths.sample(sample, random_state=0)

        rows = []
        for p in paths:
            with Image.open(p) as pil_img:
                img = np.array(pil_img.convert("L")).astype(np.float32)

            # Normalize to 0-(levels-1)
            img_norm = (
                (img - img.min()) / (img.max() - img.min() + 1e-6) * (levels - 1)
            ).astype(np.uint8)

            glcm = graycomatrix(
                img_norm,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=levels,
                symmetric=True,
                normed=True,
            )
            props = {
                prop: graycoprops(glcm, prop).mean()
                for prop in ["homogeneity", "energy", "correlation"]
            }
            props["image_path"] = p
            rows.append(props)

        return pd.DataFrame(rows)
