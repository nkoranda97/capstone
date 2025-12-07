import os
import random

import numpy as np

if not hasattr(np, "int"):
    np.int = int
import pandas as pd

import configparser

if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

import pylidc as pl
from PIL import Image

"""
Google Gemini 3 used to assist with debugging, RGB logic, and Union logic.
Microsoft copliot used to refactor code into smaller, modular functions
"""


def _apply_window(img_hu, window_range):
    """Clamps HU values and normalizes to 0-255 uint8."""
    min_hu, max_hu = window_range
    img_clamped = np.clip(img_hu, min_hu, max_hu)
    # Normalize to 0-1 then scale to 255
    img_norm = (img_clamped - min_hu) / (max_hu - min_hu)
    return (img_norm * 255).astype(np.uint8)


def _setup_directories(root_dir: str) -> dict:
    dirs = {
        "train_img": os.path.join(root_dir, "images/train"),
        "val_img": os.path.join(root_dir, "images/val"),
        "train_lbl": os.path.join(root_dir, "labels/train"),
        "val_lbl": os.path.join(root_dir, "labels/val"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _split_patients(scans: list[pl.Scan], train_frac: float) -> set:
    patient_ids = sorted({scan.patient_id for scan in scans})
    random.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * train_frac)
    train_patients = set(patient_ids[:split_idx])
    return train_patients


def _process_nodules(
    scan: pl.Scan, num_slices: int, height: int, width: int
) -> tuple[set, dict]:
    pos_k_indices = set()
    bboxes_by_k = {}

    nodules = scan.cluster_annotations(verbose=False)

    for nod in nodules:
        if len(nod) < 1:
            continue

        # collect all annotations for a nod
        i0_list, i1_list, j0_list, j1_list, k_list = [], [], [], [], []

        for ann in nod:
            s_i, s_j, _ = ann.bbox()
            i0_list.append(s_i.start)
            i1_list.append(s_i.stop)
            j0_list.append(s_j.start)
            j1_list.append(s_j.stop)
            k_list.append(ann.centroid[2])

        # create union box of annotations for max area
        i0, i1 = min(i0_list), max(i1_list)
        j0, j1 = min(j0_list), max(j1_list)

        # Determine Center Z (k)
        k_center = int(round(np.mean(k_list)))

        # make sure k_center is valid
        if 0 <= k_center < num_slices:
            pos_k_indices.add(k_center)

            # normalize xywh for yolo
            box_h = i1 - i0
            box_w = j1 - j0
            y_c = i0 + (box_h / 2.0)
            x_c = j0 + (box_w / 2.0)

            if k_center not in bboxes_by_k:
                bboxes_by_k[k_center] = []

            bboxes_by_k[k_center].append(
                f"0 {x_c / width:.6f} {y_c / height:.6f} {box_w / width:.6f} {box_h / height:.6f}"
            )

    return pos_k_indices, bboxes_by_k


def _select_negative_slices(
    pos_k_indices: set, num_slices: int, neg_pos_ratio: float
) -> list:
    # Negative Slices
    all_indices = set(range(num_slices))

    # make sure negative slices are far away from positive slides
    buffer_indices = set()
    for k in pos_k_indices:
        buffer_indices.update(range(k - 2, k + 3))

    candidate_negatives = list(all_indices - buffer_indices)

    num_pos = len(pos_k_indices)
    num_neg = int(num_pos * neg_pos_ratio)

    chosen_negatives = []
    if candidate_negatives and num_neg > 0:
        n = min(len(candidate_negatives), num_neg)
        chosen_negatives = random.sample(candidate_negatives, n)

    return chosen_negatives


def _export_slice(
    vol,
    k: int,
    num_slices: int,
    window_range: tuple[int, int],
    dirs: dict,
    split: str,
    clean_uid: str,
    bboxes_by_k: dict,
    pos_k_indices: set,
    series_uid: str,
    metadata: list,
) -> None:
    # 1. Prepare 3 Channels (k-1, k, k+1)
    k_prev = max(0, k - 1)
    k_curr = k
    k_next = min(num_slices - 1, k + 1)

    # Extract Hounsfield Units
    slice_r = vol[:, :, k_prev]
    slice_g = vol[:, :, k_curr]
    slice_b = vol[:, :, k_next]

    # Window and stack
    img_r = _apply_window(slice_r, window_range)
    img_g = _apply_window(slice_g, window_range)
    img_b = _apply_window(slice_b, window_range)

    # Stack to (H, W, 3) -> RGB
    img_rgb = np.dstack((img_r, img_g, img_b))

    # 2. Save Image
    fname = f"{clean_uid}_k{k:04d}.jpg"
    save_dir_img = dirs[f"{split}_img"]
    save_dir_lbl = dirs[f"{split}_lbl"]

    Image.fromarray(img_rgb).save(os.path.join(save_dir_img, fname), quality=95)

    # 3. Save Label (if positive)
    txt_path = os.path.join(save_dir_lbl, fname.replace(".jpg", ".txt"))
    if k in bboxes_by_k:
        with open(txt_path, "w") as f:
            f.write("\n".join(bboxes_by_k[k]))
    else:
        # Empty file for negative
        open(txt_path, "w").close()

    metadata.append(
        {
            "uid": series_uid,
            "split": split,
            "k": k,
            "is_nodule": 1 if k in pos_k_indices else 0,
        }
    )


def _process_scan(
    scan: pl.Scan,
    train_patients: set,
    dirs: dict,
    window_range: tuple[int, int],
    neg_pos_ratio: float,
    metadata: list,
) -> None:
    pid = scan.patient_id
    series_uid = scan.series_instance_uid
    split = "train" if pid in train_patients else "val"

    vol = scan.to_volume()
    num_slices = vol.shape[2]
    height, width = vol.shape[0], vol.shape[1]

    # Track which slices have nodules
    pos_k_indices, bboxes_by_k = _process_nodules(scan, num_slices, height, width)

    chosen_negatives = _select_negative_slices(pos_k_indices, num_slices, neg_pos_ratio)

    slices_to_export = list(pos_k_indices) + chosen_negatives

    # export images to JPG
    clean_uid = series_uid.replace(".", "")[-10:]  # Shorten UID for filename

    for k in slices_to_export:
        _export_slice(
            vol,
            k,
            num_slices,
            window_range,
            dirs,
            split,
            clean_uid,
            bboxes_by_k,
            pos_k_indices,
            series_uid,
            metadata,
        )


def _write_yaml(root_dir: str) -> None:
    yaml_path = os.path.join(root_dir, "ct.yaml")
    abs_path = os.path.abspath(root_dir)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names: {0: nodule}\n")


def build_dataset(
    scans: list[pl.Scan],
    root_dir: str = "data/ct_yolo",
    window_range: tuple[int, int] = (-1350, 150),
    train_frac: float = 0.8,
    neg_pos_ratio: float = 0.25,
    seed: int = 15,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # 1. Setup Directories
    dirs = _setup_directories(root_dir)

    # 2. Patient-Level Split
    train_patients = _split_patients(scans, train_frac)

    metadata = []

    # 3. Main Loop (Process one scan at a time)
    for scan in scans:
        _process_scan(scan, train_patients, dirs, window_range, neg_pos_ratio, metadata)

    _write_yaml(root_dir)

    # 4. Save meta data
    pd.DataFrame(metadata).to_csv(os.path.join(root_dir, "metadata.csv"), index=False)
