import os
import re

# needed to make pylidc work with python3.13
import numpy as np

if not hasattr(np, "int"):
    np.int = int
import pandas as pd

import configparser

if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

import pylidc as pl


def _clean_uid(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", uid)

# ChatGPT 5 assisted with troubleshooting this function
def export_data(
    annotations: list[pl.Annotation],
    out_dir: str = "npy_data",
    csv_name: str = "annotations.csv",
) -> None:
    """
    For each Annotation in `annotations`, iterate k in its bbox slice range,
    save the FULL 2D slice as .npy (once per (series_uid,k)), and write a row
    with bbox + image info to a DataFrame.

    """
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "slices")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    saved_slices: set[tuple[str, int]] = set()

    for ann in annotations:
        scan = ann.scan
        series_uid = scan.series_instance_uid
        patient_id = scan.patient_id
        reader_id = getattr(ann, "reader_id", None)

        s_i, s_j, s_k = ann.bbox()
        i0, i1 = int(s_i.start), int(s_i.stop)
        j0, j1 = int(s_j.start), int(s_j.stop)
        k0, k1 = int(s_k.start), int(s_k.stop)

        # for a given annotation, load all slices
        slices = scan.load_all_dicom_images(verbose=False)
        n = len(slices)
        k0 = max(0, min(k0, n - 1))
        k1 = max(k0 + 1, min(k1, n))

        # any slice from sample will have same attributes
        sample = slices[k0]
        sample = (
            sample.pixel_array if hasattr(sample, "pixel_array") else np.asarray(sample)
        )
        height, width = int(sample.shape[0]), int(sample.shape[1])

        # clamp bbox to image bounds
        i0c, i1c = max(0, i0), min(height, i1)
        j0c, j1c = max(0, j0), min(width, j1)

        # iterate the slices that intersect annotation
        for k in range(k0, k1):
            im = slices[k]

            im = im.pixel_array if hasattr(im, "pixel_array") else np.asarray(im)

            # save full slice once per (series_uid, k)
            key = (series_uid, k)
            uid_safe = _clean_uid(series_uid)
            img_path = os.path.join(img_dir, f"{uid_safe}_k{k:04d}.npy")
            if key not in saved_slices:
                np.save(img_path, im.astype(np.int16, copy=False))
                saved_slices.add(key)

            # record one row per (annotation, slice)
            rows.append(
                {
                    "series_instance_uid": series_uid,
                    "patient_id": patient_id,
                    "annotation_id": int(ann.id),
                    "reader_id": reader_id,
                    "slice_index": int(k),
                    "image_path": img_path,  # full 2D slice for PyTorch
                    # bbox in pixel coords (row/col)
                    "bbox_i0": int(i0c),
                    "bbox_i1": int(i1c),
                    "bbox_j0": int(j0c),
                    "bbox_j1": int(j1c),
                    "bbox_h": int(max(0, i1c - i0c)),
                    "bbox_w": int(max(0, j1c - j0c)),
                    "height": height,
                    "width": width,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows -> {csv_path}")
