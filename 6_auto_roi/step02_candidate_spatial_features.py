#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

from scipy.ndimage import zoom, label, binary_opening

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

MANIFEST = ROOT / (
    "2_dataset/"
    "finetune_preprocessed_manifest_v3_corrected_dedup15.csv"
)

OUTDIR = ROOT / "9_results/auto_roi_candidates_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_SPACING = (1.5, 1.5, 1.5)
HU_MIN = 150.0
HU_MAX = 400.0

STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)

df = pd.read_csv(MANIFEST)
records = []

print("=" * 110)
print("AAA — IMAGE-ONLY CANDIDATE SPATIAL FEATURES")
print("=" * 110)

for case_id in df["case_id"].astype(str):

    group = "Anev" if case_id.startswith("Anev") else "Nat"

    image_file = ROOT / (
        f"0_raw_data/{group}/{case_id}/nifti/"
        f"{case_id}_image.nii.gz"
    )

    nii = nib.load(str(image_file))
    image = nii.get_fdata(dtype=np.float32)

    spacing = nii.header.get_zooms()[:3]
    factors = tuple(float(s) / 1.5 for s in spacing)

    image_rs = zoom(image, factors, order=1)

    candidate = (
        (image_rs >= HU_MIN) &
        (image_rs <= HU_MAX)
    )

    candidate = binary_opening(
        candidate,
        structure=STRUCTURE,
        iterations=1
    )

    components, num = label(
        candidate,
        structure=STRUCTURE
    )

    sizes = np.bincount(components.ravel())[1:]
    order = np.argsort(sizes)[::-1][:10]

    shape = np.array(candidate.shape, dtype=float)

    print(f"\n{case_id} | components={num}")

    for rank, idx in enumerate(order, 1):

        component_id = int(idx + 1)
        component_size = int(sizes[idx])

        coords = np.argwhere(
            components == component_id
        )

        low = coords.min(axis=0)
        high = coords.max(axis=0)

        center = coords.mean(axis=0)

        normalized_center = center / shape

        extent = high - low + 1
        normalized_extent = extent / shape

        record = {
            "case_id": case_id,
            "rank": rank,
            "component_id": component_id,
            "voxels": component_size,
            "center_axis0": normalized_center[0],
            "center_axis1": normalized_center[1],
            "center_axis2": normalized_center[2],
            "extent_axis0": normalized_extent[0],
            "extent_axis1": normalized_extent[1],
            "extent_axis2": normalized_extent[2],
            "bbox_min": str(low.tolist()),
            "bbox_max": str(high.tolist())
        }

        records.append(record)

        print(
            f"  #{rank:02d} "
            f"size={component_size:8d} "
            f"center=({normalized_center[0]:.3f}, "
            f"{normalized_center[1]:.3f}, "
            f"{normalized_center[2]:.3f}) "
            f"extent=({normalized_extent[0]:.3f}, "
            f"{normalized_extent[1]:.3f}, "
            f"{normalized_extent[2]:.3f})"
        )

    del image, image_rs, candidate, components

result = pd.DataFrame(records)

result.to_csv(
    OUTDIR / "candidate_spatial_features.csv",
    index=False
)

print("\n" + "=" * 110)
print("SUMMARY — TOP-RANKED COMPONENTS")
print("=" * 110)

top1 = result[result["rank"] == 1]

print(
    top1[
        [
            "voxels",
            "center_axis0",
            "center_axis1",
            "center_axis2",
            "extent_axis2"
        ]
    ].describe().round(4).to_string()
)

print("\nSaved:", OUTDIR / "candidate_spatial_features.csv")
print("=" * 110)
