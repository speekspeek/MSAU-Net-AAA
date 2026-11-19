#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from scipy.ndimage import (
    zoom,
    label,
    binary_opening
)

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

print("=" * 100)
print("AAA — IMAGE-ONLY HU CANDIDATE EXTRACTION")
print("=" * 100)

for case_id in df["case_id"].astype(str):

    group = (
        "Anev"
        if case_id.startswith("Anev")
        else "Nat"
    )

    image_path = ROOT / (
        f"0_raw_data/{group}/{case_id}/nifti/"
        f"{case_id}_image.nii.gz"
    )

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    nii = nib.load(str(image_path))

    image = nii.get_fdata(dtype=np.float32)

    spacing = nii.header.get_zooms()[:3]

    factors = tuple(
        float(spacing[i]) / TARGET_SPACING[i]
        for i in range(3)
    )

    image_rs = zoom(
        image,
        factors,
        order=1
    )

    # HU-based candidate extraction
    candidate = (
        (image_rs >= HU_MIN)
        &
        (image_rs <= HU_MAX)
    )

    # Remove small isolated structures
    candidate_clean = binary_opening(
        candidate,
        structure=STRUCTURE,
        iterations=1
    )

    components, num = label(
        candidate_clean,
        structure=STRUCTURE
    )

    sizes = np.bincount(
        components.ravel()
    )[1:]

    largest_sizes = (
        np.sort(sizes)[::-1][:5]
        if len(sizes)
        else np.array([], dtype=np.int64)
    )

    record = {
        "case_id": case_id,
        "resampled_shape": str(image_rs.shape),
        "candidate_fraction": float(candidate.mean()),
        "clean_fraction": float(candidate_clean.mean()),
        "component_count": int(num),
        "largest_component_voxels": (
            int(largest_sizes[0])
            if len(largest_sizes)
            else 0
        ),
        "top5_component_sizes": str(
            largest_sizes.tolist()
        )
    }

    records.append(record)

    print(
        f"{case_id:10s} "
        f"Candidate={record['candidate_fraction']:.4f} "
        f"Clean={record['clean_fraction']:.4f} "
        f"Components={num:5d} "
        f"Largest={record['largest_component_voxels']:8d}"
    )

    del image, image_rs, candidate
    del candidate_clean, components

result = pd.DataFrame(records)

result.to_csv(
    OUTDIR / "hu_candidate_summary.csv",
    index=False
)

print("\nSUMMARY")

print(
    result[
        [
            "candidate_fraction",
            "clean_fraction",
            "component_count",
            "largest_component_voxels"
        ]
    ].describe().round(4).to_string()
)

print("\nSaved:", OUTDIR / "hu_candidate_summary.csv")
print("=" * 100)
