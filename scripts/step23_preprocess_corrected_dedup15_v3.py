import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from scipy.ndimage import zoom


ROOT = Path("/home/ubuntu/AAA_CTA_Project")

DATASET_CSV = (
    ROOT /
    "2_dataset" /
    "AAA_finetune_dataset.csv"
)

CORR_CSV = (
    ROOT /
    "2_dataset" /
    "label_z_orientation_correction_audit.csv"
)

INDEPENDENT_CSV = (
    ROOT /
    "2_dataset" /
    "independent_cases_15.csv"
)

OUT_IMG = (
    ROOT /
    "3_preprocessed_v3_corrected" /
    "images"
)

OUT_LAB = (
    ROOT /
    "3_preprocessed_v3_corrected" /
    "labels"
)

OUT_IMG.mkdir(
    parents=True,
    exist_ok=True
)

OUT_LAB.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_SPACING = (
    1.5,
    1.5,
    1.5
)

MARGIN = 24


# ============================================================
# Load manifests
# ============================================================

df = pd.read_csv(
    DATASET_CSV
)

corr = pd.read_csv(
    CORR_CSV
)

independent = pd.read_csv(
    INDEPENDENT_CSV
)

df["case_id"] = df["case_id"].astype(str)
corr["case_id"] = corr["case_id"].astype(str)
independent["case_id"] = independent["case_id"].astype(str)

keep_cases = set(
    independent["case_id"]
)

df = df[
    df["case_id"].isin(
        keep_cases
    )
].copy()

df = df.merge(
    corr[
        [
            "case_id",
            "action",
            "corrected_label"
        ]
    ],
    on="case_id",
    how="inner"
)

df = (
    df
    .drop_duplicates(
        subset=["case_id"]
    )
    .sort_values(
        "case_id"
    )
    .reset_index(
        drop=True
    )
)


print("=" * 90)
print("PREPROCESSING V3 - CORRECTED LABELS")
print("=" * 90)
print("Independent cases:", len(df))
print("Target spacing   :", TARGET_SPACING)
print("ROI margin       :", MARGIN)
print("Pipeline         : corrected label -> resample -> HU clip -> crop -> ROI Z-score")
print("=" * 90)


records = []


for index, row in df.iterrows():

    case_id = str(
        row["case_id"]
    )

    print(
        f"\n[{index + 1:02d}/{len(df):02d}] "
        f"{case_id}"
    )

    # ========================================================
    # Load raw CTA + corrected label
    # ========================================================

    image_nii = nib.load(
        str(
            ROOT /
            row["image"]
        )
    )

    label_nii = nib.load(
        str(
            ROOT /
            row["corrected_label"]
        )
    )

    image = (
        image_nii
        .get_fdata()
        .astype(
            np.float32
        )
    )

    label = (
        label_nii
        .get_fdata()
        > 0
    ).astype(
        np.uint8
    )


    # ========================================================
    # Physical-space safety check
    # ========================================================

    if image.shape != label.shape:
        raise RuntimeError(
            f"{case_id}: image/label shape mismatch"
        )

    if not np.allclose(
        image_nii.affine,
        label_nii.affine,
        atol=1e-4
    ):
        raise RuntimeError(
            f"{case_id}: image/label affine mismatch"
        )


    original_shape = image.shape

    original_spacing = (
        image_nii
        .header
        .get_zooms()[:3]
    )


    # ========================================================
    # Resample to 1.5-mm isotropic
    # ========================================================

    factors = [
        original_spacing[i]
        /
        TARGET_SPACING[i]
        for i in range(3)
    ]

    image_rs = zoom(
        image,
        factors,
        order=1
    )

    label_rs = zoom(
        label,
        factors,
        order=0
    )

    label_rs = (
        label_rs > 0
    ).astype(
        np.uint8
    )


    # ========================================================
    # HU clipping
    # ========================================================

    image_rs = np.clip(
        image_rs,
        -200.0,
        400.0
    )


    # ========================================================
    # ROI from corrected label
    # ========================================================

    coordinates = np.argwhere(
        label_rs > 0
    )

    if coordinates.size == 0:
        raise RuntimeError(
            f"{case_id}: empty corrected label"
        )

    zmin, ymin, xmin = (
        coordinates.min(
            axis=0
        )
    )

    zmax, ymax, xmax = (
        coordinates.max(
            axis=0
        )
    )

    zmin = max(
        int(zmin) - MARGIN,
        0
    )

    ymin = max(
        int(ymin) - MARGIN,
        0
    )

    xmin = max(
        int(xmin) - MARGIN,
        0
    )

    zmax = min(
        int(zmax) + MARGIN,
        image_rs.shape[0] - 1
    )

    ymax = min(
        int(ymax) + MARGIN,
        image_rs.shape[1] - 1
    )

    xmax = min(
        int(xmax) + MARGIN,
        image_rs.shape[2] - 1
    )


    # ========================================================
    # Crop
    # ========================================================

    image_crop = image_rs[
        zmin:zmax + 1,
        ymin:ymax + 1,
        xmin:xmax + 1
    ].copy()

    label_crop = label_rs[
        zmin:zmax + 1,
        ymin:ymax + 1,
        xmin:xmax + 1
    ].copy()


    # ========================================================
    # ROI-level Z-score
    # ========================================================

    crop_mean = float(
        image_crop.mean()
    )

    crop_std = float(
        image_crop.std()
    )

    image_crop = (
        image_crop -
        crop_mean
    ) / (
        crop_std +
        1e-8
    )

    image_crop = (
        image_crop
        .astype(
            np.float32
        )
    )

    label_crop = (
        label_crop
        .astype(
            np.uint8
        )
    )


    # ========================================================
    # QC
    # ========================================================

    if (
        image_crop.shape
        !=
        label_crop.shape
    ):
        raise RuntimeError(
            f"{case_id}: crop shape mismatch"
        )

    if not np.isfinite(
        image_crop
    ).all():
        raise RuntimeError(
            f"{case_id}: non-finite values"
        )

    foreground = (
        label_crop > 0
    )

    if foreground.sum() == 0:
        raise RuntimeError(
            f"{case_id}: empty crop label"
        )


    # ========================================================
    # Save
    # ========================================================

    image_out = (
        OUT_IMG /
        f"{case_id}_crop_img.npy"
    )

    label_out = (
        OUT_LAB /
        f"{case_id}_crop_lab.npy"
    )

    np.save(
        image_out,
        image_crop
    )

    np.save(
        label_out,
        label_crop
    )


    # ========================================================
    # Statistics
    # ========================================================

    inside = image_crop[
        foreground
    ]

    outside = image_crop[
        ~foreground
    ]

    records.append({
        "case_id":
            case_id,

        "dataset":
            row.get(
                "dataset",
                ""
            ),

        "class":
            row.get(
                "class",
                ""
            ),

        "orientation_action":
            row["action"],

        "orig_shape":
            str(
                original_shape
            ),

        "orig_spacing":
            str(
                tuple(
                    float(x)
                    for x in original_spacing
                )
            ),

        "resampled_shape":
            str(
                image_rs.shape
            ),

        "crop_shape":
            str(
                image_crop.shape
            ),

        "post_norm_mean":
            float(
                image_crop.mean()
            ),

        "post_norm_std":
            float(
                image_crop.std()
            ),

        "inside_mean":
            float(
                inside.mean()
            ),

        "inside_p50":
            float(
                np.median(
                    inside
                )
            ),

        "outside_mean":
            float(
                outside.mean()
            ),

        "gt_voxels":
            int(
                foreground.sum()
            ),

        "gt_fraction":
            float(
                foreground.mean()
            ),

        "image_npy":
            str(
                image_out.relative_to(
                    ROOT
                )
            ),

        "label_npy":
            str(
                label_out.relative_to(
                    ROOT
                )
            )
    })


    print(
        f"  action    : {row['action']}"
    )

    print(
        "  crop      :",
        image_crop.shape
    )

    print(
        "  mean/std  :",
        f"{image_crop.mean():.5f}",
        f"{image_crop.std():.5f}"
    )

    print(
        "  GT voxels :",
        int(
            foreground.sum()
        )
    )


# ============================================================
# Save manifest
# ============================================================

manifest = pd.DataFrame(
    records
)

manifest_file = (
    ROOT /
    "2_dataset" /
    "finetune_preprocessed_manifest_v3_corrected_dedup15.csv"
)

manifest.to_csv(
    manifest_file,
    index=False
)


print("\n" + "=" * 90)
print("PREPROCESSING V3 FINISHED")
print("=" * 90)

print(
    "Processed:",
    len(manifest)
)

print(
    "Unique cases:",
    manifest[
        "case_id"
    ].nunique()
)

print(
    "Images:",
    len(
        list(
            OUT_IMG.glob(
                "*_crop_img.npy"
            )
        )
    )
)

print(
    "Labels:",
    len(
        list(
            OUT_LAB.glob(
                "*_crop_lab.npy"
            )
        )
    )
)

print("\nOrientation actions:")

print(
    manifest[
        "orientation_action"
    ]
    .value_counts()
    .to_string()
)

print("\nVessel statistics:")

print(
    manifest[
        [
            "inside_mean",
            "inside_p50",
            "gt_fraction"
        ]
    ]
    .describe()
    .round(4)
    .to_string()
)

print("\nManifest:")
print(manifest_file)

print("=" * 90)
