import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

DATASET_CSV = ROOT / "2_dataset/AAA_finetune_dataset.csv"
CORR_CSV = ROOT / "2_dataset/label_z_orientation_correction_audit.csv"

df = pd.read_csv(DATASET_CSV)
corr = pd.read_csv(CORR_CSV)

df["case_id"] = df["case_id"].astype(str)
corr["case_id"] = corr["case_id"].astype(str)

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

rows = []

print("=" * 125)
print("CORRECTED 20-CASE FINAL PHYSICAL / VOXEL AUDIT")
print("=" * 125)

for _, r in df.iterrows():

    case = r["case_id"]

    img_nii = nib.load(
        str(ROOT / r["image"])
    )

    lab_nii = nib.load(
        str(ROOT / r["corrected_label"])
    )

    img = img_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata() > 0

    # -------------------------------------------------------
    # Basic geometry
    # -------------------------------------------------------

    shape_ok = img.shape == lab.shape

    spacing_img = np.array(
        img_nii.header.get_zooms()[:3],
        dtype=float
    )

    spacing_lab = np.array(
        lab_nii.header.get_zooms()[:3],
        dtype=float
    )

    spacing_ok = np.allclose(
        spacing_img,
        spacing_lab,
        atol=1e-5
    )

    affine_diff = float(
        np.max(
            np.abs(
                img_nii.affine -
                lab_nii.affine
            )
        )
    )

    affine_ok = affine_diff < 0.01

    # -------------------------------------------------------
    # Label
    # -------------------------------------------------------

    fg = int(lab.sum())

    label_ok = fg > 0

    # -------------------------------------------------------
    # CTA intensity inside label
    # Only diagnostic, not correction criterion
    # -------------------------------------------------------

    img_clip = np.clip(
        img,
        -200,
        400
    )

    inside = img_clip[lab]

    inside_mean = float(
        inside.mean()
    )

    inside_median = float(
        np.median(inside)
    )

    inside_p10 = float(
        np.percentile(
            inside,
            10
        )
    )

    inside_p90 = float(
        np.percentile(
            inside,
            90
        )
    )

    # -------------------------------------------------------
    # Z extent
    # -------------------------------------------------------

    z_has = np.where(
        lab.any(axis=(0, 1))
    )[0]

    z0 = int(z_has.min())
    z1 = int(z_has.max())

    z_center = float(
        (z0 + z1) / 2
    )

    z_center_fraction = (
        z_center /
        (img.shape[2] - 1)
    )

    overall = (
        shape_ok
        and spacing_ok
        and affine_ok
        and label_ok
    )

    rows.append({
        "case_id": case,
        "class": r["class"],
        "action": r["action"],

        "shape_ok": shape_ok,
        "spacing_ok": spacing_ok,
        "affine_ok": affine_ok,
        "affine_max_diff": affine_diff,

        "fg_voxels": fg,

        "z0": z0,
        "z1": z1,
        "z_center_fraction":
            z_center_fraction,

        "inside_mean_HU":
            inside_mean,

        "inside_median_HU":
            inside_median,

        "inside_p10_HU":
            inside_p10,

        "inside_p90_HU":
            inside_p90,

        "overall":
            overall
    })

    print(
        f"{case:10s} "
        f"{r['action']:10s} "
        f"shape={str(shape_ok):5s} "
        f"spacing={str(spacing_ok):5s} "
        f"affine={str(affine_ok):5s} "
        f"FG={fg:9d} "
        f"HU50={inside_median:7.1f} "
        f"Z={z0:4d}-{z1:4d} "
        f"{'PASS' if overall else 'FAIL'}"
    )


out = pd.DataFrame(rows)

save = (
    ROOT /
    "4_results/corrected20_final_physical_voxel_audit.csv"
)

out.to_csv(
    save,
    index=False
)

print("\n" + "=" * 125)
print("FINAL SUMMARY")
print("=" * 125)

print(
    "Cases:",
    len(out)
)

print(
    "Overall PASS:",
    int(out["overall"].sum()),
    "/",
    len(out)
)

print(
    "Shape PASS:",
    int(out["shape_ok"].sum()),
    "/",
    len(out)
)

print(
    "Spacing PASS:",
    int(out["spacing_ok"].sum()),
    "/",
    len(out)
)

print(
    "Affine PASS:",
    int(out["affine_ok"].sum()),
    "/",
    len(out)
)

print("\nAction:")
print(
    out["action"]
    .value_counts()
    .to_string()
)

print("\nInside-label HU statistics:")
print(
    out[
        [
            "inside_mean_HU",
            "inside_median_HU",
            "inside_p10_HU",
            "inside_p90_HU"
        ]
    ]
    .describe()
    .round(2)
    .to_string()
)

print("\nSaved:")
print(save)

if not out["overall"].all():
    raise RuntimeError(
        "Corrected dataset audit FAILED."
    )

print("\n" + "=" * 125)
print("CORRECTED DATASET AUDIT PASSED")
print("=" * 125)
