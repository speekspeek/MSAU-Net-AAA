import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

CSV = ROOT / "2_dataset/AAA_finetune_dataset.csv"

OUT_ROOT = ROOT / "0_raw_data_corrected"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df["case_id"] = df["case_id"].astype(str)

records = []

print("=" * 100)
print("RAW LABEL Z-ORIENTATION CORRECTION")
print("=" * 100)

for _, r in df.iterrows():

    case = r["case_id"]
    cls = r["class"]
    dataset = r["dataset"]

    img_path = ROOT / r["image"]
    lab_path = ROOT / r["label"]

    img_nii = nib.load(str(img_path))
    lab_nii = nib.load(str(lab_path))

    img = img_nii.get_fdata()
    lab = lab_nii.get_fdata()

    if img.shape != lab.shape:
        raise RuntimeError(
            f"{case}: image/label shape mismatch "
            f"{img.shape} vs {lab.shape}"
        )

    Z = img.shape[2]

    dz = float(
        img_nii.header.get_zooms()[2]
    )

    delta_z = float(
        lab_nii.affine[2, 3]
        -
        img_nii.affine[2, 3]
    )

    expected_reverse = -(Z - 1) * dz

    reverse_signature = np.isclose(
        delta_z,
        expected_reverse,
        atol=max(0.01, dz * 0.05)
    )

    # --------------------------------------------------------
    # Correct voxel orientation
    # --------------------------------------------------------

    if reverse_signature:

        corrected = np.flip(
            lab,
            axis=2
        ).copy()

        action = "Z_FLIPPED"

    else:

        corrected = lab.copy()

        action = "DIRECT"

    corrected = (
        corrected > 0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # IMPORTANT:
    # Corrected segmentation is now on the IMAGE voxel grid.
    # Therefore use IMAGE affine/header spatial definition.
    # --------------------------------------------------------

    header = img_nii.header.copy()
    header.set_data_dtype(np.uint8)

    corrected_nii = nib.Nifti1Image(
        corrected,
        img_nii.affine,
        header=header
    )

    # explicitly make qform/sform consistent with image
    corrected_nii.set_qform(
        img_nii.affine,
        code=1
    )

    corrected_nii.set_sform(
        img_nii.affine,
        code=1
    )

    out_dir = (
        OUT_ROOT /
        str(dataset) /
        case /
        "segmentation"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path = (
        out_dir /
        f"{case}_segmentation_corrected.nii.gz"
    )

    nib.save(
        corrected_nii,
        str(out_path)
    )

    # --------------------------------------------------------
    # Reload verification
    # --------------------------------------------------------

    check = nib.load(
        str(out_path)
    )

    affine_ok = np.allclose(
        check.affine,
        img_nii.affine,
        atol=1e-5
    )

    shape_ok = (
        check.shape ==
        img_nii.shape
    )

    fg_before = int(
        np.sum(lab > 0)
    )

    fg_after = int(
        np.sum(corrected > 0)
    )

    if fg_before != fg_after:
        raise RuntimeError(
            f"{case}: foreground voxel count changed."
        )

    if not affine_ok:
        raise RuntimeError(
            f"{case}: corrected affine mismatch."
        )

    if not shape_ok:
        raise RuntimeError(
            f"{case}: corrected shape mismatch."
        )

    records.append({
        "case_id": case,
        "dataset": dataset,
        "class": cls,
        "action": action,
        "reverse_signature":
            bool(reverse_signature),
        "delta_z_mm":
            delta_z,
        "expected_reverse_delta_mm":
            expected_reverse,
        "foreground_voxels":
            fg_after,
        "shape_ok":
            shape_ok,
        "affine_ok":
            affine_ok,
        "corrected_label":
            str(
                out_path.relative_to(ROOT)
            )
    })

    print(
        f"{case:10s} "
        f"{action:10s} "
        f"FG={fg_after:10d} "
        f"shape={shape_ok} "
        f"affine={affine_ok}"
    )


audit = pd.DataFrame(records)

audit_path = (
    ROOT /
    "2_dataset/label_z_orientation_correction_audit.csv"
)

audit.to_csv(
    audit_path,
    index=False
)

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    audit["action"]
    .value_counts()
    .to_string()
)

print("\nCorrected labels:")
print(
    audit["affine_ok"]
    .value_counts()
    .to_string()
)

print("\nAudit:")
print(audit_path)

print("\nCorrected root:")
print(OUT_ROOT)

print("=" * 100)
