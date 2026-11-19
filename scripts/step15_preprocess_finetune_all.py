import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
DF = pd.read_csv(ROOT / "2_dataset" / "AAA_finetune_dataset.csv")

OUT_IMG = ROOT / "3_preprocessed" / "Anev" / "images"
OUT_LAB = ROOT / "3_preprocessed" / "Anev" / "labels"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LAB.mkdir(parents=True, exist_ok=True)

TARGET_SPACING = (1.5, 1.5, 1.5)
MARGIN = 24

records = []

for _, r in DF.iterrows():
    case_id = r["case_id"]

    img_nii = nib.load(str(ROOT / r["image"]))
    lab_nii = nib.load(str(ROOT / r["label"]))

    img = img_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata().astype(np.uint8)

    spacing = img_nii.header.get_zooms()[:3]
    factors = [spacing[i] / TARGET_SPACING[i] for i in range(3)]

    img_rs = zoom(img, factors, order=1)
    lab_rs = zoom(lab, factors, order=0)

    img_rs = np.clip(img_rs, -200, 400)
    img_rs = (img_rs - img_rs.mean()) / (img_rs.std() + 1e-8)

    coords = np.argwhere(lab_rs > 0)
    if coords.size == 0:
        print("Skip empty label:", case_id)
        continue

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    zmin = max(zmin - MARGIN, 0)
    ymin = max(ymin - MARGIN, 0)
    xmin = max(xmin - MARGIN, 0)
    zmax = min(zmax + MARGIN, img_rs.shape[0] - 1)
    ymax = min(ymax + MARGIN, img_rs.shape[1] - 1)
    xmax = min(xmax + MARGIN, img_rs.shape[2] - 1)

    img_crop = img_rs[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    lab_crop = lab_rs[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

    img_out = OUT_IMG / f"{case_id}_crop_img.npy"
    lab_out = OUT_LAB / f"{case_id}_crop_lab.npy"

    np.save(img_out, img_crop)
    np.save(lab_out, lab_crop)

    records.append({
        "case_id": case_id,
        "dataset": r["dataset"],
        "class": r["class"],
        "orig_shape": str(img.shape),
        "orig_spacing": str(spacing),
        "resampled_shape": str(img_rs.shape),
        "crop_shape": str(img_crop.shape),
        "image_npy": str(img_out.relative_to(ROOT)),
        "label_npy": str(lab_out.relative_to(ROOT))
    })

    print(case_id, "orig", img.shape, "rs", img_rs.shape, "crop", img_crop.shape)

out_csv = ROOT / "2_dataset" / "finetune_preprocessed_manifest.csv"
pd.DataFrame(records).to_csv(out_csv, index=False)

print("\nProcessed:", len(records))
print("Saved:", out_csv)
