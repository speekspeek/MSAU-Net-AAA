import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

df = pd.read_csv(ROOT / "2_dataset" / "AAA_finetune_dataset.csv")

OUT_IMG = ROOT / "3_preprocessed" / "Anev" / "images"
OUT_LAB = ROOT / "3_preprocessed" / "Anev" / "labels"

TARGET_SPACING = (1.5, 1.5, 1.5)

df = df[df["class"]=="AAA"].head(1)

for _, r in df.iterrows():

    case_id = r["case_id"]

    img_nii = nib.load(str(ROOT / r["image"]))
    lab_nii = nib.load(str(ROOT / r["label"]))

    img = img_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata().astype(np.uint8)

    spacing = img_nii.header.get_zooms()[:3]

    resize_factor = [
        spacing[0] / TARGET_SPACING[0],
        spacing[1] / TARGET_SPACING[1],
        spacing[2] / TARGET_SPACING[2]
    ]

    img_resampled = zoom(
        img,
        resize_factor,
        order=1
    )

    lab_resampled = zoom(
        lab,
        resize_factor,
        order=0
    )

    # HU clipping
    img_resampled = np.clip(img_resampled, -200, 400)

    # normalization
    img_resampled = (
        img_resampled - img_resampled.mean()
    ) / (img_resampled.std() + 1e-8)

    np.save(
        OUT_IMG / f"{case_id}_resampled_img.npy",
        img_resampled
    )

    np.save(
        OUT_LAB / f"{case_id}_resampled_lab.npy",
        lab_resampled
    )

    print("Case:", case_id)
    print("Original shape :", img.shape)
    print("Original spacing:", spacing)

    print("Resampled shape:", img_resampled.shape)
    print("Target spacing :", TARGET_SPACING)

    print("Label unique:", np.unique(lab_resampled))

print("\nSaved resampled sample.")
