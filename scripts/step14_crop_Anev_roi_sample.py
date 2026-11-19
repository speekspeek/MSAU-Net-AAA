import numpy as np
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

IMG_DIR = ROOT / "3_preprocessed" / "Anev" / "images"
LAB_DIR = ROOT / "3_preprocessed" / "Anev" / "labels"

OUT_IMG = ROOT / "3_preprocessed" / "Anev" / "images"
OUT_LAB = ROOT / "3_preprocessed" / "Anev" / "labels"

case_id = "Anev_01"
margin = 24

img = np.load(IMG_DIR / f"{case_id}_resampled_img.npy")
lab = np.load(LAB_DIR / f"{case_id}_resampled_lab.npy")

coords = np.argwhere(lab > 0)

zmin, ymin, xmin = coords.min(axis=0)
zmax, ymax, xmax = coords.max(axis=0)

zmin = max(zmin - margin, 0)
ymin = max(ymin - margin, 0)
xmin = max(xmin - margin, 0)

zmax = min(zmax + margin, img.shape[0] - 1)
ymax = min(ymax + margin, img.shape[1] - 1)
xmax = min(xmax + margin, img.shape[2] - 1)

img_crop = img[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
lab_crop = lab[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

np.save(OUT_IMG / f"{case_id}_crop_img.npy", img_crop)
np.save(OUT_LAB / f"{case_id}_crop_lab.npy", lab_crop)

print("Case:", case_id)
print("Original shape:", img.shape)
print("Crop bbox:", (zmin, zmax, ymin, ymax, xmin, xmax))
print("Cropped shape:", img_crop.shape)
print("Label unique:", np.unique(lab_crop))
print("Saved crop files.")
