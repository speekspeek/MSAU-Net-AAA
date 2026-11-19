import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

df = pd.read_csv(ROOT / "2_dataset" / "AAA_finetune_dataset.csv")

OUT_IMG = ROOT / "3_preprocessed" / "Anev" / "images"
OUT_LAB = ROOT / "3_preprocessed" / "Anev" / "labels"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LAB.mkdir(parents=True, exist_ok=True)

# 只处理前2个AAA病例测试
df = df[df["class"]=="AAA"].head(2)

for _, r in df.iterrows():

    case_id = r["case_id"]

    img_path = ROOT / r["image"]
    lab_path = ROOT / r["label"]

    img_nii = nib.load(str(img_path))
    lab_nii = nib.load(str(lab_path))

    img = img_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata().astype(np.uint8)

    # HU clipping
    img = np.clip(img, -200, 400)

    # Z-score normalization
    mean = img.mean()
    std = img.std()

    img = (img - mean) / (std + 1e-8)

    np.save(OUT_IMG / f"{case_id}_img.npy", img)
    np.save(OUT_LAB / f"{case_id}_lab.npy", lab)

    print(f"{case_id}")
    print(" shape:", img.shape)
    print(" mean :", img.mean())
    print(" std  :", img.std())
    print(" label unique:", np.unique(lab))

print("\nSaved preprocessed samples.")
