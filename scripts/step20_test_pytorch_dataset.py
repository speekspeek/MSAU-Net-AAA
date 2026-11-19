import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

class AortaPatchDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        img = np.load(ROOT / r["image_npy"])
        lab = np.load(ROOT / r["label_npy"])

        z0, y0, x0 = int(r["z0"]), int(r["y0"]), int(r["x0"])
        dz, dy, dx = int(r["dz"]), int(r["dy"]), int(r["dx"])

        img = img[z0:z0+dz, y0:y0+dy, x0:x0+dx]
        lab = lab[z0:z0+dz, y0:y0+dy, x0:x0+dx]

        img = torch.from_numpy(img).float().unsqueeze(0)
        lab = torch.from_numpy(lab).long()

        return img, lab, r["case_id"], r["class"]

csv_path = ROOT / "2_dataset" / "finetune_patch_index_v2.csv"

ds = AortaPatchDataset(csv_path)
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

print("Dataset patches:", len(ds))

for img, lab, case_id, cls in dl:
    print("Image batch:", img.shape, img.dtype)
    print("Label batch:", lab.shape, lab.dtype)
    print("Label unique:", torch.unique(lab))
    print("Case:", case_id)
    print("Class:", cls)
    break
