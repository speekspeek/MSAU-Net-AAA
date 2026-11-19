import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from scipy.ndimage import zoom

ROOT = Path("/home/ubuntu/AAA_CTA_Project")
sys.path.append(str(ROOT))

model_mod = SourceFileLoader(
    "msau_unet3d_v2",
    str(ROOT / "3_models/msau_unet3d_v2.py")
).load_module()

df = pd.read_csv(ROOT / "2_dataset/TotalSeg_val.csv")
r = df.iloc[0]

img = nib.load(str(ROOT / r["image_path"])).get_fdata().astype(np.float32)
lab = nib.load(str(ROOT / r["label_path"])).get_fdata()
lab = (lab > 0).astype(np.uint8)

print("case:", Path(r["image_path"]).parent.name)
print("raw img shape:", img.shape)
print("raw img min/max/mean:", img.min(), img.max(), img.mean())
print("label voxels:", lab.sum())

img2 = np.clip(img, -200, 800)
img2 = (img2 + 200) / 1000.0
print("norm min/max/mean:", img2.min(), img2.max(), img2.mean())

# center crop one patch around label
coords = np.argwhere(lab > 0)
z, y, x = coords.mean(axis=0).astype(int)
pd, ph, pw = 64, 96, 96

z0 = max(0, z - pd // 2)
y0 = max(0, y - ph // 2)
x0 = max(0, x - pw // 2)

patch = img2[z0:z0+pd, y0:y0+ph, x0:x0+pw]
gt = lab[z0:z0+pd, y0:y0+ph, x0:x0+pw]

patch = np.pad(
    patch,
    ((0, max(0, pd-patch.shape[0])),
     (0, max(0, ph-patch.shape[1])),
     (0, max(0, pw-patch.shape[2]))),
    mode="constant"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_mod.MSAUNet3DV2(in_channels=1, num_classes=2, base_ch=16).to(device)
model.load_state_dict(torch.load(ROOT / "5_checkpoints/finetune/msau_unet3d_v2_best.pth", map_location=device))
model.eval()

with torch.no_grad():
    t = torch.from_numpy(patch).float()[None, None].to(device)
    logits = model(t)
    prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

print("patch shape:", patch.shape)
print("patch img min/max/mean:", patch.min(), patch.max(), patch.mean())
print("patch gt voxels:", gt.sum())
print("pred prob min/max/mean:", prob.min(), prob.max(), prob.mean())
print("pred voxels >0.5:", (prob > 0.5).sum())
print("pred voxels >0.3:", (prob > 0.3).sum())
print("pred voxels >0.1:", (prob > 0.1).sum())
