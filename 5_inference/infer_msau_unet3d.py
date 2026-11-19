import os
import sys
import glob
import importlib.util
from pathlib import Path

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

MODEL_PATH = ROOT / "5_checkpoints/finetune/msau_unet3d_best.pth"
CSV_PATH = ROOT / "2_dataset/finetune_test.csv"
OUT_DIR = ROOT / "8_results/predictions/msau_unet3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_file = ROOT / "3_models/msau_unet3d.py"
spec = importlib.util.spec_from_file_location("attention_unet3d", model_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mod.MSAUNet3D(in_channels=1, num_classes=2).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt)
model.eval()

df = pd.read_csv(CSV_PATH)
print("Test cases:", len(df))
print("Columns:", df.columns.tolist())

def get_col(df, keys):
    for k in keys:
        for c in df.columns:
            if k.lower() in c.lower():
                return c
    return None

img_col = get_col(df, ["image", "img", "ct"])
case_col = get_col(df, ["case", "id", "name"])

if img_col is None:
    raise RuntimeError("Cannot find image column in finetune_test.csv")

for i, row in df.iterrows():
    img_path = Path(row[img_col])
    if not img_path.is_absolute():
        img_path = ROOT / img_path

    case_id = str(row[case_col]) if case_col else f"case_{i:03d}"

    arr = np.load(img_path)
    if arr.ndim == 3:
        arr = arr[None, None]
    elif arr.ndim == 4:
        arr = arr[None]
    else:
        raise RuntimeError(f"Unsupported shape: {arr.shape}")

    orig_d, orig_h, orig_w = arr.shape[-3:]

    pad_d = (8 - orig_d % 8) % 8
    pad_h = (8 - orig_h % 8) % 8
    pad_w = (8 - orig_w % 8) % 8

    if pad_d or pad_h or pad_w:
        arr = np.pad(
            arr,
            ((0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant"
        )

    x = torch.from_numpy(arr).float().to(device)

    with torch.no_grad():
        logits = model(x)
        if logits.shape[1] == 1:
            pred = torch.sigmoid(logits)
            pred = (pred > 0.5).float().cpu().numpy()[0, 0]
        else:
            pred = torch.softmax(logits, dim=1)
            pred = torch.argmax(pred, dim=1).float().cpu().numpy()[0]

        pred = pred[:orig_d, :orig_h, :orig_w]

    save_path = OUT_DIR / f"{case_id}_pred.npy"
    np.save(save_path, pred.astype(np.uint8))
    print("Saved:", save_path)

print("Inference finished.")
