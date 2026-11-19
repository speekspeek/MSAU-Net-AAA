import sys
import argparse
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path("/home/ubuntu/AAA_CTA_Project")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model configuration
# ============================================================

MODEL_CONFIG = {

    "unet3d": {
        "file": "unet3d.py",
        "class": "UNet3D",
    },

    "attention_unet3d": {
        "file": "attention_unet3d.py",
        "class": "AttentionUNet3D",
    },

    "msau_no_multiscale": {
        "file": "msau_unet3d_no_multiscale.py",
        "class": "MSAUNet3D_NoMultiScale",
    },

    "msau_no_msa": {
        "file": "msau_unet3d_msa_only.py",
        "class": "MSAUNet3DMSAOnly",
    },

    "msau_unet3d_v2": {
        "file": "msau_unet3d_v2.py",
        "class": "MSAUNet3DV2",
    },
}


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    choices=list(MODEL_CONFIG.keys())
)

parser.add_argument(
    "--fold",
    type=int,
    required=True,
    choices=[1, 2, 3, 4, 5]
)

args = parser.parse_args()

model_name = args.model
fold = args.fold


# ============================================================
# Paths
# ============================================================

CSV = (
    ROOT /
    "2_dataset" /
    "folds" /
    f"fold{fold}_test.csv"
)

CKPT = (
    ROOT /
    "5_checkpoints" /
    "folds" /
    f"fold{fold}_{model_name}_best.pth"
)

SAVE_DIR = (
    ROOT /
    "8_results" /
    "predictions_strict_5fold" /
    model_name /
    f"fold{fold}"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Checks
# ============================================================

if not CSV.exists():
    raise FileNotFoundError(CSV)

if not CKPT.exists():
    raise FileNotFoundError(CKPT)


# ============================================================
# Load model
# ============================================================

cfg = MODEL_CONFIG[model_name]

mod = SourceFileLoader(
    f"{model_name}_strict_infer",
    str(
        ROOT /
        "3_models" /
        cfg["file"]
    )
).load_module()

ModelClass = getattr(
    mod,
    cfg["class"]
)

model = ModelClass(
    in_channels=1,
    num_classes=2,
    base_ch=16
)

state = torch.load(
    CKPT,
    map_location="cpu"
)

model.load_state_dict(
    state
)

model.to(
    DEVICE
)

model.eval()


# ============================================================
# Pad / crop
# ============================================================

def pad_to_8(x):

    d, h, w = x.shape[-3:]

    pd_total = (8 - d % 8) % 8
    ph_total = (8 - h % 8) % 8
    pw_total = (8 - w % 8) % 8

    pad = [
        pw_total // 2,
        pw_total - pw_total // 2,

        ph_total // 2,
        ph_total - ph_total // 2,

        pd_total // 2,
        pd_total - pd_total // 2
    ]

    x = F.pad(
        x,
        pad
    )

    return x, pad


def crop_back(x, pad):

    pw1, pw2, ph1, ph2, pd1, pd2 = pad

    d = x.shape[-3] - pd1 - pd2
    h = x.shape[-2] - ph1 - ph2
    w = x.shape[-1] - pw1 - pw2

    return x[
        ...,
        pd1:pd1 + d,
        ph1:ph1 + h,
        pw1:pw1 + w
    ]


# ============================================================
# Test cases
# ============================================================

df = pd.read_csv(
    CSV
)

print("=" * 80)
print("STRICT 5-FOLD INFERENCE")
print("=" * 80)

print("Model      :", model_name)
print("Fold       :", fold)
print("Checkpoint :", CKPT)
print("Test cases :", len(df))
print("Device     :", DEVICE)

print("=" * 80)


# ============================================================
# Inference
# ============================================================

with torch.no_grad():

    for _, row in df.iterrows():

        case_id = str(
            row["case_id"]
        )

        image_path = (
            ROOT /
            row["image_npy"]
        )

        img = np.load(
            image_path
        ).astype(
            np.float32
        )

        original_shape = img.shape

        x = torch.from_numpy(
            img
        )

        x = x.unsqueeze(0).unsqueeze(0)

        x, pad = pad_to_8(
            x
        )

        x = x.to(
            DEVICE
        )

        logits = model(
            x
        )

        pred = torch.argmax(
            logits,
            dim=1
        )

        pred = crop_back(
            pred,
            pad
        )

        pred = (
            pred.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        if pred.shape != original_shape:

            raise RuntimeError(
                f"{case_id}: "
                f"prediction shape {pred.shape} "
                f"!= image shape {original_shape}"
            )

        out = (
            SAVE_DIR /
            f"{case_id}_pred.npy"
        )

        np.save(
            out,
            pred
        )

        print(
            f"{case_id:10s} "
            f"{original_shape} -> "
            f"{pred.shape} "
            f"Saved"
        )


print("=" * 80)
print("Inference finished")
print("Output:", SAVE_DIR)
print("=" * 80)

