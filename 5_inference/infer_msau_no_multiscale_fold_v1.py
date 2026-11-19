import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F


ROOT = Path("/home/ubuntu/AAA_CTA_Project")

sys.path.append(str(ROOT))


from importlib.machinery import SourceFileLoader


# ==============================
# Load model
# ==============================

model_mod = SourceFileLoader(
    "msau_no_multiscale",
    str(
        ROOT /
        "3_models" /
        "msau_unet3d_no_multiscale.py"
    )
).load_module()



# ==============================
# Arguments
# ==============================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--fold",
    type=int,
    required=True
)

args = parser.parse_args()

fold = args.fold



device = torch.device("cuda")



# ==============================
# Paths
# ==============================

TEST_CSV = (
    ROOT /
    "2_dataset" /
    "folds" /
    f"fold{fold}_test.csv"
)


MODEL_PATH = (
    ROOT /
    "5_checkpoints" /
    "finetune" /
    "msau_no_multiscale_best.pth"
)



OUT_DIR = (
    ROOT /
    "8_results" /
    "predictions" /
    "msau_no_multiscale_fold" /
    f"fold{fold}"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)




# ==============================
# Model
# ==============================

model = model_mod.MSAUNet3D_NoMultiScale(
    in_channels=1,
    num_classes=2,
    base_ch=16
)



checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)


model.load_state_dict(
    checkpoint
)


model.to(device)

model.eval()



# ==============================
# Load test cases
# ==============================

df = pd.read_csv(TEST_CSV)


print("Fold:", fold)

print(
    "Test cases:",
    len(df)
)


print(
    "Columns:",
    list(df.columns)
)




# ==============================
# Inference
# ==============================

with torch.no_grad():


    for _, row in df.iterrows():


        case_id = row["case_id"]



        img_path = (
            ROOT /
            row["image_npy"]
        )



        img = np.load(
            img_path
        )



        x = torch.from_numpy(
            img
        ).float()



        # [D,H,W]
        # -> [1,1,D,H,W]

        x = (
            x
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )



        pred = model(x)



        # 保证尺寸一致

        if pred.shape[2:] != x.shape[2:]:

            pred = F.interpolate(
                pred,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False
            )



        pred = torch.argmax(
            pred,
            dim=1
        )



        pred = (
            pred
            .cpu()
            .numpy()[0]
            .astype(np.uint8)
        )



        save_path = (
            OUT_DIR /
            f"{case_id}_pred.npy"
        )


        np.save(
            save_path,
            pred
        )


        print(
            "Saved:",
            save_path
        )



print("\nInference finished.")

