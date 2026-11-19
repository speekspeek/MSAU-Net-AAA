import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


import sys
sys.path.append(
    "/home/ubuntu/AAA_CTA_Project/3_models"
)


import msau_unet3d_v2 as model_mod



ROOT=Path(
"/home/ubuntu/AAA_CTA_Project"
)


DEVICE="cuda"



parser=argparse.ArgumentParser()

parser.add_argument(
    "--fold",
    type=int,
    required=True
)

args=parser.parse_args()


fold=args.fold



# ==========================
# paths
# ==========================

CSV=(
ROOT/
"2_dataset"/
"folds"/
f"fold{fold}_test.csv"
)



SAVE_DIR=(
ROOT/
"8_results"/
"predictions"/
"msau_unet3d_v2_fold"/
f"fold{fold}"
)


SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)



CKPT=(
ROOT/
"5_checkpoints"/
"finetune"/
"msau_unet3d_v2_best.pth"
)



# ==========================
# load model
# ==========================


model=model_mod.MSAUNet3DV2(
    in_channels=1,
    num_classes=2,
    base_ch=16
)



state=torch.load(
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



# ==========================
# test list
# ==========================

df=pd.read_csv(
    CSV
)


print(
"Fold:",
fold
)

print(
"Test cases:",
len(df)
)



# ==========================
# inference
# ==========================

with torch.no_grad():

    for _,row in df.iterrows():

        case=row["case_id"]


        img_path=ROOT/row["image_npy"]


        img=np.load(
            img_path
        )


        img=torch.tensor(
            img,
            dtype=torch.float32
        )


        if img.ndim==3:
            img=img.unsqueeze(0).unsqueeze(0)


        img=img.to(
            DEVICE
        )


        pred=model(img)


        pred=torch.argmax(
            pred,
            dim=1
        )


        pred=pred.squeeze().cpu().numpy()


        out=(
            SAVE_DIR/
            f"{case}_pred.npy"
        )


        np.save(
            out,
            pred.astype(np.uint8)
        )


        print(
            "Saved:",
            out
        )


print(
"Inference finished"
)

