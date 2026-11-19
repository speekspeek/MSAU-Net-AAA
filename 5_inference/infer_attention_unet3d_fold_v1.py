import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.append("/home/ubuntu/AAA_CTA_Project/3_models")

import attention_unet3d as model_mod


ROOT=Path("/home/ubuntu/AAA_CTA_Project")
DEVICE="cuda"


parser=argparse.ArgumentParser()
parser.add_argument("--fold",type=int,required=True)
args=parser.parse_args()

fold=args.fold


CSV=ROOT/"2_dataset/folds"/f"fold{fold}_test.csv"


SAVE_DIR=ROOT/"8_results/predictions/attention_unet3d_fold"/f"fold{fold}"
SAVE_DIR.mkdir(parents=True,exist_ok=True)


CKPT=ROOT/"5_checkpoints/finetune/attention_unet3d_best.pth"



model=model_mod.AttentionUNet3D(
    in_channels=1,
    num_classes=2
)


state=torch.load(
    CKPT,
    map_location="cpu"
)

model.load_state_dict(state)

model.to(DEVICE)

model.eval()



def pad_to_8(x):

    d,h,w=x.shape[-3:]

    pd=(8-d%8)%8
    ph=(8-h%8)%8
    pw=(8-w%8)%8

    pad=[
        pw//2,pw-pw//2,
        ph//2,ph-ph//2,
        pd//2,pd-pd//2
    ]

    return F.pad(x,pad),pad



def crop_back(x,pad):

    pw1,pw2,ph1,ph2,pd1,pd2=pad

    return x[
        ...,
        pd1:x.shape[-3]-pd2,
        ph1:x.shape[-2]-ph2,
        pw1:x.shape[-1]-pw2
    ]



df=pd.read_csv(CSV)

print("Fold:",fold)
print("Test cases:",len(df))


with torch.no_grad():

    for _,row in df.iterrows():

        case=row["case_id"]


        img=np.load(
            ROOT/row["image_npy"]
        )


        img=torch.tensor(
            img,
            dtype=torch.float32
        )


        img=img.unsqueeze(0).unsqueeze(0)


        img,pad=pad_to_8(img)


        img=img.to(DEVICE)


        pred=model(img)


        pred=torch.argmax(
            pred,
            dim=1
        )


        pred=crop_back(
            pred,
            pad
        )


        pred=pred.squeeze().cpu().numpy()


        out=SAVE_DIR/f"{case}_pred.npy"


        np.save(
            out,
            pred.astype(np.uint8)
        )


        print("Saved:",out)



print("Inference finished")

