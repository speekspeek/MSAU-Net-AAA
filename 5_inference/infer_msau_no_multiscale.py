import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

sys.path.append(str(ROOT))

from importlib.machinery import SourceFileLoader


model_mod = SourceFileLoader(
    "model",
    str(ROOT/"3_models"/"msau_unet3d_no_multiscale.py")
).load_module()



device=torch.device("cuda")


TEST_CSV = ROOT/"2_dataset"/"test_cases.csv"


MODEL_PATH = (
    ROOT/
    "5_checkpoints"/
    "finetune"/
    "msau_no_multiscale_best.pth"
)


OUT_DIR = (
    ROOT/
    "8_results"/
    "predictions"/
    "msau_no_multiscale"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



model=model_mod.MSAUNet3D_NoMultiScale(
    in_channels=1,
    num_classes=2,
    base_ch=16
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)


model.to(device)
model.eval()



df=pd.read_csv(TEST_CSV)



with torch.no_grad():

    for _,r in df.iterrows():

        cid=r["case_id"]

        img=np.load(
            ROOT/r["image_npy"]
        )


        x=torch.from_numpy(
            img
        ).float().unsqueeze(0).unsqueeze(0)


        x=x.to(device)


        pred=model(x)


        pred=torch.argmax(
            pred,
            dim=1
        )


        pred=pred.cpu().numpy()[0]


        np.save(
            OUT_DIR/f"{cid}_pred.npy",
            pred
        )


        print(
            "Saved:",
            cid
        )


print("Inference finished")
