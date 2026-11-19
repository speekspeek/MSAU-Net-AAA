from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


ROOT = Path(
    "/home/ubuntu/AAA_CTA_Project"
)


MODELS = {

    "3D U-Net":
    ROOT / "8_results/predictions/unet3d",

    "Attention U-Net":
    ROOT / "8_results/predictions/attention_unet3d",

    "MSAU-Net":
    ROOT / "8_results/predictions/msau_unet3d_v2_fold"

}


GT_ROOT = ROOT / "3_preprocessed"


OUT = ROOT / "9_results/model_comparison.csv"



def dice(pred, gt):

    inter = np.logical_and(
        pred,
        gt
    ).sum()

    return (
        2.0 * inter /
        (
            pred.sum()
            +
            gt.sum()
            +
            1e-6
        )
    )



def iou(pred, gt):

    inter = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    return inter/(union+1e-6)



def hd95(pred, gt):

    if pred.sum()==0 or gt.sum()==0:
        return 999


    dt_pred = distance_transform_edt(
        ~pred
    )

    dt_gt = distance_transform_edt(
        ~gt
    )


    surface1 = dt_gt[pred]

    surface2 = dt_pred[gt]


    return np.percentile(
        np.concatenate(
            [
                surface1,
                surface2
            ]
        ),
        95
    )



def assd(pred, gt):

    if pred.sum()==0 or gt.sum()==0:
        return 999


    dt_pred = distance_transform_edt(
        ~pred
    )

    dt_gt = distance_transform_edt(
        ~gt
    )


    return (
        dt_gt[pred].mean()
        +
        dt_pred[gt].mean()
    ) / 2



results=[]



for name, pred_dir in MODELS.items():

    print("\nModel:", name)


    dices=[]
    ious=[]
    hd95s=[]
    assds=[]


    files=list(
        pred_dir.rglob(
            "*_pred.npy"
        )
    )


    for pred_file in files:


        case = pred_file.stem.replace(
            "_pred",
            ""
        )


        # crop label
        gt_files=list(
            GT_ROOT.rglob(
                f"{case}_crop_lab.npy"
            )
        )


        if len(gt_files)==0:

            print(
                "GT missing:",
                case
            )

            continue



        pred=np.load(
            pred_file
        ) > 0


        gt=np.load(
            gt_files[0]
        ) > 0



        if pred.shape != gt.shape:

            print(
                "Skip shape mismatch:",
                case,
                pred.shape,
                gt.shape
            )

            continue



        d=dice(
            pred,
            gt
        )

        i=iou(
            pred,
            gt
        )

        h=hd95(
            pred,
            gt
        )

        a=assd(
            pred,
            gt
        )


        dices.append(d)
        ious.append(i)
        hd95s.append(h)
        assds.append(a)



        print(
            case,
            "Dice",
            round(d,4)
        )



    results.append(
        {
            "Method":name,
            "N":len(dices),
            "Dice":np.mean(dices),
            "IoU":np.mean(ious),
            "HD95":np.mean(hd95s),
            "ASSD":np.mean(assds)
        }
    )



df=pd.DataFrame(results)


df.to_csv(
    OUT,
    index=False
)



print("\n================")
print(df)


print("\nSaved:")
print(OUT)

