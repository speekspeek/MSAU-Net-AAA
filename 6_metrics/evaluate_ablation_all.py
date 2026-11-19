from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


ROOT=Path(
"/home/ubuntu/AAA_CTA_Project"
)


MODELS={

"3D U-Net":
"8_results/predictions/unet3d_fold",

"Attention U-Net":
"8_results/predictions/attention_unet3d_fold",

"MSAU w/o Multiscale":
"8_results/predictions/msau_no_multiscale_fold",

"MSAU w/o MSA":
"8_results/predictions/msau_msa_only_fold",

"MSAU-Net":
"8_results/predictions/msau_unet3d_v2_fold"

}



GT_ROOT=ROOT/"3_preprocessed"


OUT=ROOT/"9_results/ablation_comparison.csv"



def dice(pred,gt):

    inter=np.logical_and(
        pred,gt
    ).sum()

    return (
        2*inter/
        (
            pred.sum()
            +
            gt.sum()
            +
            1e-6
        )
    )



def iou(pred,gt):

    inter=np.logical_and(
        pred,gt
    ).sum()

    union=np.logical_or(
        pred,gt
    ).sum()

    return inter/(union+1e-6)



def hd95(pred,gt):

    if pred.sum()==0 or gt.sum()==0:
        return 999


    dt1=distance_transform_edt(~pred)
    dt2=distance_transform_edt(~gt)


    s1=dt2[pred]
    s2=dt1[gt]


    return np.percentile(
        np.concatenate(
            [s1,s2]
        ),
        95
    )



def assd(pred,gt):

    if pred.sum()==0 or gt.sum()==0:
        return 999


    dt1=distance_transform_edt(~pred)
    dt2=distance_transform_edt(~gt)


    return (
        dt2[pred].mean()
        +
        dt1[gt].mean()
    )/2



results=[]



for name,folder in MODELS.items():

    print("\nModel:",name)


    pred_dir=ROOT/folder


    dices=[]
    ious=[]
    hd=[]
    ass=[]


    files=list(
        pred_dir.rglob(
            "*_pred.npy"
        )
    )


    for f in files:


        case=f.stem.replace(
            "_pred",
            ""
        )


        gt_files=list(
            GT_ROOT.rglob(
                f"{case}_crop_lab.npy"
            )
        )


        if len(gt_files)==0:
            continue


        pred=np.load(f)>0
        gt=np.load(gt_files[0])>0


        if pred.shape!=gt.shape:

            print(
                "skip",
                case,
                pred.shape,
                gt.shape
            )

            continue



        dices.append(
            dice(pred,gt)
        )

        ious.append(
            iou(pred,gt)
        )

        hd.append(
            hd95(pred,gt)
        )

        ass.append(
            assd(pred,gt)
        )


    results.append({

        "Method":name,

        "N":len(dices),

        "Dice":np.mean(dices),

        "IoU":np.mean(ious),

        "HD95":np.mean(hd),

        "ASSD":np.mean(ass)

    })



df=pd.DataFrame(results)


df.to_csv(
    OUT,
    index=False
)



print("\n==========")
print(df)


print("\nSaved:")
print(OUT)

