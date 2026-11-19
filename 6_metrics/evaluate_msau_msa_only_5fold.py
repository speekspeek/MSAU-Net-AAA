import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


ROOT = Path("/home/ubuntu/AAA_CTA_Project")


PRED_ROOT = (
    ROOT /
    "8_results" /
    "predictions" /
    "msau_msa_only_fold"
)


OUT_DIR = (
    ROOT /
    "8_results" /
    "tables"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ==========================
# Metrics
# ==========================

def dice_score(pred, gt):

    inter = np.logical_and(pred, gt).sum()

    return (
        2.0 * inter /
        (pred.sum() + gt.sum() + 1e-8)
    )



def iou_score(pred, gt):

    inter = np.logical_and(pred, gt).sum()

    union = np.logical_or(pred, gt).sum()

    return inter/(union+1e-8)



def precision_score(pred, gt):

    tp = np.logical_and(pred,gt).sum()

    return tp/(pred.sum()+1e-8)



def recall_score(pred, gt):

    tp = np.logical_and(pred,gt).sum()

    return tp/(gt.sum()+1e-8)



def surface_points(mask):

    eroded = binary_erosion(mask)

    surface = mask ^ eroded

    return np.argwhere(surface)



# ==========================
# Fast HD95 + ASSD
# ==========================

def hd95_assd(pred, gt):

    p1 = surface_points(pred)

    p2 = surface_points(gt)


    if len(p1)==0 or len(p2)==0:

        return np.nan, np.nan



    tree_pred = cKDTree(p1)

    tree_gt = cKDTree(p2)



    d_pred_to_gt, _ = tree_gt.query(p1)

    d_gt_to_pred, _ = tree_pred.query(p2)



    all_dist = np.concatenate(
        [
            d_pred_to_gt,
            d_gt_to_pred
        ]
    )


    hd95 = np.percentile(
        all_dist,
        95
    )


    assd = (
        d_pred_to_gt.mean()
        +
        d_gt_to_pred.mean()
    ) / 2.0



    return hd95, assd





# ==========================
# Main
# ==========================


case_rows=[]



for fold_dir in sorted(PRED_ROOT.iterdir()):


    if not fold_dir.is_dir():

        continue



    fold = int(
        fold_dir.name.replace(
            "fold",
            ""
        )
    )


    test_csv = (
        ROOT /
        "2_dataset" /
        "folds" /
        f"fold{fold}_test.csv"
    )


    df = pd.read_csv(
        test_csv
    )


    print(
        "\nProcessing fold",
        fold
    )



    for _,row in df.iterrows():


        case_id = row["case_id"]



        pred_path = (
            fold_dir /
            f"{case_id}_pred.npy"
        )


        gt_path = (
            ROOT /
            row["label_npy"]
        )


        pred = np.load(
            pred_path
        )


        gt = np.load(
            gt_path
        )



        pred = pred > 0

        gt = gt > 0



        hd95, assd = hd95_assd(
            pred,
            gt
        )



        result = {

            "fold":fold,

            "case_id":case_id,

            "Dice":
                dice_score(pred,gt),

            "IoU":
                iou_score(pred,gt),

            "Precision":
                precision_score(pred,gt),

            "Recall":
                recall_score(pred,gt),

            "HD95_voxel":
                hd95,

            "ASSD_voxel":
                assd

        }


        case_rows.append(
            result
        )


        print(
            case_id,
            "Dice",
            round(result["Dice"],4),
            "HD95",
            round(hd95,4)
        )





case_df = pd.DataFrame(
    case_rows
)



case_file = (
    OUT_DIR /
    "msau_msa_only_5fold_case_metrics.csv"
)


case_df.to_csv(
    case_file,
    index=False
)



fold_df = (
    case_df
    .groupby("fold")
    [
        [
            "Dice",
            "IoU",
            "Precision",
            "Recall",
            "HD95_voxel",
            "ASSD_voxel"
        ]
    ]
    .mean()
    .reset_index()
)



fold_file = (
    OUT_DIR /
    "msau_msa_only_5fold_fold_metrics.csv"
)


fold_df.to_csv(
    fold_file,
    index=False
)




summary = pd.DataFrame(

    {

        "mean":
        fold_df.iloc[:,1:].mean(),

        "std":
        fold_df.iloc[:,1:].std()

    }

)



summary_file = (
    OUT_DIR /
    "msau_msa_only_5fold_mean_std.csv"
)



summary.to_csv(
    summary_file
)



print("\n===================")

print("Fold metrics:")

print(fold_df)


print("\nMean ± SD:")

print(summary)


print("\nSaved:")

print(case_file)

print(fold_file)

print(summary_file)

