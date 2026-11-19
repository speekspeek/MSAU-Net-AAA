from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("/home/ubuntu/AAA_CTA_Project")


PRED_DIR = (
    ROOT /
    "8_results" /
    "predictions" /
    "msau_no_multiscale_fold"
)


MANIFEST = (
    ROOT /
    "2_dataset" /
    "finetune_preprocessed_manifest.csv"
)


OUT = (
    ROOT /
    "9_results" /
    "AAA_risk_stratification_20cases.csv"
)


manifest = pd.read_csv(MANIFEST)


def get_class(case):

    row = manifest[
        manifest.case_id == case
    ]

    return row.iloc[0]["class"]



def compute_volume(mask):

    return float(mask.sum())



def compute_max_diameter(mask):

    max_d = 0


    # Z,Y,X
    for z in range(mask.shape[0]):

        area = mask[z].sum()

        if area == 0:
            continue


        d = 2 * np.sqrt(
            area / np.pi
        )


        max_d = max(
            max_d,
            d
        )


    return max_d



def risk_level(d):

    if d < 40:
        return "Low"

    elif d < 55:
        return "Moderate"

    else:
        return "High"



results=[]


for f in sorted(
    PRED_DIR.rglob("*_pred.npy")
):

    case = f.stem.replace(
        "_pred",
        ""
    )


    cls = get_class(case)


    print(
        "Processing",
        case,
        cls
    )


    mask = np.load(f) > 0


    volume = compute_volume(mask)

    diameter = compute_max_diameter(mask)


    if cls == "AAA":

        risk = risk_level(
            diameter
        )

    else:

        risk = "Normal"



    results.append(
        {
            "case_id":case,
            "class":cls,
            "volume_mm3":volume,
            "diameter_mm":diameter,
            "risk_level":risk
        }
    )



df = pd.DataFrame(results)


df.to_csv(
    OUT,
    index=False
)


print("\nSaved:")
print(OUT)


print("\nResults:")
print(df)


print("\nSummary:")
print(
    df.groupby("class")
    [
        ["volume_mm3","diameter_mm"]
    ]
    .mean()
)


print("\nRisk distribution:")
print(
    df.risk_level.value_counts()
)

