from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(
    "/home/ubuntu/AAA_CTA_Project"
)


PRED_DIR = (
    ROOT /
    "8_results" /
    "predictions" /
    "msau_unet3d_v2"
)


MANIFEST = (
    ROOT /
    "2_dataset" /
    "finetune_preprocessed_manifest.csv"
)


OUT = (
    ROOT /
    "9_results" /
    "AAA_risk_stratification_final.csv"
)


manifest = pd.read_csv(
    MANIFEST
)



def get_info(case):

    row = manifest[
        manifest.case_id == case
    ].iloc[0]

    return (
        row["dataset"],
        row["class"]
    )



def compute_volume(mask):

    # resampled spacing = 1mm
    return float(
        mask.sum()
    )



def compute_max_diameter(mask):

    """
    Maximum axial diameter
    """

    max_d = 0


    for z in range(mask.shape[0]):

        slice_mask = mask[z]


        area = slice_mask.sum()


        if area == 0:
            continue


        diameter = (
            2 *
            np.sqrt(
                area / np.pi
            )
        )


        max_d = max(
            max_d,
            diameter
        )


    return max_d



def aaa_risk(d):

    if d < 40:
        return "Low"

    elif d < 55:
        return "Moderate"

    else:
        return "High"



results = []



for f in sorted(
    PRED_DIR.glob("*_pred.npy")
):

    case = f.stem.replace(
        "_pred",
        ""
    )


    print(
        "Processing",
        case
    )


    dataset, cls = get_info(case)


    mask = np.load(f) > 0


    volume = compute_volume(
        mask
    )


    diameter = compute_max_diameter(
        mask
    )


    if cls == "AAA":

        risk = aaa_risk(
            diameter
        )

    else:

        risk = "Normal"



    results.append(
        {

        "case_id":
            case,

        "dataset":
            dataset,

        "class":
            cls,

        "volume_mm3":
            volume,

        "diameter_mm":
            diameter,

        "risk_level":
            risk

        }
    )



df = pd.DataFrame(
    results
)


df.to_csv(
    OUT,
    index=False
)


print("\nSaved:")
print(OUT)


print("\nResults:")
print(df)


print("\nClass summary:")

print(
    df.groupby("class")
    [
        [
        "volume_mm3",
        "diameter_mm"
        ]
    ]
    .mean()
)

