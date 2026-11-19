from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT=Path(
    "/home/ubuntu/AAA_CTA_Project"
)


CASES=[
    "Anev_05",
    "Anev_07",
    "Nat_03"
]


METHOD_DIRS={

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



OUT=ROOT/"9_results/figures"

OUT.mkdir(
    parents=True,
    exist_ok=True
)



plt.rcParams["font.family"]="Arial"

plt.rcParams["svg.fonttype"]="none"



def find_prediction(folder,case):

    files=list(
        (
            ROOT/folder
        ).rglob(
            f"{case}_pred.npy"
        )
    )


    if len(files)==0:

        raise FileNotFoundError(
            f"{case} not found in {folder}"
        )


    return files[0]



def load_case(case):


    img=np.load(
        ROOT/
        f"3_preprocessed/Anev/images/{case}_crop_img.npy"
    )


    gt=np.load(
        ROOT/
        f"3_preprocessed/Anev/labels/{case}_crop_lab.npy"
    )


    preds={}


    for name,folder in METHOD_DIRS.items():


        path=find_prediction(
            folder,
            case
        )


        print(
            name,
            case,
            path
        )


        preds[name]=np.load(
            path
        )



    preds["Ground Truth"]=gt


    return img,preds



METHODS=[

"Ground Truth",
"3D U-Net",
"Attention U-Net",
"MSAU w/o Multiscale",
"MSAU w/o MSA",
"MSAU-Net"

]



fig,axs=plt.subplots(
    len(METHODS),
    len(CASES),
    figsize=(10,15),
    dpi=300
)



for c,case in enumerate(CASES):


    img,preds=load_case(case)


    gt=preds["Ground Truth"]


    z=np.argmax(
        gt.sum(axis=(0,1))
    )


    print(
        case,
        "slice",
        z
    )


    img2d=img[:,:,z]


    for r,method in enumerate(METHODS):


        ax=axs[r,c]


        ax.imshow(
            img2d,
            cmap="gray"
        )


        mask=preds[method][:,:,z]


        if method=="Ground Truth":

            ax.imshow(
                mask,
                cmap="Greens",
                alpha=0.45
            )

        else:

            ax.imshow(
                mask,
                cmap="Reds",
                alpha=0.45
            )


        ax.axis("off")



        if r==0:

            ax.set_title(
                case,
                fontsize=12,
                fontweight="bold"
            )



for r,m in enumerate(METHODS):

    axs[r,0].set_ylabel(
        m,
        fontsize=10
    )



plt.tight_layout()



for ext in ["png","svg","pdf"]:

    plt.savefig(
        OUT/f"Fig2_segmentation_3cases.{ext}",
        dpi=600,
        bbox_inches="tight",
        format=ext
    )



print("Saved:")
print(
OUT/"Fig2_segmentation_3cases.svg"
)

