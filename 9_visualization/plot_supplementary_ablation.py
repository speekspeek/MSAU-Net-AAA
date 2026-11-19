from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT=Path(
    "/home/ubuntu/AAA_CTA_Project"
)


# ======================
# cases
# ======================

CASES=[
    "Anev_05",
    "Anev_07",
    "Anev_09",
    "Nat_03"
]


# ======================
# methods
# ======================

METHODS={

"Ground Truth":None,


"MSAU w/o Multiscale":
"8_results/predictions/msau_no_multiscale_fold",


"MSAU w/o MSA":
"8_results/predictions/msau_msa_only_fold",


"MSAU-Net":
"8_results/predictions/msau_unet3d_v2_fold"

}



ROWS=list(METHODS.keys())



OUT=ROOT/"9_results/figures"

OUT.mkdir(
    parents=True,
    exist_ok=True
)



plt.rcParams["font.family"]="Arial"
plt.rcParams["svg.fonttype"]="none"



# ======================
# find prediction
# ======================

def find_pred(folder,case):

    files=list(
        (ROOT/folder).rglob(
            f"{case}_pred.npy"
        )
    )

    if len(files)==0:
        raise FileNotFoundError(
            case,
            folder
        )

    return files[0]




# ======================
# load
# ======================

def load_case(case):


    img=np.load(
        ROOT/
        f"3_preprocessed/Anev/images/{case}_crop_img.npy"
    )


    gt=np.load(
        ROOT/
        f"3_preprocessed/Anev/labels/{case}_crop_lab.npy"
    )


    data={}


    data["Ground Truth"]=gt


    for name,folder in METHODS.items():

        if folder is None:
            continue


        data[name]=np.load(
            find_pred(
                folder,
                case
            )
        )


    return img,data




# ======================
# contour visualization
# ======================


def draw(
        ax,
        img,
        gt,
        pred,
        title
):


    ax.imshow(
        img,
        cmap="gray"
    )


    # GT green

    if gt.sum()>0:

        ax.contour(
            gt,
            colors="lime",
            linewidths=1.5
        )


    # prediction red

    if pred.sum()>0:

        ax.contour(
            pred,
            colors="red",
            linewidths=1.3
        )


    # overlap yellow

    overlap=np.logical_and(
        gt,
        pred
    )


    if overlap.sum()>0:

        ax.contour(
            overlap,
            colors="yellow",
            linewidths=0.8
        )


    ax.set_title(
        title,
        fontsize=9
    )

    ax.axis("off")





# ======================
# create figure
# ======================


fig,axs=plt.subplots(

    len(ROWS),
    len(CASES),

    figsize=(12,9),

    dpi=300

)



for c,case in enumerate(CASES):


    img,data=load_case(case)


    gt=data["Ground Truth"]



    z=np.argmax(
        gt.sum(axis=(0,1))
    )


    img2d=img[:,:,z]


    for r,name in enumerate(ROWS):


        ax=axs[r,c]


        if name=="Ground Truth":

            ax.imshow(
                img2d,
                cmap="gray"
            )


            ax.contour(
                gt[:,:,z],
                colors="lime",
                linewidths=1.5
            )

            ax.axis("off")


        else:


            draw(

                ax,

                img2d,

                gt[:,:,z],

                data[name][:,:,z],

                name

            )



        if r==0:

            ax.set_title(
                case,
                fontsize=12,
                fontweight="bold"
            )



for r,name in enumerate(ROWS):

    axs[r,0].set_ylabel(
        name,
        fontsize=10
    )



plt.tight_layout()



for ext in [
    "png",
    "svg",
    "pdf"
]:

    save=(
        OUT/
        f"Supplementary_Fig_S1_Ablation.{ext}"
    )


    plt.savefig(
        save,
        dpi=600,
        bbox_inches="tight",
        format=ext
    )


    print(
        "Saved:",
        save
    )


