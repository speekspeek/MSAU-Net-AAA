from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(
    "/home/ubuntu/AAA_CTA_Project"
)


CASES = [
    "Anev_05",
    "Anev_07",
    "Nat_03"
]


METHOD_DIRS = {

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



METHODS = [
"CTA",
"Ground Truth",
"3D U-Net",
"Attention U-Net",
"MSAU w/o Multiscale",
"MSAU w/o MSA",
"MSAU-Net"
]



OUT = ROOT/"9_results/figures"

OUT.mkdir(
    parents=True,
    exist_ok=True
)



plt.rcParams["svg.fonttype"]="none"
plt.rcParams["font.family"]="Arial"



# ==========================
# find prediction automatically
# ==========================

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



# ==========================
# load case
# ==========================

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

    data["CTA"]=img

    data["Ground Truth"]=gt



    for name,folder in METHOD_DIRS.items():

        p=find_pred(
            folder,
            case
        )

        print(
            case,
            name,
            p
        )


        data[name]=np.load(p)


    return data



# ==========================
# plot overlap contour
# ==========================

def show_mask(
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


    # Prediction red

    if pred.sum()>0:

        ax.contour(
            pred,
            colors="red",
            linewidths=1.2
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



# ==========================
# create figure
# ==========================


fig,axs=plt.subplots(

    len(METHODS),
    len(CASES),

    figsize=(12,16),

    dpi=300

)



for col,case in enumerate(CASES):


    data=load_case(case)



    gt=data["Ground Truth"]



    # maximum aneurysm slice

    z=np.argmax(
        gt.sum(axis=(0,1))
    )


    print(
        case,
        "slice:",
        z
    )


    img=data["CTA"][:,:,z]



    # CTA row

    axs[0,col].imshow(
        img,
        cmap="gray"
    )

    axs[0,col].set_title(
        case,
        fontsize=12,
        fontweight="bold"
    )


    axs[0,col].axis("off")



    # GT

    axs[1,col].imshow(
        img,
        cmap="gray"
    )


    axs[1,col].contour(
        gt[:,:,z],
        colors="lime",
        linewidths=1.5
    )


    axs[1,col].axis("off")



    # predictions

    names=[
        "3D U-Net",
        "Attention U-Net",
        "MSAU w/o Multiscale",
        "MSAU w/o MSA",
        "MSAU-Net"
    ]



    for r,name in enumerate(names,start=2):


        show_mask(

            axs[r,col],

            img,

            gt[:,:,z],

            data[name][:,:,z],

            name

        )



# row labels

for r,name in enumerate(METHODS):

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
        f"Fig2_AAA_segmentation_advanced.{ext}"
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



