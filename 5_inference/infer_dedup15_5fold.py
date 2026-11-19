import argparse
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path("/home/ubuntu/AAA_CTA_Project")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_CONFIG = {
    "unet3d": (
        "unet3d.py",
        "UNet3D"
    ),

    "attention_unet3d": (
        "attention_unet3d.py",
        "AttentionUNet3D"
    ),

    "msau_no_multiscale": (
        "msau_unet3d_no_multiscale.py",
        "MSAUNet3D_NoMultiScale"
    ),

    "msau_no_msa": (
        "msau_unet3d_msa_only.py",
        "MSAUNet3DMSAOnly"
    ),

    "msau_unet3d_v2": (
        "msau_unet3d_v2.py",
        "MSAUNet3DV2"
    ),
}


def pad_to_8(x):

    d, h, w = x.shape[-3:]

    pd_total = (8 - d % 8) % 8
    ph_total = (8 - h % 8) % 8
    pw_total = (8 - w % 8) % 8

    pad = [
        pw_total // 2,
        pw_total - pw_total // 2,
        ph_total // 2,
        ph_total - ph_total // 2,
        pd_total // 2,
        pd_total - pd_total // 2
    ]

    return F.pad(x, pad), pad


def crop_back(x, pad):

    pw1, pw2, ph1, ph2, pd1, pd2 = pad

    d = x.shape[-3] - pd1 - pd2
    h = x.shape[-2] - ph1 - ph2
    w = x.shape[-1] - pw1 - pw2

    return x[
        ...,
        pd1:pd1 + d,
        ph1:ph1 + h,
        pw1:pw1 + w
    ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIG.keys())
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5]
    )

    args = parser.parse_args()

    model_name = args.model
    fold = args.fold

    test_csv = (
        ROOT /
        "2_dataset" /
        "folds_dedup15" /
        f"fold{fold}_test.csv"
    )

    checkpoint = (
        ROOT /
        "5_checkpoints" /
        "folds_dedup15" /
        f"fold{fold}_{model_name}_best.pth"
    )

    save_dir = (
        ROOT /
        "8_results" /
        "predictions_dedup15_5fold" /
        model_name /
        f"fold{fold}"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not test_csv.exists():
        raise FileNotFoundError(test_csv)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model_file, class_name = MODEL_CONFIG[
        model_name
    ]

    module = SourceFileLoader(
        f"{model_name}_dedup15_inference",
        str(
            ROOT /
            "3_models" /
            model_file
        )
    ).load_module()

    ModelClass = getattr(
        module,
        class_name
    )

    model = ModelClass(
        in_channels=1,
        num_classes=2,
        base_ch=16
    )

    state = torch.load(
        checkpoint,
        map_location="cpu"
    )

    model.load_state_dict(
        state
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    df = pd.read_csv(
        test_csv
    )

    print("=" * 80)
    print("DEDUP15 STRICT OUT-OF-FOLD INFERENCE")
    print("=" * 80)

    print("Model      :", model_name)
    print("Fold       :", fold)
    print("Checkpoint :", checkpoint)
    print("Test cases :", len(df))
    print("Device     :", DEVICE)

    with torch.no_grad():

        for _, row in df.iterrows():

            case_id = str(
                row["case_id"]
            )

            image_path = (
                ROOT /
                row["image_npy"]
            )

            image = np.load(
                image_path
            ).astype(
                np.float32
            )

            original_shape = image.shape

            x = torch.from_numpy(
                image
            ).unsqueeze(0).unsqueeze(0)

            x, pad = pad_to_8(
                x
            )

            x = x.to(
                DEVICE
            )

            logits = model(
                x
            )

            pred = torch.argmax(
                logits,
                dim=1
            )

            pred = crop_back(
                pred,
                pad
            )

            pred = (
                pred
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            if pred.shape != original_shape:

                raise RuntimeError(
                    f"{case_id}: "
                    f"{pred.shape} != {original_shape}"
                )

            out = (
                save_dir /
                f"{case_id}_pred.npy"
            )

            np.save(
                out,
                pred
            )

            print(
                f"{case_id:10s} "
                f"{original_shape} -> "
                f"{pred.shape} "
                f"Saved"
            )

    print("=" * 80)
    print("Inference finished")
    print("Output:", save_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
