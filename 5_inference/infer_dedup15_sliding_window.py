import argparse
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path("/home/ubuntu/AAA_CTA_Project")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = (96, 96, 96)
STRIDE = (48, 48, 48)


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


def get_starts(length, patch, stride):

    if length <= patch:
        return [0]

    starts = list(
        range(
            0,
            length - patch + 1,
            stride
        )
    )

    last = length - patch

    if starts[-1] != last:
        starts.append(last)

    return starts


def pad_to_patch(x, patch_size):

    _, _, d, h, w = x.shape

    pd = max(0, patch_size[0] - d)
    ph = max(0, patch_size[1] - h)
    pw = max(0, patch_size[2] - w)

    pad = [
        0, pw,
        0, ph,
        0, pd
    ]

    if pd or ph or pw:
        x = F.pad(
            x,
            pad,
            mode="constant",
            value=0
        )

    return x, (d, h, w)


def sliding_window_predict(
    model,
    image,
    patch_size=(96, 96, 96),
    stride=(48, 48, 48)
):

    image = torch.from_numpy(
        image.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    image, original_shape = pad_to_patch(
        image,
        patch_size
    )

    _, _, D, H, W = image.shape

    z_starts = get_starts(
        D,
        patch_size[0],
        stride[0]
    )

    y_starts = get_starts(
        H,
        patch_size[1],
        stride[1]
    )

    x_starts = get_starts(
        W,
        patch_size[2],
        stride[2]
    )

    total_windows = (
        len(z_starts)
        *
        len(y_starts)
        *
        len(x_starts)
    )

    print(
        "Sliding windows:",
        total_windows,
        f"({len(z_starts)} x "
        f"{len(y_starts)} x "
        f"{len(x_starts)})"
    )

    prob_sum = torch.zeros(
        (1, 2, D, H, W),
        dtype=torch.float32,
        device="cpu"
    )

    count_sum = torch.zeros(
        (1, 1, D, H, W),
        dtype=torch.float32,
        device="cpu"
    )

    window_count = 0

    with torch.no_grad():

        for z in z_starts:

            for y in y_starts:

                for x in x_starts:

                    patch = image[
                        :,
                        :,
                        z:z + patch_size[0],
                        y:y + patch_size[1],
                        x:x + patch_size[2]
                    ]

                    patch = patch.to(
                        DEVICE,
                        non_blocking=True
                    )

                    logits = model(
                        patch
                    )

                    probs = torch.softmax(
                        logits,
                        dim=1
                    ).cpu()

                    prob_sum[
                        :,
                        :,
                        z:z + patch_size[0],
                        y:y + patch_size[1],
                        x:x + patch_size[2]
                    ] += probs

                    count_sum[
                        :,
                        :,
                        z:z + patch_size[0],
                        y:y + patch_size[1],
                        x:x + patch_size[2]
                    ] += 1.0

                    window_count += 1

    if torch.any(
        count_sum == 0
    ):
        raise RuntimeError(
            "Some voxels were not covered "
            "by sliding windows."
        )

    mean_prob = (
        prob_sum /
        count_sum
    )

    pred = torch.argmax(
        mean_prob,
        dim=1
    )[0]

    d, h, w = original_shape

    pred = pred[
        :d,
        :h,
        :w
    ]

    return (
        pred.numpy().astype(np.uint8),
        window_count
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIG.keys())
    )

    parser.add_argument(
        "--fold",
        required=True,
        type=int,
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
        "predictions_dedup15_sliding" /
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

    model_file, class_name = (
        MODEL_CONFIG[
            model_name
        ]
    )

    module = SourceFileLoader(
        f"{model_name}_sliding",
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
    print("DEDUP15 SLIDING-WINDOW INFERENCE")
    print("=" * 80)

    print("Model      :", model_name)
    print("Fold       :", fold)
    print("Patch size :", PATCH_SIZE)
    print("Stride     :", STRIDE)
    print("Checkpoint :", checkpoint)
    print("Test cases :", len(df))
    print("Device     :", DEVICE)

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

        print("\n" + "-" * 70)
        print("Case :", case_id)
        print("Shape:", image.shape)

        pred, n_windows = (
            sliding_window_predict(
                model,
                image,
                PATCH_SIZE,
                STRIDE
            )
        )

        if pred.shape != image.shape:
            raise RuntimeError(
                f"{case_id}: prediction "
                f"{pred.shape} != image "
                f"{image.shape}"
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
            "Foreground voxels:",
            int(
                pred.sum()
            )
        )

        print(
            "Windows:",
            n_windows
        )

        print(
            "Saved:",
            out
        )

    print("\n" + "=" * 80)
    print("Inference finished")
    print("=" * 80)


if __name__ == "__main__":
    main()
