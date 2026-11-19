from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FIGURE2_DIR = ROOT / "12_publication_figures/figure2"

if not (FIGURE2_DIR / "plot_figure2.py").is_file():
    raise SystemExit(
        f"未找到图2脚本：{FIGURE2_DIR / 'plot_figure2.py'}"
    )

sys.path.insert(0, str(FIGURE2_DIR))

from plot_figure2 import configure_font


FOLD_ROOT = (
    ROOT / "2_dataset/folds_dedup15_v3_corrected"
)

RESULT_ROOT = ROOT / "9_results/corrected_v5_oof"

MODEL_CONFIG = [
    ("unet3d", "3D U-Net"),
    ("attention_unet3d", "Attention\nU-Net"),
    ("msau_no_multiscale", "MSAU-Net\nw/o Multi-Scale"),
    ("msau_no_msa", "MSAU-Net\nw/o Attention"),
    ("msau_unet3d_v2", "Full\nMSAU-Net"),
]

REF_COLOR = "#40D9E8"
PRED_COLOR = "#FF7676"

TP_COLOR = "#48C998"
FP_COLOR = "#F29A45"
FN_COLOR = "#B981E5"

ROI_COLOR = "#FFE27A"


# ============================================================
# Load outer-test manifests
# ============================================================

def resolve_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_manifest():
    frames = []

    for fold in range(1, 6):
        path = FOLD_ROOT / f"fold{fold}_test.csv"

        if not path.is_file():
            raise ValueError(f"未找到外层测试清单：{path}")

        frame = pd.read_csv(path)

        required = {
            "case_id",
            "image_npy",
            "label_npy",
        }

        if not required.issubset(frame.columns):
            raise ValueError(
                f"{path.name}缺少列："
                f"{sorted(required - set(frame.columns))}"
            )

        if len(frame) != 3:
            raise ValueError(
                f"{path.name}应包含3例外层测试病例。"
            )

        frame["case_id"] = (
            frame["case_id"].astype(str).str.strip()
        )

        frame["outer_fold"] = fold
        frames.append(frame)

    manifest = pd.concat(frames, ignore_index=True)

    if (
        len(manifest) != 15
        or manifest["case_id"].nunique() != 15
    ):
        raise ValueError(
            "五折测试清单必须包含15例无重复病例。"
        )

    return manifest.set_index("case_id")


# ============================================================
# Images, masks and slices
# ============================================================

def load_case(case_id, manifest):
    if case_id not in manifest.index:
        raise ValueError(
            f"{case_id}不在最终15例测试清单中。"
        )

    row = manifest.loc[case_id]
    fold = int(row["outer_fold"])

    image_path = resolve_path(row["image_npy"])
    reference_path = resolve_path(row["label_npy"])

    for path in [image_path, reference_path]:
        if not path.is_file():
            raise ValueError(f"未找到文件：{path}")

    image = np.load(image_path).astype(np.float32)
    reference = np.load(reference_path) > 0

    if (
        image.ndim != 3
        or image.shape != reference.shape
    ):
        raise ValueError(
            f"{case_id}图像和参考掩膜尺寸不一致："
            f"{image.shape} / {reference.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError(f"{case_id}图像存在非有限值。")

    predictions = []
    prediction_paths = []

    for model_id, _ in MODEL_CONFIG:
        path = (
            RESULT_ROOT
            / model_id
            / "predictions"
            / f"fold{fold}"
            / f"{case_id}_pred.npy"
        )

        if not path.is_file():
            raise ValueError(
                f"未找到外层测试预测：\n{path}\n"
                "请确认对应模型的最终OOF评价已保存预测。"
            )

        prediction = np.load(path) > 0

        if prediction.shape != reference.shape:
            raise ValueError(
                f"{case_id} / {model_id}尺寸不一致："
                f"{prediction.shape} / {reference.shape}"
            )

        predictions.append(prediction)
        prediction_paths.append(str(path))

    return {
        "case_id": case_id,
        "fold": fold,
        "image": image,
        "reference": reference,
        "predictions": predictions,
        "image_path": str(image_path),
        "reference_path": str(reference_path),
        "prediction_paths": prediction_paths,
    }


def choose_slice(reference, axis, supplied=None):
    if supplied is not None:
        if not 0 <= supplied < reference.shape[axis]:
            raise ValueError(
                f"切片索引{supplied}越界；"
                f"该轴长度为{reference.shape[axis]}。"
            )
        return supplied

    other_axes = tuple(
        index for index in range(3)
        if index != axis
    )

    areas = reference.sum(axis=other_axes)

    if areas.max() == 0:
        raise ValueError(
            "参考掩膜为空，无法自动选择展示切片。"
        )

    return int(np.argmax(areas))


def section(volume, axis, index):
    # Apply the same display rotation to all arrays.
    # No anatomical orientation labels are inferred.
    return np.rot90(
        np.take(volume, index, axis=axis)
    )


def display_limits(image):
    # The saved image is z-score normalized, not raw HU.
    low, high = np.percentile(image, [1, 99])

    if high <= low:
        high = low + 1.0

    return float(low), float(high)


def zoom_bounds(reference, predictions, margin):
    # Use one common crop for all models.
    union = reference.copy()

    for prediction in predictions:
        union |= prediction

    coordinates = np.argwhere(union)

    if not len(coordinates):
        return (
            0,
            reference.shape[0],
            0,
            reference.shape[1],
        )

    lower = coordinates.min(axis=0) - margin
    upper = coordinates.max(axis=0) + margin + 1

    r0 = max(0, int(lower[0]))
    r1 = min(reference.shape[0], int(upper[0]))
    c0 = max(0, int(lower[1]))
    c1 = min(reference.shape[1], int(upper[1]))

    return r0, r1, c0, c1


# ============================================================
# Rendering
# ============================================================

def rgb_color(hex_color):
    return np.array(
        matplotlib.colors.to_rgb(hex_color)
    )


def overlay(ax, mask, color, alpha=0.42):
    rgba = np.zeros(
        (*mask.shape, 4),
        dtype=np.float32,
    )

    rgba[..., :3] = rgb_color(color)
    rgba[..., 3] = mask.astype(float) * alpha

    ax.imshow(
        rgba,
        interpolation="nearest",
    )


def show_image(ax, image, limits):
    ax.imshow(
        image,
        cmap="gray",
        vmin=limits[0],
        vmax=limits[1],
        interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def show_errors(ax, prediction, reference):
    tp = prediction & reference
    fp = prediction & ~reference
    fn = ~prediction & reference

    overlay(ax, tp, TP_COLOR, alpha=0.52)
    overlay(ax, fp, FP_COLOR, alpha=0.78)
    overlay(ax, fn, FN_COLOR, alpha=0.78)


def dice(prediction, reference):
    denominator = (
        prediction.sum() + reference.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        2.0
        * np.logical_and(
            prediction, reference
        ).sum()
        / denominator
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        nargs="+",
        default=["Anev_04", "Nat_03"],
    )

    parser.add_argument(
        "--axis",
        type=int,
        choices=[0, 1, 2],
        default=2,
    )

    parser.add_argument(
        "--slices",
        nargs="+",
        type=int,
        default=None,
        help="Each index refers to the saved cropped array.",
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--font",
        default=(
            "/usr/share/fonts/truetype/"
            "msttcorefonts/times.ttf"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=HERE / "outputs",
    )

    args = parser.parse_args()

    if args.slices is not None:
        if len(args.slices) != len(args.cases):
            raise ValueError(
                "--slices的数量必须与--cases一致。"
            )

    if args.margin < 0:
        raise ValueError("--margin不能为负数。")

    if len(set(args.cases)) != len(args.cases):
        raise ValueError("--cases中不能有重复病例。")

    configure_font(args.font)
    manifest = load_manifest()

    # Load all files before generating the figure.
    cases = [
        load_case(case_id, manifest)
        for case_id in args.cases
    ]

    n_cases = len(cases)
    n_rows = n_cases * 2
    n_columns = 7

    figure_height = 2.7 * n_rows + 1.5

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(18, figure_height),
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.085,
        top=0.92,
        wspace=0.07,
        hspace=0.18,
    )

    headers = [
        "CT/CTA\ninput",
        "Reference\nmask",
    ] + [
        label
        for _, label in MODEL_CONFIG
    ]

    for column, header in enumerate(headers):
        position = axes[0, column].get_position()

        fig.text(
            (position.x0 + position.x1) / 2,
            0.945,
            header,
            ha="center",
            va="center",
            fontsize=14,
        )

    metadata = []
    metric_rows = []

    for case_index, data in enumerate(cases):
        supplied = (
            None
            if args.slices is None
            else args.slices[case_index]
        )

        slice_index = choose_slice(
            data["reference"],
            args.axis,
            supplied,
        )

        image = section(
            data["image"],
            args.axis,
            slice_index,
        )

        reference = section(
            data["reference"],
            args.axis,
            slice_index,
        )

        predictions = [
            section(
                prediction,
                args.axis,
                slice_index,
            )
            for prediction in data["predictions"]
        ]

        limits = display_limits(
            data["image"]
        )

        r0, r1, c0, c1 = zoom_bounds(
            reference,
            predictions,
            args.margin,
        )

        crop = np.s_[r0:r1, c0:c1]

        overview_row = 2 * case_index
        detail_row = overview_row + 1

        # Overview row: input, reference, five predictions.
        for column in range(n_columns):
            ax = axes[overview_row, column]

            show_image(
                ax,
                image,
                limits,
            )

            if column == 1:
                overlay(
                    ax,
                    reference,
                    REF_COLOR,
                )

            elif column >= 2:
                overlay(
                    ax,
                    predictions[column - 2],
                    PRED_COLOR,
                )

            rectangle = Rectangle(
                (c0 - 0.5, r0 - 0.5),
                c1 - c0,
                r1 - r0,
                fill=False,
                edgecolor=ROI_COLOR,
                linewidth=1.0,
            )

            ax.add_patch(rectangle)

        # Detail row: common enlarged crop and error maps.
        for column in range(n_columns):
            ax = axes[detail_row, column]

            show_image(
                ax,
                image[crop],
                limits,
            )

            if column == 1:
                overlay(
                    ax,
                    reference[crop],
                    REF_COLOR,
                )

            elif column >= 2:
                show_errors(
                    ax,
                    predictions[column - 2][crop],
                    reference[crop],
                )

        for row_index, label in [
            (
                overview_row,
                f"{data['case_id']}\nOverview",
            ),
            (
                detail_row,
                f"{data['case_id']}\nError detail",
            ),
        ]:
            position = axes[
                row_index, 0
            ].get_position()

            fig.text(
                0.032,
                (position.y0 + position.y1) / 2,
                label,
                rotation=90,
                ha="center",
                va="center",
                fontsize=14,
            )

        for model_index, (model_id, _) in enumerate(
            MODEL_CONFIG
        ):
            metric_rows.append({
                "case_id": data["case_id"],
                "outer_fold": data["fold"],
                "model": model_id,
                "volume_dice": dice(
                    data["predictions"][model_index],
                    data["reference"],
                ),
                "slice_dice": dice(
                    predictions[model_index],
                    reference,
                ),
                "array_axis": args.axis,
                "slice_index": slice_index,
            })

        metadata.append({
            "case_id": data["case_id"],
            "outer_fold": data["fold"],
            "image_path": data["image_path"],
            "reference_path": data["reference_path"],
            "prediction_paths": data["prediction_paths"],
            "array_shape": list(data["image"].shape),
            "array_axis": args.axis,
            "slice_index": slice_index,
            "slice_selection": (
                "User-specified"
                if supplied is not None
                else "Maximum reference foreground area"
            ),
            "display_rotation": "np.rot90, applied to all arrays",
            "display_limits": list(limits),
            "common_zoom_bounds_rc": [
                r0, r1, c0, c1
            ],
        })

        print(
            f"{data['case_id']}: "
            f"fold={data['fold']}, "
            f"axis={args.axis}, "
            f"slice={slice_index}, "
            f"shape={data['image'].shape}"
        )

    handles = [
        Patch(
            facecolor=REF_COLOR,
            label="Reference",
        ),
        Patch(
            facecolor=PRED_COLOR,
            label="Prediction",
        ),
        Patch(
            facecolor=TP_COLOR,
            label="True positive",
        ),
        Patch(
            facecolor=FP_COLOR,
            label="False positive",
        ),
        Patch(
            facecolor=FN_COLOR,
            label="False negative",
        ),
        Patch(
            facecolor="none",
            edgecolor=ROI_COLOR,
            label="Detail region",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.034),
        ncol=6,
        frameon=False,
        fontsize=14,
        handlelength=1.2,
        columnspacing=1.3,
    )

    fig.text(
        0.075,
        0.018,
        "Each case uses the same slice, display window "
        "and detail region across all models.",
        ha="left",
        va="center",
        fontsize=14,
    )

    outdir = args.outdir.expanduser().resolve()

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = (
        outdir
        / "Figure4_segmentation_error_comparison"
    )

    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".pdf"),
        dpi=300,
        facecolor="white",
    )

    pd.DataFrame(metric_rows).to_csv(
        outdir / "Figure4_case_metrics.csv",
        index=False,
    )

    (
        outdir / "Figure4_display_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plt.close(fig)

    print("=" * 65)
    print("Figure 4 completed")
    print("Font   : Times New Roman, 14 pt")
    print("PNG    : 600 dpi")
    print("Data   : corrected V5 outer-test predictions")
    print(f"Output : {outdir}")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        KeyError,
        OSError,
    ) as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
