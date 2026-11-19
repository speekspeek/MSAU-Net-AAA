from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FIGURE2_DIR = ROOT / "12_publication_figures/figure2"

if not (FIGURE2_DIR / "plot_figure2.py").is_file():
    raise SystemExit("请保留figure2文件夹及plot_figure2.py。")

sys.path.insert(0, str(FIGURE2_DIR))
from plot_figure2 import configure_font


AUTO_ROOT = ROOT / "9_results/auto_roi_5fold_v6_end2end"

PRIMARY_CSV = (
    ROOT
    / "9_results/corrected_v5_oof"
    / "FINAL_5MODEL_RESULTS"
    / "final_5model_per_case.csv"
)

FOLD_COLORS = {
    1: "#6588AA",
    2: "#609D9A",
    3: "#B299C6",
    4: "#D6A15D",
    5: "#CB6676",
}


def load_auto_results(folder, selected_folds=None):
    frames = []
    missing_folds = []

    folds = selected_folds or [1, 2, 3, 4, 5]

    for fold in folds:
        path = folder / f"fold{fold}" / "metrics.csv"

        if not path.is_file():
            missing_folds.append(fold)
            continue

        frame = pd.read_csv(path)

        required = {
            "case_id",
            "fold",
            "Dice",
            "roi_gt_coverage",
            "roi_fraction",
        }

        if not required.issubset(frame.columns):
            raise ValueError(
                f"{path}缺少列："
                f"{sorted(required - set(frame.columns))}"
            )

        if frame.empty:
            continue

        frame["case_id"] = (
            frame["case_id"].astype(str).str.strip()
        )

        frame["fold"] = pd.to_numeric(
            frame["fold"], errors="raise"
        )

        if not frame["fold"].eq(fold).all():
            raise ValueError(f"{path}中的fold与文件夹不一致。")

        if len(frame) > 3:
            raise ValueError(f"{path}超过3例，请核对结果版本。")

        frame["source_csv"] = str(path)
        frames.append(frame)

    if selected_folds and missing_folds:
        raise ValueError(
            f"指定的折尚无结果：{missing_folds}"
        )

    if not frames:
        raise ValueError(
            f"未找到V6结果：{folder}/fold*/metrics.csv\n"
            "需要已有评价结果，不会自动重新训练或推理。"
        )

    df = pd.concat(frames, ignore_index=True)

    if df["case_id"].duplicated().any():
        raise ValueError("V6结果中存在重复病例，已停止合并。")

    for column in [
        "Dice",
        "roi_gt_coverage",
        "roi_fraction",
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

        if (
            not np.isfinite(df[column]).all()
            or not df[column].between(0, 1).all()
        ):
            raise ValueError(
                f"{column}应为0至1之间的有限小数。"
            )

    df = df.rename(columns={"Dice": "auto_dice"})

    return (
        df.sort_values(["fold", "case_id"]).reset_index(drop=True),
        missing_folds,
    )


def attach_primary(df, path):
    if not path.is_file():
        raise ValueError(f"未找到主实验结果：{path}")

    primary = pd.read_csv(path)

    required = {"model", "case_id", "dice"}

    if not required.issubset(primary.columns):
        raise ValueError(
            f"主实验CSV缺少列："
            f"{sorted(required - set(primary.columns))}"
        )

    primary["model"] = (
        primary["model"].astype(str).str.strip()
    )

    primary = primary.loc[
        primary["model"].isin([
            "Full MSAU-Net",
            "msau_unet3d_v2",
            "MSAU-Net",
        ])
    ].copy()

    primary["case_id"] = (
        primary["case_id"].astype(str).str.strip()
    )

    if (
        len(primary) != 15
        or primary["case_id"].nunique() != 15
    ):
        raise ValueError(
            "主实验完整MSAU-Net应包含15例唯一病例。"
        )

    primary["dice"] = pd.to_numeric(
        primary["dice"], errors="raise"
    )

    if (
        not np.isfinite(primary["dice"]).all()
        or not primary["dice"].between(0, 1).all()
    ):
        raise ValueError("主实验Dice数值无效。")

    keep = ["case_id", "dice"]

    if "fold" in primary.columns:
        primary = primary.rename(
            columns={"fold": "primary_fold"}
        )
        keep.append("primary_fold")

    primary = primary[keep].rename(
        columns={"dice": "primary_dice"}
    )

    merged = df.merge(
        primary,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    if merged["primary_dice"].isna().any():
        missing = merged.loc[
            merged["primary_dice"].isna(),
            "case_id",
        ].tolist()
        raise ValueError(
            f"以下V6病例无法匹配主实验：{missing}"
        )

    if "primary_fold" in merged.columns:
        primary_fold = pd.to_numeric(
            merged["primary_fold"], errors="raise"
        )

        if not primary_fold.eq(merged["fold"]).all():
            raise ValueError(
                "主实验与V6病例的外层测试折不一致。"
            )

    merged["dice_change_auto_minus_primary"] = (
        merged["auto_dice"] - merged["primary_dice"]
    )

    return merged


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#9BA6B0")

    ax.grid(
        axis="y",
        color="#E5E9ED",
        linewidth=0.7,
    )
    ax.set_axisbelow(True)


def marker_for_case(case_id):
    return "^" if case_id.startswith("Anev_") else "o"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--auto-root",
        type=Path,
        default=AUTO_ROOT,
    )

    parser.add_argument(
        "--primary-csv",
        type=Path,
        default=PRIMARY_CSV,
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
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

    configure_font(args.font)

    df, missing_folds = load_auto_results(
        args.auto_root.expanduser().resolve(),
        args.folds,
    )

    df = attach_primary(
        df,
        args.primary_csv.expanduser().resolve(),
    )

    colors = [
        FOLD_COLORS[int(fold)]
        for fold in df["fold"]
    ]

    coverage = df["roi_gt_coverage"].to_numpy() * 100
    roi_fraction = df["roi_fraction"].to_numpy() * 100

    positions = np.arange(len(df))
    case_labels = df["case_id"].tolist()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 12),
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.92,
        bottom=0.18,
        wspace=0.32,
        hspace=0.75,
    )

    # --------------------------------------------------------
    # (a) Reference foreground covered by candidate ROIs
    # --------------------------------------------------------

    ax = axes[0, 0]

    for i, row in df.iterrows():
        ax.scatter(
            i,
            coverage[i],
            s=75,
            marker=marker_for_case(row["case_id"]),
            color=colors[i],
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    ax.axhline(
        98,
        color="#9C6A56",
        linestyle=(0, (4, 3)),
        linewidth=1.2,
    )

    ax.set_title(
        "(a) ROI foreground coverage",
        loc="left",
        pad=14,
    )

    ax.set_ylabel("Reference foreground coverage (%)")

    lower = max(
        0,
        min(97.5, float(coverage.min()) - 0.5),
    )

    ax.set_ylim(lower, 100.6)
    ax.set_xticks(positions, case_labels)

    plt.setp(
        ax.get_xticklabels(),
        rotation=55,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_xlim(-0.6, len(df) - 0.4)
    style_axis(ax)

    # --------------------------------------------------------
    # (b) Candidate ROI union fraction
    # --------------------------------------------------------

    ax = axes[0, 1]

    ax.bar(
        positions,
        roi_fraction,
        width=0.58,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )

    ax.axhline(
        65,
        color="#9C6A56",
        linestyle=(0, (4, 3)),
        linewidth=1.2,
    )

    ax.set_title(
        "(b) ROI volume fraction",
        loc="left",
        pad=14,
    )

    ax.set_ylabel("ROI union / full-volume voxels (%)")
    ax.set_ylim(
        0,
        max(70, float(roi_fraction.max()) + 8),
    )

    ax.set_xticks(positions, case_labels)

    plt.setp(
        ax.get_xticklabels(),
        rotation=55,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_xlim(-0.6, len(df) - 0.4)
    style_axis(ax)

    # --------------------------------------------------------
    # (c) ROI coverage versus final segmentation Dice
    # --------------------------------------------------------

    ax = axes[1, 0]

    for i, row in df.iterrows():
        ax.scatter(
            coverage[i],
            row["auto_dice"],
            s=85,
            marker=marker_for_case(row["case_id"]),
            color=colors[i],
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

        # Explicitly identify empty predictions when Dice = 0.
        if row["auto_dice"] == 0:
            ax.annotate(
                row["case_id"],
                (
                    coverage[i],
                    row["auto_dice"],
                ),
                xytext=(5, 9 + 15 * (i % 2)),
                textcoords="offset points",
                fontsize=14,
            )

    ax.axvline(
        98,
        color="#9C6A56",
        linestyle=(0, (4, 3)),
        linewidth=1.2,
    )

    ax.set_title(
        "(c) Localization coverage and segmentation",
        loc="left",
        pad=14,
    )

    ax.set_xlabel("Reference foreground coverage (%)")
    ax.set_ylabel("Automatic full-volume Dice")

    ax.set_xlim(lower, 100.6)
    ax.set_ylim(-0.05, 1.05)
    style_axis(ax)

    # --------------------------------------------------------
    # (d) Matched cases under two evaluation workflows
    # --------------------------------------------------------

    ax = axes[1, 1]

    for i, row in df.iterrows():
        values = [
            row["primary_dice"],
            row["auto_dice"],
        ]

        ax.plot(
            [0, 1],
            values,
            color=colors[i],
            alpha=0.65,
            linewidth=1.1,
            zorder=2,
        )

        ax.scatter(
            [0, 1],
            values,
            s=60,
            marker=marker_for_case(row["case_id"]),
            color=colors[i],
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

    ax.set_xticks(
        [0, 1],
        [
            "Mask-guided ROI\ncropped-volume evaluation",
            "Automatic ROI\nfull-volume evaluation",
        ],
    )

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Dice")

    ax.set_title(
        "(d) Matched-case workflow comparison",
        loc="left",
        pad=14,
    )

    style_axis(ax)

    # --------------------------------------------------------
    # Legend and notes
    # --------------------------------------------------------

    completed_folds = sorted(
        df["fold"].astype(int).unique()
    )

    handles = [
        Line2D(
            [], [],
            marker="s",
            linestyle="",
            color=FOLD_COLORS[fold],
            markersize=8,
            label=f"Fold {fold}",
        )
        for fold in completed_folds
    ]

    handles.extend([
        Line2D(
            [], [],
            marker="^",
            linestyle="",
            color="#64717D",
            markersize=8,
            label="Anev",
        ),
        Line2D(
            [], [],
            marker="o",
            linestyle="",
            color="#64717D",
            markersize=8,
            label="Nat",
        ),
    ])

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.092),
        ncol=min(len(handles), 7),
        frameon=False,
    )

    fig.text(
        0.09,
        0.075,
        f"Available outer-test results: n = {len(df)}; "
        f"folds = {', '.join(map(str, completed_folds))}.",
        fontsize=14,
    )

    fig.text(
        0.09,
        0.050,
        "Dashed lines: development-set selection targets "
        "(coverage ≥98%; ROI fraction ≤65%), not test-set guarantees.",
        fontsize=14,
    )

    fig.text(
        0.09,
        0.025,
        "Panel (d) compares different evaluation domains; "
        "paired lines show matched cases, not an isolated localization effect.",
        fontsize=14,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    outdir = (
        args.outdir.expanduser().resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)

    base = outdir / "Figure5_automatic_ROI_analysis"

    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".pdf"),
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".svg"),
        facecolor="white",
    )

    df.to_csv(
        outdir / "Figure5_matched_case_results.csv",
        index=False,
    )

    plt.close(fig)

    print("=" * 70)
    print("Figure 5 completed")
    print(f"Available cases : {len(df)}")
    print(f"Available folds : {completed_folds}")
    print(f"Missing folds   : {missing_folds}")
    print("Font            : Times New Roman, 14 pt")
    print("PNG             : 600 dpi, 9600 × 7200 pixels")
    print(f"Output          : {outdir}")
    print("=" * 70)

    print(
        df[[
            "case_id",
            "fold",
            "roi_gt_coverage",
            "roi_fraction",
            "primary_dice",
            "auto_dice",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
