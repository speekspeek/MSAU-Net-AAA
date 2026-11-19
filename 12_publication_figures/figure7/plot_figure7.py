from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram
from PIL import Image


# ============================================================
# Paths and configuration
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FIGURE2_DIR = ROOT / "12_publication_figures/figure2"

if not (FIGURE2_DIR / "plot_figure2.py").is_file():
    raise SystemExit("请保留图2脚本，用于统一字体。")

sys.path.insert(0, str(FIGURE2_DIR))

from plot_figure2 import configure_font


DEFAULT_CSV = (
    ROOT
    / "9_results/morphology_risk_v2"
    / "AAA100_morphology_risk_score_v2.csv"
)

FEATURES = [
    "max_feret_diameter_mm",
    "mesh_volume_mm3",
    "tortuosity",
    "sphericity",
    "max_section_area_mm2",
]

FEATURE_LABELS = [
    "Maximum Feret\ndiameter",
    "Mesh\nvolume",
    "Tortuosity",
    "Sphericity",
    "Area at maximum-\nFeret section",
]

SCORE_COLUMN = "morphology_risk_score_v2"
GROUP_COLUMN = "morphology_risk_group_v2"

GROUP_NAMES = [
    "Low",
    "Intermediate",
    "High",
]

GROUP_COLORS = [
    "#80B6AD",
    "#E4C47B",
    "#C97A87",
]

INK = "#2D3E4D"


# ============================================================
# Data validation
# ============================================================

def load_results(path):
    if not path.is_file():
        raise ValueError(
            f"未找到最终形态学评分表：\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "case_id",
        SCORE_COLUMN,
        GROUP_COLUMN,
        *FEATURES,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"结果表缺少列：{sorted(missing)}"
        )

    if df["case_id"].isna().any():
        raise ValueError("病例编号存在缺失值。")

    df["case_id"] = (
        df["case_id"].astype(str).str.strip()
    )

    if df["case_id"].duplicated().any():
        raise ValueError("存在重复病例。")

    if len(df) < 3:
        raise ValueError("至少需要3例有效病例。")

    for column in FEATURES + [SCORE_COLUMN]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if not np.isfinite(df[column]).all():
            raise ValueError(
                f"{column}存在缺失值或非有限值；"
                "未自动删除病例。"
            )

    if not df[SCORE_COLUMN].between(0, 100).all():
        raise ValueError("评分应在0至100之间。")

    if df[GROUP_COLUMN].isna().any():
        raise ValueError("形态分层存在缺失值。")

    df[GROUP_COLUMN] = (
        df[GROUP_COLUMN].astype(str).str.strip()
    )

    unknown = (
        set(df[GROUP_COLUMN]) - set(GROUP_NAMES)
    )

    if unknown:
        raise ValueError(
            f"无法识别的分层名称：{sorted(unknown)}"
        )

    return df.sort_values(
        "case_id"
    ).reset_index(drop=True)


def standardize_features(df):
    raw = df[FEATURES].copy()

    lower = raw.quantile(0.01)
    upper = raw.quantile(0.99)

    clipped = raw.clip(
        lower=lower,
        upper=upper,
        axis=1,
    )

    means = clipped.mean(axis=0)
    standard_deviations = clipped.std(
        axis=0,
        ddof=0,
    )

    constant = standard_deviations.le(1e-12)

    if constant.all():
        raise ValueError(
            "所有特征均无变异，无法进行聚类。"
        )

    safe_sd = standard_deviations.mask(
        constant,
        1.0,
    )

    z = (clipped - means) / safe_sd

    scaling = pd.DataFrame({
        "feature": FEATURES,
        "q01": lower.to_numpy(),
        "q99": upper.to_numpy(),
        "mean_after_winsorization": means.to_numpy(),
        "sd_after_winsorization":
            standard_deviations.to_numpy(),
        "constant_feature": constant.to_numpy(),
    })

    return z, scaling


# ============================================================
# Figure
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
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

    csv_path = args.csv.expanduser().resolve()

    df = load_results(csv_path)
    z, scaling = standardize_features(df)

    # Cluster morphology only.
    # Composite scores and strata are not clustering inputs.
    tree = linkage(
        z.to_numpy(dtype=float),
        method="ward",
        metric="euclidean",
        optimal_ordering=True,
    )

    fig = plt.figure(
        figsize=(16, 8.8),
        facecolor="white",
    )

    left = 0.22
    width = 0.66

    # Independent figure-level headings avoid title collisions.
    fig.text(
        left,
        0.925,
        "(a) Hierarchical clustering of cases",
        fontsize=14,
        color=INK,
        ha="left",
        va="center",
    )

    fig.text(
        left,
        0.695,
        "(b) Morphological feature profiles and exploratory strata",
        fontsize=14,
        color=INK,
        ha="left",
        va="center",
    )

    ax_tree = fig.add_axes(
        [left, 0.750, width, 0.145]
    )

    ax_group = fig.add_axes(
        [left, 0.650, width, 0.025]
    )

    ax_heat = fig.add_axes(
        [left, 0.290, width, 0.330]
    )

    ax_score = fig.add_axes(
        [left, 0.230, width, 0.030]
    )

    ax_zbar = fig.add_axes(
        [0.920, 0.310, 0.014, 0.290]
    )

    ax_scorebar = fig.add_axes(
        [0.650, 0.065, 0.230, 0.015]
    )

    # --------------------------------------------------------
    # (a) Hierarchical clustering
    # --------------------------------------------------------

    result = dendrogram(
        tree,
        ax=ax_tree,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#748C9F",
        link_color_func=lambda _: "#748C9F",
    )

    order = np.asarray(
        result["leaves"],
        dtype=int,
    )

    ordered = df.iloc[
        order
    ].reset_index(drop=True)

    z_ordered = z.iloc[
        order
    ].reset_index(drop=True)

    n_cases = len(ordered)

    ax_tree.set_ylabel(
        "Ward distance",
        fontsize=14,
        labelpad=10,
    )

    ax_tree.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_tree.locator_params(
        axis="y",
        nbins=3,
    )

    for side in ["top", "right", "bottom"]:
        ax_tree.spines[side].set_visible(False)

    ax_tree.spines["left"].set_color("#AAB7C1")

    # Dendrogram leaf centers: 5, 15, 25, ...
    extent_x = 10 * n_cases
    centers = np.arange(n_cases) * 10 + 5

    ax_tree.set_xlim(0, extent_x)

    # --------------------------------------------------------
    # Top legend
    # --------------------------------------------------------

    handles = []

    for name, color in zip(
        GROUP_NAMES,
        GROUP_COLORS,
    ):
        count = int(
            ordered[GROUP_COLUMN].eq(name).sum()
        )

        handles.append(
            Patch(
                facecolor=color,
                edgecolor="none",
                label=f"{name} (n = {count})",
            )
        )

    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(left, 0.995),
        ncol=3,
        frameon=False,
        fontsize=14,
        columnspacing=2.0,
        handlelength=1.5,
        borderaxespad=0,
    )

    # --------------------------------------------------------
    # (b) Exploratory stratum strip
    # --------------------------------------------------------

    group_codes = (
        ordered[GROUP_COLUMN]
        .map({
            name: index
            for index, name in enumerate(GROUP_NAMES)
        })
        .to_numpy(dtype=int)
    )

    ax_group.imshow(
        group_codes[None, :],
        cmap=ListedColormap(GROUP_COLORS),
        vmin=-0.5,
        vmax=2.5,
        aspect="auto",
        interpolation="nearest",
        extent=[0, extent_x, 0, 1],
    )

    ax_group.set_xlim(0, extent_x)
    ax_group.set_xticks([])
    ax_group.set_yticks([])

    ax_group.text(
        -0.025,
        0.5,
        "Morphology stratum",
        transform=ax_group.transAxes,
        ha="right",
        va="center",
        fontsize=14,
        color=INK,
    )

    for spine in ax_group.spines.values():
        spine.set_visible(False)

    # --------------------------------------------------------
    # Standardized feature heatmap
    # --------------------------------------------------------

    heat = ax_heat.imshow(
        z_ordered.to_numpy(dtype=float).T,
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=2.5,
        aspect="auto",
        interpolation="nearest",
        extent=[
            0,
            extent_x,
            len(FEATURES)-0.5,
            -0.5,
        ],
        rasterized=True,
    )

    ax_heat.set_xlim(0, extent_x)

    ax_heat.set_yticks(
        np.arange(len(FEATURES)),
        FEATURE_LABELS,
    )

    ax_heat.tick_params(
        axis="y",
        length=0,
        pad=12,
    )

    ax_heat.set_xticks([])

    for boundary in np.arange(
        0.5,
        len(FEATURES),
        1,
    ):
        ax_heat.axhline(
            boundary,
            color="white",
            linewidth=1.0,
        )

    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    zbar = fig.colorbar(
        heat,
        cax=ax_zbar,
        extend="both",
    )

    zbar.set_ticks([-2, 0, 2])

    zbar.ax.set_title(
        "Z",
        fontsize=14,
        pad=8,
    )

    zbar.outline.set_visible(False)

    # --------------------------------------------------------
    # Composite score strip
    # --------------------------------------------------------

    score_image = ax_score.imshow(
        ordered[SCORE_COLUMN].to_numpy()[None, :],
        cmap="cividis",
        norm=Normalize(0, 100),
        aspect="auto",
        interpolation="nearest",
        extent=[0, extent_x, 0, 1],
    )

    ax_score.set_xlim(0, extent_x)
    ax_score.set_yticks([])

    ax_score.text(
        -0.025,
        0.5,
        "Composite score",
        transform=ax_score.transAxes,
        ha="right",
        va="center",
        fontsize=14,
        color=INK,
    )

    # Label a subset of cases at readable 14-pt size.
    # All cases remain in the heatmap and ordered CSV.
    step = max(
        1,
        int(np.ceil(n_cases / 12)),
    )

    tick_indices = np.unique(
        np.r_[
            np.arange(0, n_cases, step),
            n_cases-1,
        ]
    ).astype(int)

    ax_score.set_xticks(
        centers[tick_indices],
        ordered["case_id"].iloc[
            tick_indices
        ].tolist(),
    )

    ax_score.tick_params(
        axis="x",
        length=0,
        pad=7,
    )

    plt.setp(
        ax_score.get_xticklabels(),
        rotation=90,
        ha="center",
        va="top",
    )

    for spine in ax_score.spines.values():
        spine.set_visible(False)

    # Horizontal score colorbar avoids overlapping 0/100 labels.
    scorebar = fig.colorbar(
        score_image,
        cax=ax_scorebar,
        orientation="horizontal",
    )

    scorebar.set_ticks([0, 50, 100])

    scorebar.set_label(
        "Composite morphology score",
        fontsize=14,
        labelpad=5,
    )

    scorebar.outline.set_visible(False)

    # --------------------------------------------------------
    # Save figures and audit data
    # --------------------------------------------------------

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base = (
        outdir
        / "Figure7_morphology_cluster_heatmap"
    )

    png_path = base.with_suffix(".png")

    fig.savefig(
        png_path,
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

    plt.close(fig)

    with Image.open(png_path) as image:
        image.convert("RGB").save(
            base.with_suffix(".tiff"),
            format="TIFF",
            compression="tiff_lzw",
            dpi=(600, 600),
        )

    ordered.insert(
        0,
        "display_order",
        np.arange(1, n_cases+1),
    )

    ordered.to_csv(
        outdir / "Figure7_ordered_case_data.csv",
        index=False,
    )

    z_export = z_ordered.copy()

    z_export.insert(
        0,
        "case_id",
        ordered["case_id"].to_numpy(),
    )

    z_export.to_csv(
        outdir / "Figure7_ordered_zscores.csv",
        index=False,
    )

    scaling.to_csv(
        outdir / "Figure7_feature_scaling.csv",
        index=False,
    )

    pd.DataFrame(
        tree,
        columns=[
            "cluster_1",
            "cluster_2",
            "ward_distance",
            "cluster_size",
        ],
    ).to_csv(
        outdir / "Figure7_linkage.csv",
        index=False,
    )

    print("=" * 70)
    print("Figure 7 completed")
    print(f"Cases  : {n_cases}")
    print("Font   : Times New Roman, 14 pt")
    print("PNG    : 600 dpi, 9600 × 5280 pixels")
    print("Formats: PNG, PDF, SVG, TIFF")
    print(f"Output : {outdir}")
    print("=" * 70)

    print(
        ordered[GROUP_COLUMN]
        .value_counts()
        .reindex(GROUP_NAMES, fill_value=0)
        .to_string()
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        KeyError,
        OSError,
        ImportError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
