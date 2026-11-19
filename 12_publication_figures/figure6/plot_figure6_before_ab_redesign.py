from pathlib import Path
import argparse
import importlib.util
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FIGURE2_DIR = ROOT / "12_publication_figures/figure2"

if not (FIGURE2_DIR / "plot_figure2.py").is_file():
    raise SystemExit("请保留图2脚本，用于统一字体。")

sys.path.insert(0, str(FIGURE2_DIR))
from plot_figure2 import configure_font


MORPH_SOURCE = (
    ROOT
    / "10_morphology"
    / "compute_true_diameter_from_stl_vtp_v3.py"
)

SUMMARY_CSV = (
    ROOT
    / "9_results"
    / "true_centerline_diameter"
    / "AAA_true_centerline_diameter_v3.csv"
)

COLORS = {
    "mesh": "#A6C8D8",
    "centerline": "#D05F73",
    "section": "#8B67AB",
    "feret": "#CE8A38",
    "diameter": "#537FA3",
    "equivalent": "#60A39A",
    "area": "#8D70AD",
}


def import_morphology():
    if not MORPH_SOURCE.is_file():
        raise ValueError(
            f"未找到最终形态学计算源码：\n{MORPH_SOURCE}"
        )

    spec = importlib.util.spec_from_file_location(
        "figure6_morphology",
        MORPH_SOURCE,
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def resolve_path(value):
    path = Path(str(value)).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path


def select_case(csv_path, requested=None):
    if not csv_path.is_file():
        raise ValueError(
            f"未找到形态学汇总表：\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    required = {
        "case_id",
        "stl_file",
        "centerline_file",
        "max_feret_diameter_mm",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"汇总表缺少列：{sorted(required-set(df.columns))}"
        )

    df["case_id"] = (
        df["case_id"].astype(str).str.strip()
    )

    if "status" in df.columns:
        df = df.loc[df["status"].eq("OK")].copy()

    df["max_feret_diameter_mm"] = pd.to_numeric(
        df["max_feret_diameter_mm"],
        errors="raise",
    )

    df = df.loc[
        np.isfinite(df["max_feret_diameter_mm"])
        & df["max_feret_diameter_mm"].gt(0)
    ].copy()

    if df.empty:
        raise ValueError("汇总表没有有效形态学病例。")

    if df["case_id"].duplicated().any():
        raise ValueError("形态学汇总表存在重复病例。")

    if requested is not None:
        selected = df.loc[
            df["case_id"].eq(requested)
        ]

        if len(selected) != 1:
            raise ValueError(
                f"未找到有效病例：{requested}"
            )

        return selected.iloc[0], "User-specified case"

    median = df["max_feret_diameter_mm"].median()

    df["distance_to_median"] = (
        df["max_feret_diameter_mm"] - median
    ).abs()

    selected = df.sort_values(
        ["distance_to_median", "case_id"]
    ).iloc[0]

    return (
        selected,
        "Case nearest the cohort median maximum Feret diameter",
    )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#9BA6B0")

    ax.grid(
        color="#E5E9ED",
        linewidth=0.7,
    )
    ax.set_axisbelow(True)


def equal_3d_axes(ax, vertices):
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (lower + upper) / 2

    radius = max(
        float((upper - lower).max()) / 2,
        1.0,
    )

    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.set_box_aspect((1, 1, 1))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        default=None,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_CSV,
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

    row, selection_rule = select_case(
        args.summary.expanduser().resolve(),
        args.case,
    )

    case_id = str(row["case_id"])

    mesh_path = resolve_path(row["stl_file"])
    centerline_path = resolve_path(
        row["centerline_file"]
    )

    for path in [mesh_path, centerline_path]:
        if not path.is_file():
            raise ValueError(f"未找到文件：{path}")

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    morph = import_morphology()

    # Recalculate geometry for ONE case using the final V3
    # measurement functions. No network training is involved.
    section_dir = outdir / "cross_sections"

    measurement = morph.measure_case(
        case_id,
        mesh_path,
        centerline_path,
        1.0,
        True,
        section_dir,
    )

    sections_path = (
        section_dir / f"{case_id}_sections.csv"
    )

    if not sections_path.is_file():
        raise ValueError(
            f"未生成截面测量表：{sections_path}"
        )

    sections = pd.read_csv(sections_path)

    required = {
        "position_mm",
        "center_x_mm",
        "center_y_mm",
        "center_z_mm",
        "tangent_x",
        "tangent_y",
        "tangent_z",
        "feret_diameter_mm",
               "equivalent_diameter_mm",
        "cross_section_area_mm2",
    }

    if not required.issubset(sections.columns):
        raise ValueError(
            f"截面表缺少列："
            f"{sorted(required-set(sections.columns))}"
        )

    sections = sections.sort_values(
        "position_mm"
    ).reset_index(drop=True)

    # Match the maximum-diameter section in the saved data.
    best = sections.loc[
        sections["feret_diameter_mm"].idxmax()
    ]

    max_diameter = float(
        best["feret_diameter_mm"]
    )

    if not np.isclose(
        max_diameter,
        float(row["max_feret_diameter_mm"]),
        rtol=1e-4,
        atol=0.01,
    ):
        raise ValueError(
            "重新计算的最大径与最终汇总表不一致。\n"
            f"汇总表：{row['max_feret_diameter_mm']}\n"
            f"重新计算：{max_diameter}\n"
            "请先核对源网格、中心线和测量版本。"
        )

    center = best[[
        "center_x_mm",
        "center_y_mm",
        "center_z_mm",
    ]].to_numpy(dtype=float)

    tangent = best[[
        "tangent_x",
        "tangent_y",
        "tangent_z",
    ]].to_numpy(dtype=float)

    mesh = morph.load_mesh(mesh_path)

    centerline = morph.read_ordered_centerline(
        centerline_path,
        verbose=False,
    )

    loops = morph.mesh_section_loops(
        mesh,
        center,
        tangent,
    )

    contour = morph.choose_local_contour(
        loops,
        center,
        tangent,
    )

    if contour is None:
        raise ValueError(
            "最大径位置未获得有效截面轮廓。"
        )

    loop3d = np.asarray(
        contour["loop"],
        dtype=float,
    )

    xy = np.asarray(
        contour["xy"],
        dtype=float,
    )

    hull = ConvexHull(xy)
    hull_points = xy[hull.vertices]

    distances = squareform(
        pdist(hull_points)
    )

    endpoint_i, endpoint_j = np.unravel_index(
        distances.argmax(),
        distances.shape,
    )

    endpoints = hull_points[[
        endpoint_i, endpoint_j
    ]]

    # Closed copies are used for display only.
    loop3d_closed = np.vstack([
        loop3d, loop3d[0]
    ])

    xy_closed = np.vstack([
        xy, xy[0]
    ])

    # ========================================================
    # Figure layout
    # ========================================================

    fig = plt.figure(figsize=(14, 12))

    grid = fig.add_gridspec(
        2, 2,
        left=0.09,
        right=0.97,
        bottom=0.14,
        top=0.93,
        wspace=0.32,
        hspace=0.44,
    )

    ax3d = fig.add_subplot(
        grid[0, 0],
        projection="3d",
    )

    ax_section = fig.add_subplot(grid[0, 1])
    ax_diameter = fig.add_subplot(grid[1, 0])
    ax_area = fig.add_subplot(grid[1, 1])

    # --------------------------------------------------------
    # (a) Mesh, principal centerline and selected section
    # --------------------------------------------------------

    vertices = np.asarray(
        mesh.vertices,
        dtype=float,
    )

    faces = np.asarray(
        mesh.faces,
        dtype=int,
    )

    # Center the visualization without changing measurements.
    origin = vertices.mean(axis=0)

    surface = Poly3DCollection(
        vertices[faces] - origin,
        facecolor=COLORS["mesh"],
        edgecolor="none",
        alpha=0.18,
        rasterized=True,
    )

    ax3d.add_collection3d(surface)

    ax3d.plot(
        centerline[:, 0] - origin[0],
        centerline[:, 1] - origin[1],
        centerline[:, 2] - origin[2],
        color=COLORS["centerline"],
        linewidth=2.0,
        zorder=5,
    )

    ax3d.plot(
        loop3d_closed[:, 0] - origin[0],
        loop3d_closed[:, 1] - origin[1],
        loop3d_closed[:, 2] - origin[2],
        color=COLORS["section"],
        linewidth=2.3,
        zorder=6,
    )

    ax3d.scatter(
        [center[0]-origin[0]],
        [center[1]-origin[1]],
        [center[2]-origin[2]],
        color=COLORS["feret"],
        s=35,
        depthshade=False,
    )

    equal_3d_axes(
        ax3d,
        vertices-origin,
    )

    ax3d.view_init(
        elev=18,
        azim=-65,
    )

    ax3d.set_axis_off()

    ax3d.set_title(
        "(a) Surface mesh and principal centerline",
        loc="left",
        pad=12,
        fontsize=14,
    )

    # --------------------------------------------------------
    # (b) Maximum-Feret cross-section
    # --------------------------------------------------------

    ax_section.fill(
        xy_closed[:, 0],
        xy_closed[:, 1],
        color=COLORS["mesh"],
        alpha=0.35,
    )

    ax_section.plot(
        xy_closed[:, 0],
        xy_closed[:, 1],
        color=COLORS["section"],
        linewidth=1.8,
        label="Mesh intersection",
    )

    ax_section.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        color=COLORS["feret"],
        linewidth=2.2,
        marker="o",
        markersize=5,
        label=f"Maximum Feret: {max_diameter:.1f} mm",
    )

    ax_section.scatter(
        [0], [0],
        marker="+",
        s=65,
        color=COLORS["centerline"],
        label="Centerline point",
    )

    ax_section.set_aspect(
        "equal",
        adjustable="datalim",
    )

    ax_section.margins(0.18)

    ax_section.set_xlabel(
        "Local plane coordinate u (mm)"
    )

    ax_section.set_ylabel(
        "Local plane coordinate v (mm)"
    )

    ax_section.set_title(
        "(b) Section at maximum Feret diameter",
        loc="left",
        pad=12,
    )

    ax_section.legend(
        loc="best",
        frameon=False,
        fontsize=14,
    )

    style_axis(ax_section)

    # --------------------------------------------------------
    # (c) Diameter profiles
    # --------------------------------------------------------

    position = sections[
        "position_mm"
    ].to_numpy(dtype=float)

    selected_position = float(
        best["position_mm"]
    )

    ax_diameter.plot(
        position,
        sections["feret_diameter_mm"],
        color=COLORS["diameter"],
        linewidth=1.8,
        label="Maximum Feret diameter",
    )

    ax_diameter.plot(
        position,
        sections["equivalent_diameter_mm"],
        color=COLORS["equivalent"],
        linewidth=1.8,
        label="Area-equivalent diameter",
    )

    ax_diameter.axvline(
        selected_position,
        color=COLORS["section"],
        linestyle=(0, (4, 3)),
        linewidth=1.2,
    )

    ax_diameter.scatter(
        [selected_position],
        [max_diameter],
        color=COLORS["feret"],
        s=50,
        zorder=4,
    )

    ax_diameter.set_xlabel(
        "Distance along selected centerline (mm)"
    )

    ax_diameter.set_ylabel("Diameter (mm)")

    ax_diameter.set_title(
        "(c) Along-centerline diameter profiles",
        loc="left",
        pad=12,
    )

    ax_diameter.legend(
        loc="best",
        frameon=False,
        fontsize=14,
    )

    style_axis(ax_diameter)

    # --------------------------------------------------------
    # (d) Area profile
    # --------------------------------------------------------

    area = sections[
        "cross_section_area_mm2"
    ].to_numpy(dtype=float)

    ax_area.plot(
        position,
        area,
        color=COLORS["area"],
        linewidth=1.8,
    )

    ax_area.fill_between(
        position,
        0,
        area,
        color=COLORS["area"],
        alpha=0.12,
    )

    ax_area.axvline(
        selected_position,
        color=COLORS["section"],
        linestyle=(0, (4, 3)),
        linewidth=1.2,
    )

    ax_area.scatter(
        [selected_position],
        [float(best["cross_section_area_mm2"])],
        color=COLORS["feret"],
        s=50,
        zorder=4,
    )

    ax_area.set_xlabel(
        "Distance along selected centerline (mm)"
    )

    ax_area.set_ylabel(
        "Cross-sectional area (mm²)"
    )

    ax_area.set_title(
        "(d) Along-centerline area profile",
        loc="left",
        pad=12,
    )

    style_axis(ax_area)

    # --------------------------------------------------------
    # Figure notes
    # --------------------------------------------------------

    fig.text(
        0.09,
        0.085,
        f"Case: {case_id}. "
        "Pink: principal centerline; purple: selected section; "
        "orange: maximum Feret measurement.",
        fontsize=14,
    )

    fig.text(
        0.09,
        0.055,
        "Dashed lines identify the maximum-Feret section. "
        "Its associated area is not necessarily the maximum area.",
        fontsize=14,
    )

    fig.text(
        0.09,
        0.025,
        "Mesh extent follows the supplied geometry. "
        "Profiles connect valid measured sections without smoothing.",
        fontsize=14,
    )

    # --------------------------------------------------------
    # Save figure and measurement provenance
    # --------------------------------------------------------

    base = outdir / f"Figure6_morphology_{case_id}"

    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".pdf"),
        facecolor="white",
    )

    sections.to_csv(
        outdir / f"Figure6_{case_id}_section_data.csv",
        index=False,
    )

    pd.DataFrame([{
        **measurement,
        "case_selection_rule": selection_rule,
    }]).to_csv(
        outdir / f"Figure6_{case_id}_measurements.csv",
        index=False,
    )

    plt.close(fig)

    print("=" * 70)
    print("Figure 6 completed")
    print(f"Case      : {case_id}")
    print(f"Selection : {selection_rule}")
    print(f"Max Feret : {max_diameter:.3f} mm")
    print("Font      : Times New Roman, 14 pt")
    print("PNG       : 600 dpi, 8400 × 7200 pixels")
    print(f"Output    : {outdir}")
    print("=" * 70)


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
