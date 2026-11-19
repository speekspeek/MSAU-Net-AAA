from pathlib import Path
import argparse
import importlib.util
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(
    0,
    str(ROOT / "12_publication_figures/figure2"),
)

from plot_figure2 import configure_font


SUMMARY = (
    ROOT
    / "9_results/true_centerline_diameter"
    / "AAA_true_centerline_diameter_v3.csv"
)

SOURCE = (
    ROOT
    / "10_morphology"
    / "compute_true_diameter_from_stl_vtp_v3.py"
)

BLUE = "#6E9EB2"
PINK = "#B44E67"
PURPLE = "#80619F"
GOLD = "#C88D38"
TEAL = "#559E96"
INK = "#2D3E4D"


def import_measurement_code():
    spec = importlib.util.spec_from_file_location(
        "figure6_measurement",
        SOURCE,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def style_axis(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#A8B5BF")

    ax.tick_params(colors=INK)

    ax.grid(
        axis="y",
        color="#E6EBEF",
        linewidth=0.7,
    )

    ax.set_axisbelow(True)


def prepare_projection(mesh, centerline, loop, center):
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    origin = vertices.mean(axis=0)

    # Rigid display rotation only; measurements remain unchanged.
    _, vectors = np.linalg.eigh(
        np.cov((vertices - origin).T)
    )

    vertical = vectors[:, -1].copy()

    # Keep the displayed centerline endpoint direction consistent.
    if np.dot(centerline[-1] - centerline[0], vertical) < 0:
        vertical *= -1

    horizontal = vectors[:, -2]
    depth_axis = np.cross(horizontal, vertical)
    depth_axis /= np.linalg.norm(depth_axis)

    basis = np.column_stack([
        horizontal,
        vertical,
        depth_axis,
    ])

    # Slight rotation around the vertical axis for surface depth.
    angle = np.deg2rad(20)

    rotation = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])

    def project(points):
        return (
            (np.asarray(points) - origin)
            @ basis
            @ rotation
        )

    display_vertices = project(vertices)
    triangles = display_vertices[faces]

    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )

    lengths = np.linalg.norm(
        normals,
        axis=1,
        keepdims=True,
    )

    normals /= np.maximum(lengths, 1e-12)

    light = np.array([-0.45, 0.55, 0.80])
    light /= np.linalg.norm(light)

    brightness = (
        0.42
        + 0.58
        * np.clip(normals @ light, 0, 1)
    )

    base = np.array(
        matplotlib.colors.to_rgb(BLUE)
    )

    colors = np.clip(
        base[None, :] * brightness[:, None],
        0,
        1,
    )

    # Orthographic painter ordering: far faces first.
    order = np.argsort(
        triangles[:, :, 2].mean(axis=1)
    )

    return {
        "triangles": triangles[order, :, :2],
        "colors": colors[order],
        "vertices": display_vertices[:, :2],
        "centerline": project(centerline)[:, :2],
        "loop": project(loop)[:, :2],
        "center": project(np.asarray(center)[None, :])[0, :2],
    }


def draw_projected_mesh(ax, geometry, limits=None):
    surface = PolyCollection(
        geometry["triangles"],
        facecolors=geometry["colors"],
        edgecolors="none",
        linewidths=0,
        antialiased=False,
        rasterized=True,
        zorder=1,
    )

    ax.add_collection(surface)

    line = geometry["centerline"]

    # Centerline is deliberately shown as an overlay.
    ax.plot(
        line[:, 0],
        line[:, 1],
        color="white",
        linewidth=3.5,
        alpha=0.85,
        zorder=3,
    )

    ax.plot(
        line[:, 0],
        line[:, 1],
        color=PINK,
        linewidth=1.7,
        zorder=4,
    )

    loop = geometry["loop"]
    loop = np.vstack([loop, loop[0]])

    ax.plot(
        loop[:, 0],
        loop[:, 1],
        color=GOLD,
        linewidth=2.1,
        zorder=5,
    )

    if limits is None:
        vertices = geometry["vertices"]
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)

        span = np.maximum(upper-lower, 1.0)
        padding = span * 0.045

        ax.set_xlim(
            lower[0]-padding[0],
            upper[0]+padding[0],
        )

        ax.set_ylim(
            lower[1]-padding[1],
            upper[1]+padding[1],
        )
    else:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        default="AAA072",
    )

    parser.add_argument(
        "--font",
        default=(
            "/usr/share/fonts/truetype/"
            "msttcorefonts/times.ttf"
        ),
    )

    args = parser.parse_args()
    configure_font(args.font)

    outdir = HERE / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    if not SUMMARY.is_file():
        raise ValueError(f"未找到汇总表：{SUMMARY}")

    table = pd.read_csv(SUMMARY)
    table["case_id"] = table["case_id"].astype(str)

    selected = table.loc[
        table["case_id"].eq(args.case)
    ]

    if len(selected) != 1:
        raise ValueError(
            f"汇总表中未找到唯一病例：{args.case}"
        )

    row = selected.iloc[0]

    mesh_path = resolve_path(row["stl_file"])
    centerline_path = resolve_path(row["centerline_file"])

    for path in [mesh_path, centerline_path]:
        if not path.is_file():
            raise ValueError(f"未找到文件：{path}")

    morph = import_measurement_code()

    section_candidates = [
        outdir / "cross_sections" / f"{args.case}_sections.csv",
        SUMMARY.parent / "cross_sections" / f"{args.case}_sections.csv",
    ]

    section_path = next(
        (
            path
            for path in section_candidates
            if path.is_file()
        ),
        None,
    )

    if section_path is None:
        section_dir = outdir / "cross_sections"

        morph.measure_case(
            args.case,
            mesh_path,
            centerline_path,
            1.0,
            True,
            section_dir,
        )

        section_path = (
            section_dir / f"{args.case}_sections.csv"
        )

    sections = pd.read_csv(section_path)

    sections = sections.sort_values(
        "position_mm"
    ).reset_index(drop=True)

    best = sections.loc[
        sections["feret_diameter_mm"].idxmax()
    ]

    diameter = float(best["feret_diameter_mm"])
    area_at_diameter = float(
        best["cross_section_area_mm2"]
    )
    equivalent = float(
        best["equivalent_diameter_mm"]
    )
    selected_position = float(best["position_mm"])

    if not np.isclose(
        diameter,
        float(row["max_feret_diameter_mm"]),
        atol=0.01,
        rtol=1e-4,
    ):
        raise ValueError(
            "截面表与最终汇总表的最大径不一致，"
            "请核对测量版本。"
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
        raise ValueError("最大径位置没有有效截面。")

    xy = np.asarray(contour["xy"], dtype=float)
    loop = np.asarray(contour["loop"], dtype=float)

    hull_points = xy[ConvexHull(xy).vertices]

    distances = squareform(pdist(hull_points))

    i, j = np.unravel_index(
        distances.argmax(),
        distances.shape,
    )

    endpoints = hull_points[[i, j]]

    geometry = prepare_projection(
        mesh,
        centerline,
        loop,
        center,
    )

    # ========================================================
    # Layout: full geometry + enlarged detail, cross-section,
    # and two measurement profiles.
    # ========================================================

    fig = plt.figure(
        figsize=(14, 12),
        facecolor="white",
    )

    ax_full = fig.add_axes(
        [0.065, 0.565, 0.18, 0.33]
    )

    ax_detail = fig.add_axes(
        [0.275, 0.610, 0.20, 0.245]
    )

    ax_section = fig.add_axes(
        [0.595, 0.605, 0.34, 0.29]
    )

    ax_info = fig.add_axes(
        [0.565, 0.535, 0.40, 0.055]
    )

    ax_diameter = fig.add_axes(
        [0.085, 0.185, 0.37, 0.255]
    )

    ax_area = fig.add_axes(
        [0.585, 0.185, 0.37, 0.255]
    )

    fig.text(
        0.065,
        0.945,
        "(a) Surface geometry and enlarged detail",
        fontsize=14,
        color=INK,
    )

    fig.text(
        0.565,
        0.945,
        "(b) Maximum-diameter cross-section",
        fontsize=14,
        color=INK,
    )

    # --------------------------------------------------------
    # (a) Full geometry and enlarged aneurysmal region
    # --------------------------------------------------------

    draw_projected_mesh(
        ax_full,
        geometry,
    )

    projected_center = geometry["center"]
    half_width = diameter * 0.72

    zoom_limits = (
        projected_center[0]-half_width,
        projected_center[0]+half_width,
        projected_center[1]-half_width,
        projected_center[1]+half_width,
    )

    draw_projected_mesh(
        ax_detail,
        geometry,
        zoom_limits,
    )

    ax_full.add_patch(
        Rectangle(
            (
                zoom_limits[0],
                zoom_limits[2],
            ),
            2*half_width,
            2*half_width,
            fill=False,
            edgecolor="#8D9CA7",
            linestyle=(0, (3, 3)),
            linewidth=0.9,
            zorder=6,
        )
    )

    connection = ConnectionPatch(
        xyA=(
            zoom_limits[1],
            projected_center[1],
        ),
        coordsA="data",
        axesA=ax_full,
        xyB=(
            zoom_limits[0],
            projected_center[1],
        ),
        coordsB="data",
        axesB=ax_detail,
        color="#AAB6BF",
        linewidth=0.8,
        linestyle=(0, (3, 3)),
        clip_on=False,
    )

    fig.add_artist(connection)

    ax_detail.set_title(
        "Maximum-diameter region",
        fontsize=14,
        pad=12,
        color=INK,
    )

    fig.text(
        0.065,
        0.535,
        "Pink: centerline overlay   "
        "Gold: selected section",
        fontsize=14,
        color=INK,
    )

    # --------------------------------------------------------
    # (b) Clean cross-section without an internal legend
    # --------------------------------------------------------

    closed = np.vstack([xy, xy[0]])

    ax_section.fill(
        closed[:, 0],
        closed[:, 1],
        color="#DEEBF1",
        alpha=0.95,
    )

    ax_section.plot(
        closed[:, 0],
        closed[:, 1],
        color=BLUE,
        linewidth=1.8,
    )

    ax_section.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        color=GOLD,
        linewidth=2.2,
        marker="o",
        markersize=4,
    )

    ax_section.scatter(
        [0],
        [0],
        marker="+",
        color=PINK,
        linewidths=1.6,
        s=75,
        zorder=5,
    )

    midpoint = endpoints.mean(axis=0)

    ax_section.annotate(
        f"{diameter:.1f} mm",
        midpoint,
        xytext=(0, 15),
        textcoords="offset points",
        ha="center",
        fontsize=14,
        color="#936024",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.95,
            "pad": 3,
        },
    )

    lower = xy.min(axis=0)
    upper = xy.max(axis=0)
    section_center = (lower+upper)/2
    radius = float((upper-lower).max())*0.60

    ax_section.set_xlim(
        section_center[0]-radius,
        section_center[0]+radius,
    )
    ax_section.set_ylim(
        section_center[1]-radius,
        section_center[1]+radius,
    )

    ax_section.set_aspect(
        "equal",
        adjustable="box",
    )

    ax_section.set_xlabel("u (mm)", labelpad=5)
    ax_section.set_ylabel("v (mm)", labelpad=5)
    ax_section.locator_params(axis="both", nbins=5)

    style_axis(ax_section)
    ax_section.grid(False)

    ax_info.set_axis_off()

    ax_info.text(
        0.5,
        0.72,
        f"Area at maximum diameter: "
        f"{area_at_diameter:.1f} mm²",
        ha="center",
        fontsize=14,
        color=INK,
    )

    ax_info.text(
        0.5,
        0.12,
        f"Area-equivalent diameter: "
        f"{equivalent:.1f} mm",
        ha="center",
        fontsize=14,
        color=INK,
    )

    # --------------------------------------------------------
    # (c) Diameter profiles
    # --------------------------------------------------------

    position = sections[
        "position_mm"
    ].to_numpy(dtype=float)

    ax_diameter.plot(
        position,
        sections["feret_diameter_mm"],
        color="#537FA3",
        linewidth=1.8,
        label="Maximum Feret",
    )

    ax_diameter.plot(
        position,
        sections["equivalent_diameter_mm"],
        color=TEAL,
        linewidth=1.8,
        label="Area-equivalent",
    )

    ax_diameter.axvline(
        selected_position,
        color=PURPLE,
        linewidth=1.1,
        linestyle=(0, (4, 3)),
    )

    ax_diameter.scatter(
        [selected_position],
        [diameter],
        color=GOLD,
        s=45,
        zorder=4,
    )

    ax_diameter.set_title(
        "(c) Diameter profiles",
        loc="left",
        pad=14,
        fontsize=14,
    )

    ax_diameter.set_xlabel(
        "Distance along centerline (mm)"
    )
    ax_diameter.set_ylabel("Diameter (mm)")

    ax_diameter.legend(
        frameon=False,
        loc="upper left",
        fontsize=14,
    )

    style_axis(ax_diameter)

    # --------------------------------------------------------
    # (d) Area profile
    # --------------------------------------------------------

    areas = sections[
        "cross_section_area_mm2"
    ].to_numpy(dtype=float)

    ax_area.plot(
        position,
        areas,
        color=PURPLE,
        linewidth=1.8,
    )

    ax_area.fill_between(
        position,
        0,
        areas,
        color=PURPLE,
        alpha=0.10,
    )

    ax_area.axvline(
        selected_position,
        color=PURPLE,
        linewidth=1.1,
        linestyle=(0, (4, 3)),
    )

    ax_area.scatter(
        [selected_position],
        [area_at_diameter],
        color=GOLD,
        s=45,
        zorder=4,
    )

    ax_area.set_title(
        "(d) Cross-sectional area profile",
        loc="left",
        pad=14,
        fontsize=14,
    )

    ax_area.set_xlabel(
        "Distance along centerline (mm)"
    )
    ax_area.set_ylabel("Area (mm²)")

    style_axis(ax_area)

    fig.text(
        0.085,
        0.100,
        f"Case: {args.case}. "
        "Panel (a): shaded orthographic projection "
        "with centerline and section overlays.",
        fontsize=14,
        color=INK,
    )

    fig.text(
        0.085,
        0.070,
        "Dashed lines in (c–d) identify the "
        "maximum-Feret section; the associated "
        "area is not independently maximized.",
        fontsize=14,
        color=INK,
    )

    fig.text(
        0.085,
        0.040,
        "Rigid display rotation preserves geometry. "
        "Profiles connect valid sections without smoothing.",
        fontsize=14,
        color=INK,
    )

    base = outdir / f"Figure6_morphology_{args.case}"

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
        outdir / f"Figure6_{args.case}_section_data.csv",
        index=False,
    )

    plt.close(fig)

    print("=" * 65)
    print("Figure 6 completed")
    print(f"Case   : {args.case}")
    print("Font   : Times New Roman, 14 pt")
    print("PNG    : 600 dpi, 8400 × 7200 pixels")
    print(f"Output : {outdir}")
    print("=" * 65)


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
