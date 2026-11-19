#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
True centerline-orthogonal diameter measurement for AAA STL meshes.

Input:
    0_raw_data/meshes/meshes/AAA001.stl
    0_raw_data/centerlines/centerlines/AAA001/abdominal_aorta.vtp

Method:
    1. Read original abdominal-aorta centerline from VTP.
    2. Recover ordered centerline polyline.
    3. Resample centerline by arc length.
    4. Estimate local tangent.
    5. Intersect STL with plane perpendicular to local tangent.
    6. Select the closed contour associated with the centerline point.
    7. Project contour to the local 2-D plane.
    8. Compute:
         - maximum Feret diameter
         - cross-sectional area
         - equivalent diameter
    9. Search along centerline for true local maximum diameter.

No voxelization.
No skeletonization.
"""

import os
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh

from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull, distance
from shapely.geometry import Polygon


# ============================================================
# configuration
# ============================================================

DEFAULT_STL_ROOT = Path(
    "/home/ubuntu/AAA_CTA_Project/0_raw_data/meshes/meshes"
)

DEFAULT_CENTERLINE_ROOT = Path(
    "/home/ubuntu/AAA_CTA_Project/0_raw_data/centerlines/centerlines"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/home/ubuntu/AAA_CTA_Project/9_results/true_centerline_diameter"
)


# Centerline sampling interval.
# 1 mm is appropriate for searching local maximum diameter.
DEFAULT_STEP_MM = 1.0

# Do not sample extremely close to endpoints because perpendicular
# planes can intersect open/end surfaces.
END_MARGIN_MM = 5.0

# Minimum acceptable contour area.
MIN_AREA_MM2 = 5.0

# Maximum allowed distance between the selected contour and
# current centerline point.
MAX_CONTOUR_CENTER_DISTANCE_MM = 20.0


# ============================================================
# basic utilities
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n < 1e-12:
        raise ValueError("Zero-length vector")

    return v / n


def cumulative_length(points):
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return np.array([0.0])

    d = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1
    )

    return np.concatenate([
        [0.0],
        np.cumsum(d)
    ])


# ============================================================
# VTP centerline reading
# ============================================================

def read_ordered_centerline(vtp_path):
    """
    Read an ordered centerline from a VTP polyline.

    Priority:
        use VTK/PyVista line connectivity rather than raw point order.
    """

    poly = pv.read(str(vtp_path))

    if poly.n_points < 2:
        raise RuntimeError(
            f"Centerline has only {poly.n_points} points"
        )

    pts = np.asarray(poly.points, dtype=float)

    # --------------------------------------------------------
    # Recover lines from VTK connectivity
    # --------------------------------------------------------

    lines = np.asarray(poly.lines)

    polylines = []

    i = 0

    while i < len(lines):

        n = int(lines[i])

        ids = lines[i + 1:i + 1 + n]

        if n >= 2:
            polylines.append(ids)

        i += n + 1

    if len(polylines) == 0:

        # fallback: use point order
        ordered = pts.copy()

    else:

        # abdominal_aorta.vtp should normally contain one main line.
        # If several exist, choose the longest geometric polyline.

        best = None
        best_length = -1

        for ids in polylines:

            p = pts[ids]

            L = cumulative_length(p)[-1]

            if L > best_length:
                best_length = L
                best = p

        ordered = np.asarray(best, dtype=float)

    # --------------------------------------------------------
    # Remove consecutive duplicate points
    # --------------------------------------------------------

    keep = [0]

    for i in range(1, len(ordered)):

        if np.linalg.norm(
            ordered[i] - ordered[keep[-1]]
        ) > 1e-6:

            keep.append(i)

    ordered = ordered[keep]

    if len(ordered) < 3:
        raise RuntimeError(
            "Centerline contains fewer than 3 unique ordered points"
        )

    return ordered


# ============================================================
# centerline resampling
# ============================================================

def resample_centerline(points, step_mm=1.0):

    points = np.asarray(points, dtype=float)

    s = cumulative_length(points)

    total_length = float(s[-1])

    if total_length <= 0:
        raise RuntimeError("Invalid centerline length")

    # Number of samples
    n = max(
        int(np.ceil(total_length / step_mm)) + 1,
        3
    )

    s_new = np.linspace(
        0,
        total_length,
        n
    )

    fx = interp1d(
        s,
        points[:, 0],
        kind="linear"
    )

    fy = interp1d(
        s,
        points[:, 1],
        kind="linear"
    )

    fz = interp1d(
        s,
        points[:, 2],
        kind="linear"
    )

    new_points = np.column_stack([
        fx(s_new),
        fy(s_new),
        fz(s_new)
    ])

    return new_points, s_new


# ============================================================
# local tangent
# ============================================================

def local_tangent(points, idx, window=2):

    n = len(points)

    i0 = max(
        0,
        idx - window
    )

    i1 = min(
        n - 1,
        idx + window
    )

    if i1 == i0:
        raise RuntimeError(
            "Unable to estimate local tangent"
        )

    tangent = points[i1] - points[i0]

    return normalize(tangent)


# ============================================================
# local coordinate system
# ============================================================

def plane_basis(normal):

    n = normalize(normal)

    # Pick vector least parallel to normal
    axes = np.eye(3)

    dots = np.abs(
        axes @ n
    )

    ref = axes[
        np.argmin(dots)
    ]

    u = normalize(
        np.cross(n, ref)
    )

    v = normalize(
        np.cross(n, u)
    )

    return u, v


def project_to_plane(points, origin, normal):

    u, v = plane_basis(normal)

    q = np.asarray(points) - origin

    xy = np.column_stack([
        q @ u,
        q @ v
    ])

    return xy


# ============================================================
# geometry measurements
# ============================================================

def polygon_area_2d(points):

    if len(points) < 3:
        return np.nan

    try:

        hull = ConvexHull(points)

        hp = points[hull.vertices]

        poly = Polygon(hp)

        if not poly.is_valid:
            poly = poly.buffer(0)

        return float(poly.area)

    except Exception:

        return np.nan


def max_feret_diameter(points):

    if len(points) < 2:
        return np.nan

    try:

        hull = ConvexHull(points)

        hp = points[hull.vertices]

    except Exception:

        hp = points

    if len(hp) < 2:
        return np.nan

    D = distance.pdist(hp)

    if len(D) == 0:
        return np.nan

    return float(np.max(D))


# ============================================================
# STL-plane intersection
# ============================================================

def get_section_loops(mesh, origin, normal):
    """
    Intersect triangular mesh with a plane.

    Returns:
        list of Nx3 loops/polylines
    """

    section = mesh.section(
        plane_origin=origin,
        plane_normal=normal
    )

    if section is None:
        return []

    try:

        discrete = section.discrete

    except Exception:

        return []

    loops = []

    for curve in discrete:

        curve = np.asarray(
            curve,
            dtype=float
        )

        if len(curve) >= 3:
            loops.append(curve)

    return loops


def choose_centerline_loop(
    loops,
    center_point,
    normal
):
    """
    Choose the section contour corresponding to the aortic lumen
    around the current centerline point.

    For each contour:
        project into local plane
        estimate contour centroid
        select closest contour to centerline point
    """

    candidates = []

    for loop in loops:

        if len(loop) < 3:
            continue

        xy = project_to_plane(
            loop,
            center_point,
            normal
        )

        if len(xy) < 3:
            continue

        # Distance between plane origin and contour centroid
        centroid_xy = np.mean(
            xy,
            axis=0
        )

        center_dist = float(
            np.linalg.norm(
                centroid_xy
            )
        )

        area = polygon_area_2d(xy)

        if not np.isfinite(area):
            continue

        if area < MIN_AREA_MM2:
            continue

        candidates.append(
            (
                center_dist,
                area,
                loop,
                xy
            )
        )

    if len(candidates) == 0:
        return None

    # closest contour to centerline
    candidates.sort(
        key=lambda x: x[0]
    )

    best = candidates[0]

    if best[0] > MAX_CONTOUR_CENTER_DISTANCE_MM:
        return None

    return best


# ============================================================
# per-case measurement
# ============================================================

def measure_case(
    case_id,
    stl_path,
    centerline_path,
    step_mm=1.0,
    save_sections=False,
    section_dir=None
):

    # --------------------------------------------------------
    # STL
    # --------------------------------------------------------

    mesh = trimesh.load_mesh(
        str(stl_path),
        process=False
    )

    if isinstance(
        mesh,
        trimesh.Scene
    ):

        mesh = trimesh.util.concatenate(
            tuple(mesh.geometry.values())
        )

    if mesh.vertices.shape[0] == 0:
        raise RuntimeError(
            "Empty STL mesh"
        )

    # --------------------------------------------------------
    # Centerline
    # --------------------------------------------------------

    original_cl = read_ordered_centerline(
        centerline_path
    )

    original_length = float(
        cumulative_length(
            original_cl
        )[-1]
    )

    cl, s = resample_centerline(
        original_cl,
        step_mm=step_mm
    )

    if len(cl) < 5:
        raise RuntimeError(
            "Resampled centerline too short"
        )

    # --------------------------------------------------------
    # Search sections
    # --------------------------------------------------------

    records = []

    for i in range(len(cl)):

        position_mm = float(s[i])

        # Skip ends
        if position_mm < END_MARGIN_MM:
            continue

        if (
            original_length - position_mm
            < END_MARGIN_MM
        ):
            continue

        origin = cl[i]

        try:

            tangent = local_tangent(
                cl,
                i,
                window=2
            )

            loops = get_section_loops(
                mesh,
                origin,
                tangent
            )

            selected = choose_centerline_loop(
                loops,
                origin,
                tangent
            )

            if selected is None:
                continue

            center_dist, area, loop, xy = selected

            feret = max_feret_diameter(
                xy
            )

            if (
                not np.isfinite(feret)
                or feret <= 0
            ):
                continue

            equivalent_diameter = np.sqrt(
                4.0 * area / np.pi
            )

            records.append({
                "case_id": case_id,
                "centerline_index": i,
                "position_mm": position_mm,

                "center_x": origin[0],
                "center_y": origin[1],
                "center_z": origin[2],

                "tangent_x": tangent[0],
                "tangent_y": tangent[1],
                "tangent_z": tangent[2],

                "feret_diameter_mm": feret,

                "cross_section_area_mm2": area,

                "equivalent_diameter_mm":
                    equivalent_diameter,

                "contour_center_distance_mm":
                    center_dist,

                "n_contour_points":
                    len(loop)
            })

        except Exception:

            continue

    if len(records) == 0:

        raise RuntimeError(
            "No valid centerline-orthogonal sections found"
        )

    sec_df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Maximum Feret diameter
    # --------------------------------------------------------

    max_idx = sec_df[
        "feret_diameter_mm"
    ].idxmax()

    max_row = sec_df.loc[
        max_idx
    ]

    # --------------------------------------------------------
    # Morphological features
    # --------------------------------------------------------

    straight_length = float(
        np.linalg.norm(
            original_cl[-1]
            - original_cl[0]
        )
    )

    tortuosity = (
        original_length /
        straight_length
        if straight_length > 0
        else np.nan
    )

    summary = {

        "case_id":
            case_id,

        "stl_file":
            str(stl_path),

        "centerline_file":
            str(centerline_path),

        "original_centerline_points":
            len(original_cl),

        "resampled_centerline_points":
            len(cl),

        "centerline_length_mm":
            original_length,

        "straight_length_mm":
            straight_length,

        "tortuosity":
            tortuosity,

        "valid_sections":
            len(sec_df),

        "valid_section_fraction":
            len(sec_df) /
            max(
                len(cl) - 2,
                1
            ),

        "max_feret_diameter_mm":
            float(
                max_row[
                    "feret_diameter_mm"
                ]
            ),

        "max_section_area_mm2":
            float(
                max_row[
                    "cross_section_area_mm2"
                ]
            ),

        "max_section_equivalent_diameter_mm":
            float(
                max_row[
                    "equivalent_diameter_mm"
                ]
            ),

        "max_diameter_position_mm":
            float(
                max_row[
                    "position_mm"
                ]
            ),

        "max_diameter_center_x":
            float(
                max_row[
                    "center_x"
                ]
            ),

        "max_diameter_center_y":
            float(
                max_row[
                    "center_y"
                ]
            ),

        "max_diameter_center_z":
            float(
                max_row[
                    "center_z"
                ]
            ),

        "max_diameter_tangent_x":
            float(
                max_row[
                    "tangent_x"
                ]
            ),

        "max_diameter_tangent_y":
            float(
                max_row[
                    "tangent_y"
                ]
            ),

        "max_diameter_tangent_z":
            float(
                max_row[
                    "tangent_z"
                ]
            ),

        "mesh_volume_mm3":
            float(abs(mesh.volume)),

        "mesh_surface_area_mm2":
            float(mesh.area),

        "status":
            "OK"
    }

    # --------------------------------------------------------
    # optional section CSV
    # --------------------------------------------------------

    if save_sections:

        section_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        sec_df.to_csv(
            section_dir /
            f"{case_id}_sections.csv",
            index=False
        )

    return summary, sec_df


# ============================================================
# main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stl-root",
        type=str,
        default=str(
            DEFAULT_STL_ROOT
        )
    )

    parser.add_argument(
        "--centerline-root",
        type=str,
        default=str(
            DEFAULT_CENTERLINE_ROOT
        )
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(
            DEFAULT_OUTPUT_ROOT
        )
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None
    )

    parser.add_argument(
        "--step-mm",
        type=float,
        default=DEFAULT_STEP_MM
    )

    parser.add_argument(
        "--save-sections",
        action="store_true"
    )

    args = parser.parse_args()

    stl_root = Path(
        args.stl_root
    )

    centerline_root = Path(
        args.centerline_root
    )

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    section_root = (
        output_root /
        "cross_sections"
    )

    # --------------------------------------------------------
    # cases
    # --------------------------------------------------------

    if args.case is not None:

        cases = [
            args.case.upper()
        ]

    else:

        cases = sorted([
            p.stem
            for p in
            stl_root.glob("AAA*.stl")
        ])

    print(
        "=" * 80,
        flush=True
    )

    print(
        "TRUE CENTERLINE-ORTHOGONAL AAA DIAMETER",
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )

    print(
        f"STL root       : {stl_root}",
        flush=True
    )

    print(
        f"Centerline root: {centerline_root}",
        flush=True
    )

    print(
        f"Cases          : {len(cases)}",
        flush=True
    )

    print(
        f"Step           : {args.step_mm:.2f} mm",
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )

    results = []
    failures = []

    progress_file = (
        output_root /
        "progress.csv"
    )

    for num, case_id in enumerate(
        cases,
        start=1
    ):

        print(
            f"\n[{num:03d}/{len(cases):03d}] {case_id}",
            flush=True
        )

        stl_path = (
            stl_root /
            f"{case_id}.stl"
        )

        centerline_path = (
            centerline_root /
            case_id /
            "abdominal_aorta.vtp"
        )

        try:

            if not stl_path.exists():

                raise FileNotFoundError(
                    f"STL not found: {stl_path}"
                )

            if not centerline_path.exists():

                raise FileNotFoundError(
                    f"Centerline not found: {centerline_path}"
                )

            summary, _ = measure_case(
                case_id=case_id,
                stl_path=stl_path,
                centerline_path=centerline_path,
                step_mm=args.step_mm,
                save_sections=args.save_sections,
                section_dir=section_root
            )

            results.append(
                summary
            )

            print(
                "  Centerline length : "
                f"{summary['centerline_length_mm']:.2f} mm",
                flush=True
            )

            print(
                "  Valid sections    : "
                f"{summary['valid_sections']}",
                flush=True
            )

            print(
                "  MAX FERET         : "
                f"{summary['max_feret_diameter_mm']:.2f} mm",
                flush=True
            )

            print(
                "  Area              : "
                f"{summary['max_section_area_mm2']:.2f} mm^2",
                flush=True
            )

            print(
                "  Equivalent D      : "
                f"{summary['max_section_equivalent_diameter_mm']:.2f} mm",
                flush=True
            )

            print(
                "  Tortuosity        : "
                f"{summary['tortuosity']:.4f}",
                flush=True
            )

            status = "OK"

        except Exception as e:

            status = (
                f"FAILED: "
                f"{type(e).__name__}: {e}"
            )

            print(
                f"  {status}",
                flush=True
            )

            failures.append({
                "case_id":
                    case_id,

                "stl_file":
                    str(stl_path),

                "centerline_file":
                    str(centerline_path),

                "error":
                    status,

                "traceback":
                    traceback.format_exc()
            })

        # ----------------------------------------------------
        # progress
        # ----------------------------------------------------

        pd.DataFrame([{
            "case_id":
                case_id,

            "status":
                status
        }]).to_csv(
            progress_file,
            mode="a",
            header=not progress_file.exists(),
            index=False
        )

        # incremental save
        if len(results) > 0:

            pd.DataFrame(
                results
            ).to_csv(
                output_root /
                "AAA_true_centerline_diameter.csv",
                index=False
            )

        if len(failures) > 0:

            pd.DataFrame(
                failures
            ).to_csv(
                output_root /
                "AAA_true_centerline_diameter_failures.csv",
                index=False
            )

    # ========================================================
    # final summary
    # ========================================================

    print(
        "\n" + "=" * 80,
        flush=True
    )

    print(
        "FINISHED",
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )

    print(
        f"Successful: {len(results)}",
        flush=True
    )

    print(
        f"Failed    : {len(failures)}",
        flush=True
    )

    if len(results) > 0:

        df = pd.DataFrame(
            results
        )

        out_csv = (
            output_root /
            "AAA_true_centerline_diameter.csv"
        )

        df.to_csv(
            out_csv,
            index=False
        )

        print(
            f"\nSaved: {out_csv}",
            flush=True
        )

        print(
            "\nMaximum Feret diameter summary:",
            flush=True
        )

        print(
            df[
                "max_feret_diameter_mm"
            ].describe(),
            flush=True
        )

        print(
            "\nTortuosity summary:",
            flush=True
        )

        print(
            df[
                "tortuosity"
            ].describe(),
            flush=True
        )


if __name__ == "__main__":
    main()
