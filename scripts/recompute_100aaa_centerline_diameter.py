#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial.distance import pdist

PROJECT_ROOT = os.path.expanduser("~/AAA_CTA_Project")

MESH_ROOT = os.path.join(
    PROJECT_ROOT,
    "0_raw_data",
    "meshes",
    "meshes"
)

CENTERLINE_ROOT = os.path.join(
    PROJECT_ROOT,
    "0_raw_data",
    "centerlines",
    "centerlines"
)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def ordered_centerline_points(poly):
    points = np.asarray(poly.points)

    if poly.n_lines > 0:
        lines = poly.lines
        best = None
        idx = 0

        while idx < len(lines):
            n = int(lines[idx])
            ids = lines[idx + 1: idx + 1 + n]

            if best is None or len(ids) > len(best):
                best = ids.copy()

            idx += n + 1

        if best is not None:
            return points[best]

    return points


def remove_duplicate_points(points, eps=1e-5):
    if len(points) <= 1:
        return points

    keep = [0]

    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > eps:
            keep.append(i)

    return points[keep]


def smooth_centerline(points, window=5):
    if len(points) < window:
        return points.copy()

    half = window // 2
    output = np.zeros_like(points)

    for i in range(len(points)):
        left = max(0, i - half)
        right = min(len(points), i + half + 1)
        output[i] = points[left:right].mean(axis=0)

    return output


def tangent_at(points, i, radius=2):
    i0 = max(0, i - radius)
    i1 = min(len(points) - 1, i + radius)

    if i0 == i1:
        return None

    return normalize(points[i1] - points[i0])


def plane_basis(normal):
    n = normalize(normal)

    if n is None:
        return None, None

    helper = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(n, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    u = normalize(np.cross(n, helper))
    v = normalize(np.cross(n, u))

    return u, v


def project_to_plane(points3d, origin, normal):
    u, v = plane_basis(normal)

    if u is None:
        return None

    q = points3d - origin

    x = q @ u
    y = q @ v

    return np.column_stack([x, y])


def polygon_area(points2d):
    if points2d is None or len(points2d) < 3:
        return np.nan

    c = points2d.mean(axis=0)

    angles = np.arctan2(
        points2d[:, 1] - c[1],
        points2d[:, 0] - c[0]
    )

    ordered = points2d[np.argsort(angles)]

    x = ordered[:, 0]
    y = ordered[:, 1]

    return 0.5 * np.abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )


def max_feret(points2d):
    if points2d is None or len(points2d) < 2:
        return np.nan

    pts = np.unique(
        np.round(points2d, 5),
        axis=0
    )

    if len(pts) < 2:
        return np.nan

    return float(np.max(pdist(pts)))


def extract_section_points(mesh, origin, normal):
    try:
        section = mesh.slice(
            normal=normal,
            origin=origin
        )
    except Exception:
        return None

    if section is None:
        return None

    if section.n_points < 5:
        return None

    return np.asarray(section.points)


def nearest_section_component(points3d, origin):
    distances = np.linalg.norm(
        points3d - origin,
        axis=1
    )

    if len(distances) < 10:
        return points3d

    order = np.argsort(distances)
    d = distances[order]

    gaps = np.diff(d)

    if len(gaps) == 0:
        return points3d

    median_gap = np.median(gaps) + 1e-6

    candidates = np.where(
        gaps > max(5.0, 8.0 * median_gap)
    )[0]

    if len(candidates) > 0:
        cut = candidates[0] + 1

        if cut >= 5:
            return points3d[order[:cut]]

    return points3d


def measure_case(
    case_id,
    sample_step=1,
    tangent_radius=3,
    trim_fraction=0.02
):
    mesh_path = os.path.join(
        MESH_ROOT,
        f"{case_id}.stl"
    )

    centerline_path = os.path.join(
        CENTERLINE_ROOT,
        case_id,
        "abdominal_aorta.vtp"
    )

    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(mesh_path)

    if not os.path.isfile(centerline_path):
        raise FileNotFoundError(centerline_path)

    mesh = pv.read(mesh_path)
    centerline = pv.read(centerline_path)

    points = ordered_centerline_points(centerline)
    points = remove_duplicate_points(points)
    points = smooth_centerline(points, window=5)

    if len(points) < 5:
        raise RuntimeError(
            f"{case_id}: centerline too short"
        )

    trim = int(
        round(
            len(points) * trim_fraction
        )
    )

    start = max(
        tangent_radius,
        trim
    )

    stop = min(
        len(points) - tangent_radius,
        len(points) - trim
    )

    rows = []

    for i in range(
        start,
        stop,
        sample_step
    ):
        origin = points[i]

        tangent = tangent_at(
            points,
            i,
            radius=tangent_radius
        )

        if tangent is None:
            continue

        section_points = extract_section_points(
            mesh,
            origin,
            tangent
        )

        if section_points is None:
            continue

        section_points = nearest_section_component(
            section_points,
            origin
        )

        points2d = project_to_plane(
            section_points,
            origin,
            tangent
        )

        if points2d is None or len(points2d) < 5:
            continue

        area = polygon_area(points2d)
        diameter = max_feret(points2d)

        if not np.isfinite(area):
            continue

        if not np.isfinite(diameter):
            continue

        if diameter <= 1.0:
            continue

        if diameter > 200.0:
            continue

        equiv_diameter = (
            2.0 * np.sqrt(area / np.pi)
            if area > 0
            else np.nan
        )

        rows.append({
            "case_id": case_id,
            "centerline_index": i,
            "center_x": origin[0],
            "center_y": origin[1],
            "center_z": origin[2],
            "cross_section_area_mm2": area,
            "feret_diameter_mm": diameter,
            "equivalent_diameter_mm": equiv_diameter,
            "n_section_points": len(points2d)
        })

    sections = pd.DataFrame(rows)

    if len(sections) == 0:
        raise RuntimeError(
            f"{case_id}: no valid cross-sections"
        )

    idx_max = sections[
        "feret_diameter_mm"
    ].idxmax()

    peak = sections.loc[idx_max]

    surface_area = float(mesh.area)

    try:
        volume = float(mesh.volume)
    except Exception:
        volume = np.nan

    if (
        np.isfinite(volume)
        and volume > 0
        and surface_area > 0
    ):
        sphericity = (
            np.pi ** (1.0 / 3.0)
            * (6.0 * volume) ** (2.0 / 3.0)
            / surface_area
        )
    else:
        sphericity = np.nan

    max_diameter = float(
        peak["feret_diameter_mm"]
    )

    volume_diameter_ratio = (
        volume / max_diameter
        if np.isfinite(volume)
        and max_diameter > 0
        else np.nan
    )

    summary = {
        "case_id": case_id,
        "surface_area_mm2": surface_area,
        "volume_mm3": volume,
        "centerline_max_diameter_mm": max_diameter,
        "max_cross_section_area_mm2":
            float(peak["cross_section_area_mm2"]),
        "equivalent_diameter_at_max_mm":
            float(peak["equivalent_diameter_mm"]),
        "max_diameter_centerline_index":
            int(peak["centerline_index"]),
        "sphericity": float(sphericity),
        "volume_diameter_ratio":
            float(volume_diameter_ratio),
        "valid_cross_sections":
            int(len(sections))
    }

    return summary, sections


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        default=1
    )

    parser.add_argument(
        "--end",
        type=int,
        default=100
    )

    parser.add_argument(
        "--sample_step",
        type=int,
        default=1
    )

    parser.add_argument(
        "--tangent_radius",
        type=int,
        default=3
    )

    args = parser.parse_args()

    out_summary = os.path.join(
        PROJECT_ROOT,
        "9_results",
        "AAA100_centerline_morphology.csv"
    )

    out_sections = os.path.join(
        PROJECT_ROOT,
        "9_results",
        "AAA100_cross_section_measurements.csv"
    )

    summaries = []
    all_sections = []
    failures = []

    for number in range(
        args.start,
        args.end + 1
    ):
        case_id = f"AAA{number:03d}"

        print("\n" + "=" * 80)
        print(case_id)
        print("=" * 80)

        try:
            summary, sections = measure_case(
                case_id,
                sample_step=args.sample_step,
                tangent_radius=args.tangent_radius
            )

            summaries.append(summary)
            all_sections.append(sections)

            print(
                f"Maximum diameter: "
                f"{summary['centerline_max_diameter_mm']:.2f} mm"
            )

            print(
                f"Volume: "
                f"{summary['volume_mm3']:.2f} mm^3"
            )

            print(
                f"Surface area: "
                f"{summary['surface_area_mm2']:.2f} mm^2"
            )

        except Exception as e:
            warnings.warn(
                f"{case_id}: {e}"
            )

            failures.append({
                "case_id": case_id,
                "error": str(e)
            })

    summary_df = pd.DataFrame(
        summaries
    )

    sections_df = (
        pd.concat(
            all_sections,
            ignore_index=True
        )
        if all_sections
        else pd.DataFrame()
    )

    os.makedirs(
        os.path.join(
            PROJECT_ROOT,
            "9_results"
        ),
        exist_ok=True
    )

    summary_df.to_csv(
        out_summary,
        index=False
    )

    sections_df.to_csv(
        out_sections,
        index=False
    )

    failure_path = os.path.join(
        PROJECT_ROOT,
        "9_results",
        "AAA100_centerline_morphology_failures.csv"
    )

    pd.DataFrame(
        failures
    ).to_csv(
        failure_path,
        index=False
    )

    print("\n===== FINAL SUMMARY =====")

    if len(summary_df):
        print(
            summary_df[
                [
                    "case_id",
                    "centerline_max_diameter_mm",
                    "volume_mm3",
                    "surface_area_mm2",
                    "sphericity"
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(out_summary)
    print(out_sections)
    print(failure_path)


if __name__ == "__main__":
    main()
