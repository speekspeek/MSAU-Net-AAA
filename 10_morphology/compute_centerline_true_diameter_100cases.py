#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Centerline-corrected local maximum cross-sectional diameter.

Pipeline
--------
STL
 -> mesh repair
 -> voxelization
 -> 3-D skeletonization
 -> skeleton graph
 -> longest centerline path
 -> centerline smoothing
 -> local tangent
 -> perpendicular cutting plane
 -> mesh-plane intersection
 -> closed cross-sectional contour
 -> maximum Feret diameter
 -> search along centerline

IMPORTANT
---------
This script does NOT use axial-slice diameter.

Diameter definition:
Maximum Feret diameter of a valid closed cross-sectional contour
on a plane locally perpendicular to the estimated vessel centerline.

Outputs are kept separate from previous morphology results.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import trimesh
import networkx as nx

from scipy.ndimage import (
    binary_fill_holes,
    binary_closing,
    binary_opening,
    label,
    gaussian_filter1d,
)

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from skimage.morphology import skeletonize

warnings.filterwarnings("ignore")


# ============================================================
# Project paths
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

OUT_ROOT = (
    ROOT /
    "9_results/centerline_morphology"
)

CENTERLINE_DIR = (
    OUT_ROOT /
    "centerlines"
)

CROSS_SECTION_DIR = (
    OUT_ROOT /
    "cross_sections"
)

QC_DIR = (
    OUT_ROOT /
    "qc"
)

for p in [
    OUT_ROOT,
    CENTERLINE_DIR,
    CROSS_SECTION_DIR,
    QC_DIR,
]:
    p.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Parameters
# ============================================================

# Centerline voxelization resolution.
# 1 mm is a reasonable starting point for AAA geometry.
VOXEL_PITCH_MM = 1.0

# Do not measure immediately at terminal centerline points.
END_MARGIN_MM = 5.0

# Sampling interval along smoothed centerline.
SECTION_INTERVAL_MM = 1.0

# Local tangent estimation radius.
TANGENT_WINDOW_MM = 5.0

# Reject implausibly tiny cross sections.
MIN_SECTION_AREA_MM2 = 20.0

# Reject very short intersection contours.
MIN_SECTION_POINTS = 8

# Numerical tolerance.
EPS = 1e-8


# ============================================================
# STL discovery
# ============================================================

def discover_stl_files(root):

    files = []

    for ext in [
        "*.stl",
        "*.STL",
    ]:
        files.extend(
            root.rglob(ext)
        )

    files = sorted(
        set(files)
    )

    # Do not accidentally read generated output.
    files = [
        p for p in files
        if "9_results/centerline_morphology"
        not in str(p)
    ]

    return files


# ============================================================
# Case ID
# ============================================================

def get_case_id(path):

    name = path.stem

    # Keep original file stem as case ID.
    # This avoids destructive assumptions about naming.
    return name


# ============================================================
# Mesh loading / repair
# ============================================================

def load_mesh(path):

    obj = trimesh.load(
        str(path),
        force="mesh",
        process=False
    )

    if isinstance(
        obj,
        trimesh.Scene
    ):

        geometries = [
            g
            for g in obj.geometry.values()
            if isinstance(
                g,
                trimesh.Trimesh
            )
        ]

        if len(geometries) == 0:
            raise RuntimeError(
                "No mesh geometry in STL."
            )

        mesh = trimesh.util.concatenate(
            geometries
        )

    else:
        mesh = obj


    if (
        mesh.vertices is None
        or
        len(mesh.vertices) == 0
    ):
        raise RuntimeError(
            "Empty mesh."
        )


    mesh = mesh.copy()


    # Basic cleanup
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        trimesh.repair.fix_normals(
            mesh
        )
    except Exception:
        pass


    return mesh


# ============================================================
# Keep largest connected mesh component
# ============================================================

def keep_largest_component(mesh):

    try:

        parts = mesh.split(
            only_watertight=False
        )

        if len(parts) <= 1:
            return mesh

        part = max(
            parts,
            key=lambda x: abs(
                x.volume
            )
            if x.is_volume
            else x.area
        )

        return part.copy()

    except Exception:

        return mesh


# ============================================================
# Mesh geometry
# ============================================================

def mesh_morphology(mesh):

    area = float(
        mesh.area
    )


    try:
        volume = float(
            abs(mesh.volume)
        )
    except Exception:
        volume = np.nan


    if (
        np.isfinite(volume)
        and
        volume > 0
        and
        area > 0
    ):

        sphericity = (
            np.pi ** (1.0 / 3.0)
            *
            (6.0 * volume) ** (2.0 / 3.0)
            /
            area
        )

    else:
        sphericity = np.nan


    return (
        volume,
        area,
        float(sphericity)
        if np.isfinite(sphericity)
        else np.nan
    )


# ============================================================
# Voxelize STL for centerline extraction
# ============================================================

def voxelize_mesh(
        mesh,
        pitch
):

    vox = mesh.voxelized(
        pitch=pitch
    )

    try:
        vox = vox.fill()
    except Exception:
        pass


    matrix = np.asarray(
        vox.matrix,
        dtype=bool
    )


    if matrix.ndim != 3:
        raise RuntimeError(
            f"Unexpected voxel shape: {matrix.shape}"
        )


    # Clean small voxelization defects.
    matrix = binary_fill_holes(
        matrix
    )

    matrix = binary_closing(
        matrix,
        iterations=1
    )

    # Keep largest connected voxel component.
    cc, n = label(
        matrix
    )

    if n > 1:

        counts = np.bincount(
            cc.ravel()
        )

        counts[0] = 0

        largest = np.argmax(
            counts
        )

        matrix = (
            cc == largest
        )


    return (
        matrix,
        vox
    )


# ============================================================
# Skeleton graph
# ============================================================

NEIGHBORS_26 = np.array(
    [
        [i, j, k]
        for i in [-1, 0, 1]
        for j in [-1, 0, 1]
        for k in [-1, 0, 1]
        if not (
            i == 0
            and
            j == 0
            and
            k == 0
        )
    ],
    dtype=int
)


def skeleton_graph(
        skeleton,
        pitch
):

    coords = np.argwhere(
        skeleton
    )

    if len(coords) < 2:
        raise RuntimeError(
            "Skeleton contains fewer than 2 voxels."
        )


    coord_to_id = {
        tuple(c): i
        for i, c in enumerate(coords)
    }


    G = nx.Graph()


    for i, c in enumerate(coords):

        G.add_node(
            i,
            coord=c
        )


    shape = np.array(
        skeleton.shape
    )


    for i, c in enumerate(coords):

        for offset in NEIGHBORS_26:

            nb = (
                c + offset
            )

            if np.any(
                nb < 0
            ):
                continue

            if np.any(
                nb >= shape
            ):
                continue

            j = coord_to_id.get(
                tuple(nb)
            )

            if j is None:
                continue

            if j <= i:
                continue

            distance = float(
                np.linalg.norm(
                    offset.astype(float)
                    *
                    pitch
                )
            )

            G.add_edge(
                i,
                j,
                weight=distance
            )


    return (
        G,
        coords
    )


# ============================================================
# Prune tiny skeleton branches
# ============================================================

def prune_short_branches(
        G,
        min_length_mm=5.0
):

    G = G.copy()

    changed = True


    while changed:

        changed = False

        endpoints = [
            n
            for n in G.nodes
            if G.degree[n] == 1
        ]


        for endpoint in endpoints:

            if endpoint not in G:
                continue

            path = [
                endpoint
            ]

            length = 0.0

            prev = None
            current = endpoint


            while True:

                nbrs = [
                    n
                    for n in G.neighbors(
                        current
                    )
                    if n != prev
                ]

                if len(nbrs) == 0:
                    break

                nxt = nbrs[0]

                length += G[
                    current
                ][
                    nxt
                ][
                    "weight"
                ]

                path.append(
                    nxt
                )

                prev = current
                current = nxt


                if G.degree[
                    current
                ] != 2:
                    break


            # Remove short branch ending at junction.
            if (
                length < min_length_mm
                and
                len(path) > 1
                and
                G.degree[
                    current
                ] >= 3
            ):

                for node in path[:-1]:

                    if node in G:
                        G.remove_node(
                            node
                        )

                changed = True


    return G


# ============================================================
# Longest graph path
# ============================================================

def farthest_node(
        G,
        source
):

    lengths = nx.single_source_dijkstra_path_length(
        G,
        source,
        weight="weight"
    )

    return max(
        lengths,
        key=lengths.get
    )


def extract_longest_path(G):

    if len(G) == 0:
        raise RuntimeError(
            "Empty skeleton graph."
        )


    # Keep largest graph component.
    components = list(
        nx.connected_components(
            G
        )
    )

    largest_nodes = max(
        components,
        key=len
    )

    H = G.subgraph(
        largest_nodes
    ).copy()


    endpoints = [
        n
        for n in H.nodes
        if H.degree[n] == 1
    ]


    if len(endpoints) >= 2:

        start = endpoints[0]

    else:

        start = next(
            iter(H.nodes)
        )


    p1 = farthest_node(
        H,
        start
    )

    p2 = farthest_node(
        H,
        p1
    )


    path = nx.shortest_path(
        H,
        p1,
        p2,
        weight="weight"
    )


    return (
        H,
        path
    )


# ============================================================
# Voxel index -> world coordinates
# ============================================================

def voxel_indices_to_world(
        vox,
        indices
):

    indices = np.asarray(
        indices,
        dtype=float
    )


    # Trimesh VoxelGrid transformation is authoritative.
    points = trimesh.transform_points(
        indices,
        vox.transform
    )

    return points


# ============================================================
# Centerline smoothing
# ============================================================

def cumulative_distance(points):

    if len(points) < 2:
        return np.array(
            [0.0]
        )

    segment = np.linalg.norm(
        np.diff(
            points,
            axis=0
        ),
        axis=1
    )

    return np.concatenate(
        [
            [0.0],
            np.cumsum(
                segment
            )
        ]
    )


def resample_polyline(
        points,
        interval_mm
):

    s = cumulative_distance(
        points
    )

    total = s[-1]

    if total <= 0:
        raise RuntimeError(
            "Zero-length centerline."
        )


    target = np.arange(
        0,
        total + interval_mm * 0.5,
        interval_mm
    )

    target = np.clip(
        target,
        0,
        total
    )


    result = np.zeros(
        (
            len(target),
            3
        ),
        dtype=float
    )


    for dim in range(3):

        result[
            :,
            dim
        ] = np.interp(
            target,
            s,
            points[:, dim]
        )


    return (
        result,
        target
    )


def smooth_centerline(
        points,
        interval_mm
):

    resampled, s = resample_polyline(
        points,
        interval_mm
    )


    # Gaussian smoothing in physical units.
    sigma_mm = 2.0

    sigma_samples = max(
        sigma_mm /
        interval_mm,
        0.5
    )


    smooth = np.zeros_like(
        resampled
    )


    for dim in range(3):

        smooth[
            :,
            dim
        ] = gaussian_filter1d(
            resampled[:, dim],
            sigma=sigma_samples,
            mode="nearest"
        )


    return (
        smooth,
        s
    )


# ============================================================
# Local tangent
# ============================================================

def local_tangent(
        centerline,
        index,
        interval_mm,
        window_mm
):

    half = max(
        int(
            round(
                window_mm /
                interval_mm
            )
        ),
        1
    )


    i0 = max(
        0,
        index - half
    )

    i1 = min(
        len(centerline) - 1,
        index + half
    )


    tangent = (
        centerline[i1]
        -
        centerline[i0]
    )


    norm = np.linalg.norm(
        tangent
    )


    if norm < EPS:
        return None


    return (
        tangent /
        norm
    )


# ============================================================
# Plane basis
# ============================================================

def plane_basis(normal):

    normal = np.asarray(
        normal,
        dtype=float
    )

    normal /= np.linalg.norm(
        normal
    )


    # Pick axis least parallel to normal.
    axes = np.eye(3)

    dots = np.abs(
        axes @ normal
    )

    ref = axes[
        np.argmin(dots)
    ]


    u = np.cross(
        normal,
        ref
    )

    u /= np.linalg.norm(
        u
    )


    v = np.cross(
        normal,
        u
    )

    v /= np.linalg.norm(
        v
    )


    return (
        u,
        v
    )


# ============================================================
# Extract closed mesh-plane section
# ============================================================

def get_section_loops(
        mesh,
        origin,
        normal
):

    try:

        section = mesh.section(
            plane_origin=origin,
            plane_normal=normal
        )

    except Exception:

        return []


    if section is None:
        return []


    try:

        planar, to_3d = section.to_planar()

    except Exception:

        return []


    loops = []


    # polygons_full is ideal because it provides closed polygons.
    try:

        polygons = planar.polygons_full

        for polygon in polygons:

            coords = np.asarray(
                polygon.exterior.coords,
                dtype=float
            )

            if len(coords) < MIN_SECTION_POINTS:
                continue

            if (
                np.linalg.norm(
                    coords[0] -
                    coords[-1]
                )
                <
                1e-6
            ):
                coords = coords[:-1]

            if len(coords) < MIN_SECTION_POINTS:
                continue

            loops.append(
                coords
            )

    except Exception:

        pass


    # Fallback: discretized closed paths.
    if len(loops) == 0:

        try:

            for path in planar.discrete:

                coords = np.asarray(
                    path,
                    dtype=float
                )

                if len(coords) < MIN_SECTION_POINTS:
                    continue

                closure = np.linalg.norm(
                    coords[0] -
                    coords[-1]
                )

                if closure > 2.5 * VOXEL_PITCH_MM:
                    continue

                if closure < 1e-6:
                    coords = coords[:-1]

                loops.append(
                    coords
                )

        except Exception:

            pass


    return loops


# ============================================================
# Polygon area
# ============================================================

def polygon_area(points):

    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * abs(
        np.dot(
            x,
            np.roll(
                y,
                1
            )
        )
        -
        np.dot(
            y,
            np.roll(
                x,
                1
            )
        )
    )


# ============================================================
# Maximum Feret diameter
# ============================================================

def maximum_feret(points):

    points = np.asarray(
        points,
        dtype=float
    )


    if len(points) < 2:
        return (
            np.nan,
            None,
            None
        )


    try:

        if len(points) >= 3:

            hull = ConvexHull(
                points
            )

            pts = points[
                hull.vertices
            ]

        else:

            pts = points

    except Exception:

        pts = points


    if len(pts) < 2:

        return (
            np.nan,
            None,
            None
        )


    distances = pdist(
        pts
    )


    if len(distances) == 0:

        return (
            np.nan,
            None,
            None
        )


    k = int(
        np.argmax(
            distances
        )
    )


    # Recover pair indices from condensed pdist vector.
    n = len(pts)

    count = 0

    pair = None


    for i in range(
        n - 1
    ):

        row_len = (
            n - i - 1
        )

        if (
            k <
            count + row_len
        ):

            j = (
                i + 1 +
                (k - count)
            )

            pair = (
                i,
                j
            )

            break

        count += row_len


    if pair is None:

        return (
            float(
                distances[k]
            ),
            None,
            None
        )


    p1 = pts[
        pair[0]
    ]

    p2 = pts[
        pair[1]
    ]


    return (
        float(
            distances[k]
        ),
        p1,
        p2
    )


# ============================================================
# Choose section containing centerline
# ============================================================

def choose_section_loop(
        loops
):

    if len(loops) == 0:
        return None


    candidates = []


    for loop in loops:

        area = polygon_area(
            loop
        )

        if area < MIN_SECTION_AREA_MM2:
            continue


        # In planar coordinates, cutting origin is approximately
        # near the local vessel lumen center. Prefer a large,
        # geometrically valid contour.
        centroid = np.mean(
            loop,
            axis=0
        )

        centroid_distance = np.linalg.norm(
            centroid
        )


        candidates.append(
            (
                loop,
                area,
                centroid_distance
            )
        )


    if len(candidates) == 0:
        return None


    # Prefer contour close to plane origin;
    # among similar candidates favor larger area.
    candidates.sort(
        key=lambda x: (
            x[2],
            -x[1]
        )
    )


    return candidates[
        0
    ][
        0
    ]


# ============================================================
# Section measurement
# ============================================================

def measure_section(
        mesh,
        origin,
        tangent
):

    loops = get_section_loops(
        mesh,
        origin,
        tangent
    )


    loop = choose_section_loop(
        loops
    )


    if loop is None:

        return None


    area = polygon_area(
        loop
    )


    feret, p1, p2 = maximum_feret(
        loop
    )


    equivalent_diameter = np.sqrt(
        4.0 * area /
        np.pi
    )


    return {
        "area_mm2":
            float(area),

        "feret_diameter_mm":
            float(feret),

        "equivalent_diameter_mm":
            float(equivalent_diameter),

        "loop_2d":
            loop,

        "feret_p1_2d":
            p1,

        "feret_p2_2d":
            p2,
    }


# ============================================================
# Centerline length / tortuosity
# ============================================================

def centerline_geometry(
        centerline
):

    s = cumulative_distance(
        centerline
    )

    length = float(
        s[-1]
    )


    straight = float(
        np.linalg.norm(
            centerline[-1]
            -
            centerline[0]
        )
    )


    if straight > EPS:

        tortuosity = (
            length /
            straight
        )

    else:

        tortuosity = np.nan


    return (
        length,
        tortuosity
    )


# ============================================================
# Process one case
# ============================================================

def process_case(
        stl_path,
        case_id
):

    print(
        f"\n{'='*70}"
    )

    print(
        f"Processing: {case_id}"
    )

    print(
        stl_path
    )


    mesh = load_mesh(
        stl_path
    )


    mesh = keep_largest_component(
        mesh
    )


    volume, surface_area, sphericity = (
        mesh_morphology(
            mesh
        )
    )


    print(
        f"Vertices: {len(mesh.vertices)}"
    )

    print(
        f"Faces:    {len(mesh.faces)}"
    )

    print(
        f"Watertight: {mesh.is_watertight}"
    )


    # --------------------------------------------------------
    # Centerline
    # --------------------------------------------------------

    matrix, vox = voxelize_mesh(
        mesh,
        VOXEL_PITCH_MM
    )


    print(
        f"Voxel grid: {matrix.shape}"
    )


    skeleton = skeletonize(
        matrix
    )


    skeleton_voxels = int(
        skeleton.sum()
    )


    print(
        f"Skeleton voxels: {skeleton_voxels}"
    )


    G, coords = skeleton_graph(
        skeleton,
        VOXEL_PITCH_MM
    )


    G = prune_short_branches(
        G,
        min_length_mm=5.0
    )


    H, path_nodes = extract_longest_path(
        G
    )


    path_indices = np.array(
        [
            H.nodes[n][
                "coord"
            ]
            for n in path_nodes
        ],
        dtype=float
    )


    raw_centerline = (
        voxel_indices_to_world(
            vox,
            path_indices
        )
    )


    centerline, s = smooth_centerline(
        raw_centerline,
        SECTION_INTERVAL_MM
    )


    centerline_length, tortuosity = (
        centerline_geometry(
            centerline
        )
    )


    print(
        f"Centerline length: "
        f"{centerline_length:.2f} mm"
    )

    print(
        f"Tortuosity: "
        f"{tortuosity:.4f}"
    )


    # Save centerline
    centerline_df = pd.DataFrame(
        {
            "index":
                np.arange(
                    len(centerline)
                ),

            "arc_length_mm":
                cumulative_distance(
                    centerline
                ),

            "x_mm":
                centerline[:, 0],

            "y_mm":
                centerline[:, 1],

            "z_mm":
                centerline[:, 2],
        }
    )


    centerline_df.to_csv(
        CENTERLINE_DIR /
        f"{case_id}_centerline.csv",
        index=False
    )


    # --------------------------------------------------------
    # Perpendicular sections
    # --------------------------------------------------------

    arc = cumulative_distance(
        centerline
    )


    total_length = arc[-1]


    section_rows = []


    best = None


    for i in range(
        len(centerline)
    ):

        # Exclude centerline ends.
        if arc[i] < END_MARGIN_MM:
            continue

        if (
            total_length -
            arc[i]
            <
            END_MARGIN_MM
        ):
            continue


        tangent = local_tangent(
            centerline,
            i,
            SECTION_INTERVAL_MM,
            TANGENT_WINDOW_MM
        )


        if tangent is None:
            continue


        result = measure_section(
            mesh,
            centerline[i],
            tangent
        )


        if result is None:
            continue


        if not np.isfinite(
            result[
                "feret_diameter_mm"
            ]
        ):
            continue


        row = {

            "case_id":
                case_id,

            "centerline_index":
                i,

            "arc_length_mm":
                arc[i],

            "center_x_mm":
                centerline[i, 0],

            "center_y_mm":
                centerline[i, 1],

            "center_z_mm":
                centerline[i, 2],

            "tangent_x":
                tangent[0],

            "tangent_y":
                tangent[1],

            "tangent_z":
                tangent[2],

            "cross_section_area_mm2":
                result[
                    "area_mm2"
                ],

            "feret_diameter_mm":
                result[
                    "feret_diameter_mm"
                ],

            "equivalent_diameter_mm":
                result[
                    "equivalent_diameter_mm"
                ],
        }


        section_rows.append(
            row
        )


        if (
            best is None
            or
            result[
                "feret_diameter_mm"
            ]
            >
            best[
                "feret_diameter_mm"
            ]
        ):

            best = {
                **row,
                "loop_2d":
                    result[
                        "loop_2d"
                    ],

                "feret_p1_2d":
                    result[
                        "feret_p1_2d"
                    ],

                "feret_p2_2d":
                    result[
                        "feret_p2_2d"
                    ],
            }


    if len(section_rows) == 0:

        raise RuntimeError(
            "No valid perpendicular "
            "cross sections were found."
        )


    section_df = pd.DataFrame(
        section_rows
    )


    section_df.to_csv(
        CROSS_SECTION_DIR /
        f"{case_id}_cross_sections.csv",
        index=False
    )


    # --------------------------------------------------------
    # Save maximum cross-section coordinates
    # --------------------------------------------------------

    loop = best[
        "loop_2d"
    ]


    max_section_df = pd.DataFrame(
        {
            "u_mm":
                loop[:, 0],

            "v_mm":
                loop[:, 1],
        }
    )


    max_section_df.to_csv(
        CROSS_SECTION_DIR /
        f"{case_id}_maximum_section_contour.csv",
        index=False
    )


    # --------------------------------------------------------
    # Final morphology
    # --------------------------------------------------------

    max_diameter = best[
        "feret_diameter_mm"
    ]

    max_area = best[
        "cross_section_area_mm2"
    ]

    eq_diameter = best[
        "equivalent_diameter_mm"
    ]


    if (
        np.isfinite(volume)
        and
        max_diameter > 0
    ):

        volume_diameter_ratio = (
            volume /
            max_diameter
        )

    else:

        volume_diameter_ratio = np.nan


    print(
        f"Valid sections: "
        f"{len(section_df)}"
    )

    print(
        f"TRUE local maximum diameter: "
        f"{max_diameter:.2f} mm"
    )

    print(
        f"Maximum-section area: "
        f"{max_area:.2f} mm^2"
    )

    print(
        f"Equivalent diameter: "
        f"{eq_diameter:.2f} mm"
    )


    return {

        "case_id":
            case_id,

        "stl_file":
            str(stl_path),

        "mesh_watertight":
            bool(
                mesh.is_watertight
            ),

        "volume_mm3":
            volume,

        "surface_area_mm2":
            surface_area,

        "sphericity":
            sphericity,

        "centerline_length_mm":
            centerline_length,

        "tortuosity":
            tortuosity,

        "n_centerline_points":
            len(centerline),

        "n_valid_cross_sections":
            len(section_df),

        "max_centerline_diameter_mm":
            max_diameter,

        "max_cross_section_area_mm2":
            max_area,

        "equivalent_diameter_at_max_mm":
            eq_diameter,

        "max_diameter_arc_length_mm":
            best[
                "arc_length_mm"
            ],

        "volume_diameter_ratio":
            volume_diameter_ratio,

        "status":
            "OK",
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stl-root",
        type=str,
        default=str(ROOT),
        help=(
            "Directory containing STL files. "
            "Default: recursively search project."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of cases for testing."
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run only one case/file stem."
    )

    args = parser.parse_args()


    stl_root = Path(
        args.stl_root
    )


    files = discover_stl_files(
        stl_root
    )


    if args.case is not None:

        files = [
            p
            for p in files
            if p.stem == args.case
        ]


    if args.limit is not None:

        files = files[
            :args.limit
        ]


    print(
        "\n========================================"
    )

    print(
        "CENTERLINE-CORRECTED AAA MORPHOLOGY"
    )

    print(
        "========================================"
    )

    print(
        f"STL root: {stl_root}"
    )

    print(
        f"STL files found: {len(files)}"
    )

    print(
        f"Voxel pitch: "
        f"{VOXEL_PITCH_MM:.2f} mm"
    )

    print(
        f"Section interval: "
        f"{SECTION_INTERVAL_MM:.2f} mm"
    )

    print(
        "Diameter definition: "
        "maximum Feret diameter on "
        "centerline-perpendicular sections"
    )


    if len(files) == 0:

        raise FileNotFoundError(
            "No STL files found. "
            "Run: find . -iname '*.stl'"
        )


    rows = []


    for index, stl_path in enumerate(
        files,
        start=1
    ):

        case_id = get_case_id(
            stl_path
        )


        print(
            f"\n[{index}/{len(files)}]"
        )


        try:

            result = process_case(
                stl_path,
                case_id
            )

        except Exception as e:

            print(
                f"[FAILED] {case_id}: "
                f"{type(e).__name__}: {e}"
            )

            result = {

                "case_id":
                    case_id,

                "stl_file":
                    str(stl_path),

                "status":
                    (
                        f"FAILED: "
                        f"{type(e).__name__}: "
                        f"{e}"
                    ),
            }


        rows.append(
            result
        )


        # Save progress after every case.
        pd.DataFrame(
            rows
        ).to_csv(
            OUT_ROOT /
            "AAA_centerline_morphology_progress.csv",
            index=False
        )


    summary = pd.DataFrame(
        rows
    )


    final_csv = (
        OUT_ROOT /
        "AAA_centerline_morphology_100cases.csv"
    )


    summary.to_csv(
        final_csv,
        index=False
    )


    ok = summary[
        summary[
            "status"
        ] == "OK"
    ].copy()


    failed = summary[
        summary[
            "status"
        ] != "OK"
    ].copy()


    failed.to_csv(
        OUT_ROOT /
        "AAA_centerline_failed_cases.csv",
        index=False
    )


    print(
        "\n========================================"
    )

    print(
        "FINISHED"
    )

    print(
        "========================================"
    )

    print(
        f"Total:   {len(summary)}"
    )

    print(
        f"Success: {len(ok)}"
    )

    print(
        f"Failed:  {len(failed)}"
    )


    if len(ok) > 0:

        cols = [
            "case_id",
            "max_centerline_diameter_mm",
            "max_cross_section_area_mm2",
            "volume_mm3",
            "surface_area_mm2",
            "sphericity",
            "centerline_length_mm",
            "tortuosity",
        ]

        cols = [
            c
            for c in cols
            if c in ok.columns
        ]


        print(
            "\nSuccessful measurements:"
        )

        print(
            ok[
                cols
            ]
            .round(3)
            .to_string(
                index=False
            )
        )


    print(
        "\nSaved:"
    )

    print(
        final_csv
    )

    print(
        OUT_ROOT /
        "AAA_centerline_failed_cases.csv"
    )


if __name__ == "__main__":
    main()

