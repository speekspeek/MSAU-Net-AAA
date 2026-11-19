#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AAA true local maximum diameter based on:
STL + original abdominal_aorta.vtp centerline.

Key improvement over V2:
The VTP may store the centerline as many 2-point line segments
rather than one long VTK PolyLine. This version reconstructs
the ordered centerline using graph connectivity.

Pipeline
--------
VTP line segments
 -> graph reconstruction
 -> ordered abdominal-aorta centerline
 -> arc-length resampling
 -> local tangent
 -> centerline-perpendicular plane
 -> STL-plane intersection
 -> local contour
 -> maximum Feret diameter
 -> search along centerline
"""

from pathlib import Path
import argparse
import traceback

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
import networkx as nx

from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from shapely.geometry import Polygon


# ============================================================
# paths
# ============================================================

ROOT = Path("/home/ubuntu/AAA_CTA_Project")

DEFAULT_STL_ROOT = (
    ROOT /
    "0_raw_data/meshes/meshes"
)

DEFAULT_CENTERLINE_ROOT = (
    ROOT /
    "0_raw_data/centerlines/centerlines"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT /
    "9_results/true_centerline_diameter"
)


# ============================================================
# parameters
# ============================================================

DEFAULT_STEP_MM = 1.0

END_MARGIN_MM = 5.0

MIN_AREA_MM2 = 5.0

MAX_CONTOUR_CENTER_DISTANCE_MM = 30.0

EPS = 1e-10


# ============================================================
# helpers
# ============================================================

def normalize(v):

    v = np.asarray(
        v,
        dtype=float
    )

    n = np.linalg.norm(v)

    if n < EPS:
        raise ValueError(
            "Zero-length vector"
        )

    return v / n


def cumulative_length(points):

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) < 2:

        return np.array(
            [0.0]
        )

    segment_lengths = np.linalg.norm(

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
                segment_lengths
            )
        ]
    )


# ============================================================
# robust VTP centerline reconstruction
# ============================================================

def parse_vtk_lines(poly):

    """
    Convert PyVista poly.lines into lists of point IDs.

    VTK format:
    [n, id0, id1, ..., n, id0, id1, ...]
    """

    raw = np.asarray(
        poly.lines,
        dtype=np.int64
    )

    cells = []

    pos = 0

    while pos < len(raw):

        n = int(
            raw[pos]
        )

        if n <= 0:
            break

        ids = raw[
            pos + 1:
            pos + 1 + n
        ]

        if len(ids) == n:

            cells.append(
                ids.copy()
            )

        pos += (
            n + 1
        )

    return cells


def build_centerline_graph(
        points,
        line_cells
):

    G = nx.Graph()

    for i in range(
        len(points)
    ):

        G.add_node(
            int(i)
        )


    for ids in line_cells:

        if len(ids) < 2:
            continue

        # A PolyLine with N points represents N-1 edges.
        for k in range(
            len(ids) - 1
        ):

            a = int(
                ids[k]
            )

            b = int(
                ids[k + 1]
            )

            if a == b:
                continue

            distance = float(
                np.linalg.norm(
                    points[a] -
                    points[b]
                )
            )

            if distance < EPS:
                continue

            G.add_edge(
                a,
                b,
                weight=distance
            )

    # remove isolated points
    isolates = list(
        nx.isolates(G)
    )

    G.remove_nodes_from(
        isolates
    )

    return G


def graph_longest_centerline(
        points,
        G
):

    if G.number_of_nodes() < 3:

        raise RuntimeError(
            "Centerline graph has fewer than 3 connected points"
        )

    # --------------------------------------------------------
    # Keep largest connected component
    # --------------------------------------------------------

    components = list(
        nx.connected_components(
            G
        )
    )

    component = max(
        components,
        key=len
    )

    H = G.subgraph(
        component
    ).copy()


    # --------------------------------------------------------
    # Endpoint search
    # --------------------------------------------------------

    endpoints = [

        node

        for node in H.nodes

        if H.degree[node] == 1
    ]


    # --------------------------------------------------------
    # Normal centerline:
    # choose longest endpoint-to-endpoint path
    # --------------------------------------------------------

    if len(endpoints) >= 2:

        best_path = None
        best_length = -1.0


        for i in range(
            len(endpoints)
        ):

            source = endpoints[i]

            lengths, paths = (
                nx.single_source_dijkstra(
                    H,
                    source,
                    weight="weight"
                )
            )


            for target in endpoints[
                i + 1:
            ]:

                if target not in lengths:
                    continue

                L = lengths[
                    target
                ]

                if L > best_length:

                    best_length = L

                    best_path = paths[
                        target
                    ]


        if best_path is None:

            raise RuntimeError(
                "Unable to reconstruct endpoint path"
            )


    else:

        # ----------------------------------------------------
        # Fallback for closed/odd topology:
        # approximate graph diameter using two Dijkstra sweeps
        # ----------------------------------------------------

        start = next(
            iter(
                H.nodes
            )
        )


        dist1 = (
            nx.single_source_dijkstra_path_length(
                H,
                start,
                weight="weight"
            )
        )


        p1 = max(
            dist1,
            key=dist1.get
        )


        dist2 = (
            nx.single_source_dijkstra_path_length(
                H,
                p1,
                weight="weight"
            )
        )


        p2 = max(
            dist2,
            key=dist2.get
        )


        best_path = nx.shortest_path(
            H,
            p1,
            p2,
            weight="weight"
        )


        best_length = (
            nx.shortest_path_length(
                H,
                p1,
                p2,
                weight="weight"
            )
        )


    ordered = points[
        np.asarray(
            best_path,
            dtype=int
        )
    ]


    return (
        ordered,
        H,
        endpoints,
        best_length
    )


def read_ordered_centerline(
        vtp_path,
        verbose=True
):

    poly = pv.read(
        str(
            vtp_path
        )
    )


    points = np.asarray(
        poly.points,
        dtype=float
    )


    if verbose:

        print(
            f"  VTP points          : {poly.n_points}",
            flush=True
        )

        print(
            f"  VTP cells           : {poly.n_cells}",
            flush=True
        )

        print(
            f"  VTP lines           : {poly.n_lines}",
            flush=True
        )


    if len(points) < 3:

        raise RuntimeError(
            f"VTP contains only {len(points)} points"
        )


    cells = parse_vtk_lines(
        poly
    )


    if verbose:

        cell_sizes = [
            len(x)
            for x in cells
        ]

        print(
            f"  Parsed line cells   : {len(cells)}",
            flush=True
        )

        if len(cell_sizes):

            print(
                "  Line cell size range: "
                f"{min(cell_sizes)}-{max(cell_sizes)}",
                flush=True
            )


    # ========================================================
    # Case A: valid lines exist
    # ========================================================

    if len(cells) > 0:

        G = build_centerline_graph(
            points,
            cells
        )


        ordered, H, endpoints, graph_length = (
            graph_longest_centerline(
                points,
                G
            )
        )


        if verbose:

            print(
                f"  Graph nodes         : {H.number_of_nodes()}",
                flush=True
            )

            print(
                f"  Graph edges         : {H.number_of_edges()}",
                flush=True
            )

            print(
                f"  Graph endpoints     : {len(endpoints)}",
                flush=True
            )

            print(
                f"  Ordered points      : {len(ordered)}",
                flush=True
            )

            print(
                f"  Graph path length   : {graph_length:.3f} mm",
                flush=True
            )


    # ========================================================
    # Case B: no line connectivity -> point order fallback
    # ========================================================

    else:

        print(
            "  WARNING: no VTK line cells; using raw point order",
            flush=True
        )

        ordered = points.copy()


    # ========================================================
    # Remove consecutive duplicates
    # ========================================================

    clean = [
        ordered[0]
    ]


    for p in ordered[
        1:
    ]:

        if np.linalg.norm(
            p - clean[-1]
        ) > 1e-6:

            clean.append(
                p
            )


    ordered = np.asarray(
        clean,
        dtype=float
    )


    if len(ordered) < 3:

        raise RuntimeError(
            "Centerline contains fewer than 3 unique ordered points"
        )


    return ordered


# ============================================================
# resampling
# ============================================================

def resample_centerline(
        points,
        step_mm
):

    s = cumulative_length(
        points
    )

    total = float(
        s[-1]
    )


    if total <= 0:

        raise RuntimeError(
            "Invalid centerline length"
        )


    n = max(

        int(
            np.ceil(
                total /
                step_mm
            )
        ) + 1,

        3
    )


    new_s = np.linspace(
        0,
        total,
        n
    )


    output = np.zeros(
        (
            n,
            3
        ),
        dtype=float
    )


    for axis in range(3):

        f = interp1d(
            s,
            points[:, axis],
            kind="linear"
        )

        output[
            :,
            axis
        ] = f(
            new_s
        )


    return (
        output,
        new_s
    )


# ============================================================
# tangent
# ============================================================

def local_tangent(
        points,
        index,
        window=3
):

    i0 = max(
        0,
        index - window
    )

    i1 = min(
        len(points) - 1,
        index + window
    )


    if i0 == i1:

        raise RuntimeError(
            "Cannot estimate tangent"
        )


    return normalize(
        points[i1] -
        points[i0]
    )


# ============================================================
# plane basis / projection
# ============================================================

def plane_basis(
        normal
):

    n = normalize(
        normal
    )


    basis = np.eye(
        3
    )


    dots = np.abs(
        basis @ n
    )


    reference = basis[
        np.argmin(
            dots
        )
    ]


    u = normalize(
        np.cross(
            n,
            reference
        )
    )


    v = normalize(
        np.cross(
            n,
            u
        )
    )


    return (
        u,
        v
    )


def project_to_plane(
        points,
        origin,
        normal
):

    u, v = plane_basis(
        normal
    )


    q = (
        np.asarray(
            points,
            dtype=float
        )
        -
        origin
    )


    return np.column_stack(
        (
            q @ u,
            q @ v
        )
    )


# ============================================================
# 2D measurements
# ============================================================

def polygon_area(
        points
):

    if len(points) < 3:

        return np.nan


    try:

        poly = Polygon(
            points
        )


        if not poly.is_valid:

            poly = poly.buffer(
                0
            )


        return float(
            abs(
                poly.area
            )
        )


    except Exception:

        return np.nan


def maximum_feret(
        points
):

    points = np.asarray(
        points,
        dtype=float
    )


    if len(points) < 2:

        return np.nan


    try:

        hull = ConvexHull(
            points
        )

        hp = points[
            hull.vertices
        ]


    except Exception:

        hp = points


    if len(hp) < 2:

        return np.nan


    distances = pdist(
        hp
    )


    if len(distances) == 0:

        return np.nan


    return float(
        np.max(
            distances
        )
    )


# ============================================================
# mesh section
# ============================================================

def mesh_section_loops(
        mesh,
        origin,
        normal
):

    section = mesh.section(

        plane_origin=origin,

        plane_normal=normal
    )


    if section is None:

        return []


    try:

        curves = (
            section.discrete
        )


    except Exception:

        return []


    loops = []


    for curve in curves:

        curve = np.asarray(
            curve,
            dtype=float
        )


        if len(curve) < 3:

            continue


        loops.append(
            curve
        )


    return loops


# ============================================================
# select local contour
# ============================================================

def choose_local_contour(
        loops,
        center,
        tangent
):

    candidates = []


    for loop in loops:

        xy = project_to_plane(
            loop,
            center,
            tangent
        )


        if len(xy) < 3:

            continue


        area = polygon_area(
            xy
        )


        if (
            not np.isfinite(
                area
            )
            or
            area < MIN_AREA_MM2
        ):

            continue


        centroid = np.mean(
            xy,
            axis=0
        )


        center_distance = float(
            np.linalg.norm(
                centroid
            )
        )


        feret = maximum_feret(
            xy
        )


        if (
            not np.isfinite(
                feret
            )
            or
            feret <= 0
        ):

            continue


        candidates.append(
            {
                "loop":
                    loop,

                "xy":
                    xy,

                "area":
                    area,

                "feret":
                    feret,

                "center_distance":
                    center_distance
            }
        )


    if len(candidates) == 0:

        return None


    # Primary rule:
    # contour closest to centerline origin.
    candidates.sort(

        key=lambda d:
        (
            d[
                "center_distance"
            ],

            -d[
                "area"
            ]
        )
    )


    result = candidates[
        0
    ]


    if (
        result[
            "center_distance"
        ]
        >
        MAX_CONTOUR_CENTER_DISTANCE_MM
    ):

        return None


    return result


# ============================================================
# load STL
# ============================================================

def load_mesh(
        path
):

    mesh = trimesh.load(
        str(
            path
        ),
        force="mesh",
        process=False
    )


    if isinstance(
        mesh,
        trimesh.Scene
    ):

        geometries = [

            x

            for x in mesh.geometry.values()

            if isinstance(
                x,
                trimesh.Trimesh
            )
        ]


        if len(
            geometries
        ) == 0:

            raise RuntimeError(
                "STL scene contains no mesh"
            )


        mesh = trimesh.util.concatenate(
            geometries
        )


    if len(
        mesh.vertices
    ) == 0:

        raise RuntimeError(
            "Empty STL"
        )


    return mesh


# ============================================================
# one case
# ============================================================

def measure_case(
        case_id,
        stl_path,
        centerline_path,
        step_mm,
        save_sections,
        section_root
):

    mesh = load_mesh(
        stl_path
    )


    centerline_original = (
        read_ordered_centerline(
            centerline_path,
            verbose=True
        )
    )


    original_length = float(
        cumulative_length(
            centerline_original
        )[-1]
    )


    straight_length = float(
        np.linalg.norm(

            centerline_original[-1]
            -
            centerline_original[0]
        )
    )


    tortuosity = (

        original_length /
        straight_length

        if straight_length > 0

        else np.nan
    )


    print(
        f"  Centerline points   : {len(centerline_original)}",
        flush=True
    )

    print(
        f"  Centerline length   : {original_length:.3f} mm",
        flush=True
    )

    print(
        f"  Straight length     : {straight_length:.3f} mm",
        flush=True
    )

    print(
        f"  Tortuosity          : {tortuosity:.6f}",
        flush=True
    )


    centerline, s = (
        resample_centerline(
            centerline_original,
            step_mm
        )
    )


    records = []


    for i in range(
        len(centerline)
    ):

        position = float(
            s[i]
        )


        if position < END_MARGIN_MM:
            continue


        if (
            original_length -
            position
            <
            END_MARGIN_MM
        ):
            continue


        center = centerline[
            i
        ]


        try:

            tangent = local_tangent(
                centerline,
                i,
                window=3
            )


            loops = mesh_section_loops(
                mesh,
                center,
                tangent
            )


            result = choose_local_contour(
                loops,
                center,
                tangent
            )


            if result is None:
                continue


            area = result[
                "area"
            ]

            feret = result[
                "feret"
            ]


            equivalent_diameter = np.sqrt(
                4.0 *
                area /
                np.pi
            )


            records.append(
                {

                    "case_id":
                        case_id,

                    "centerline_index":
                        i,

                    "position_mm":
                        position,

                    "center_x_mm":
                        center[0],

                    "center_y_mm":
                        center[1],

                    "center_z_mm":
                        center[2],

                    "tangent_x":
                        tangent[0],

                    "tangent_y":
                        tangent[1],

                    "tangent_z":
                        tangent[2],

                    "feret_diameter_mm":
                        feret,

                    "cross_section_area_mm2":
                        area,

                    "equivalent_diameter_mm":
                        equivalent_diameter,

                    "contour_center_distance_mm":
                        result[
                            "center_distance"
                        ],

                    "contour_points":
                        len(
                            result[
                                "loop"
                            ]
                        ),
                }
            )


        except Exception:

            continue


    if len(
        records
    ) == 0:

        raise RuntimeError(
            "No valid perpendicular cross-sections"
        )


    section_df = pd.DataFrame(
        records
    )


    best_index = (
        section_df[
            "feret_diameter_mm"
        ].idxmax()
    )


    best = section_df.loc[
        best_index
    ]


    # --------------------------------------------------------
    # basic QC
    # --------------------------------------------------------

    max_diameter = float(
        best[
            "feret_diameter_mm"
        ]
    )


    max_area = float(
        best[
            "cross_section_area_mm2"
        ]
    )


    equivalent_diameter = float(
        best[
            "equivalent_diameter_mm"
        ]
    )


    valid_fraction = (

        len(section_df) /
        max(
            len(centerline) - 2,
            1
        )
    )


    # --------------------------------------------------------
    # save section-level data
    # --------------------------------------------------------

    if save_sections:

        section_root.mkdir(
            parents=True,
            exist_ok=True
        )


        section_df.to_csv(

            section_root /
            f"{case_id}_sections.csv",

            index=False
        )


    # --------------------------------------------------------
    # mesh properties
    # --------------------------------------------------------

    volume = float(
        abs(
            mesh.volume
        )
    )


    area3d = float(
        mesh.area
    )


    if (
        volume > 0
        and
        area3d > 0
    ):

        sphericity = float(
            (
                np.pi ** (
                    1.0 / 3.0
                )
            )
            *
            (
                6.0 * volume
            ) ** (
                2.0 / 3.0
            )
            /
            area3d
        )


    else:

        sphericity = np.nan


    return {

        "case_id":
            case_id,

        "stl_file":
            str(
                stl_path
            ),

        "centerline_file":
            str(
                centerline_path
            ),

        "original_centerline_points":
            len(
                centerline_original
            ),

        "centerline_length_mm":
            original_length,

        "straight_length_mm":
            straight_length,

        "tortuosity":
            tortuosity,

        "resampled_centerline_points":
            len(
                centerline
            ),

        "valid_sections":
            len(
                section_df
            ),

        "valid_section_fraction":
            valid_fraction,

        "max_feret_diameter_mm":
            max_diameter,

        "max_section_area_mm2":
            max_area,

        "max_section_equivalent_diameter_mm":
            equivalent_diameter,

        "max_diameter_position_mm":
            float(
                best[
                    "position_mm"
                ]
            ),

        "mesh_volume_mm3":
            volume,

        "mesh_surface_area_mm2":
            area3d,

        "sphericity":
            sphericity,

        "volume_diameter_ratio":
            (
                volume /
                max_diameter
                if max_diameter > 0
                else np.nan
            ),

        "status":
            "OK"
    }


# ============================================================
# main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--stl-root",

        default=str(
            DEFAULT_STL_ROOT
        )
    )


    parser.add_argument(

        "--centerline-root",

        default=str(
            DEFAULT_CENTERLINE_ROOT
        )
    )


    parser.add_argument(

        "--output-root",

        default=str(
            DEFAULT_OUTPUT_ROOT
        )
    )


    parser.add_argument(

        "--case",

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
    # determine cases
    # --------------------------------------------------------

    if args.case:

        cases = [
            args.case.upper()
        ]


    else:

        cases = sorted(
            [
                p.stem

                for p in stl_root.glob(
                    "AAA*.stl"
                )
            ]
        )


    print(
        "=" * 80,
        flush=True
    )

    print(
        "TRUE CENTERLINE-ORTHOGONAL AAA DIAMETER V3",
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )

    print(
        f"Cases : {len(cases)}",
        flush=True
    )

    print(
        f"Step  : {args.step_mm:.2f} mm",
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )


    results = []

    failures = []


    for number, case_id in enumerate(
        cases,
        start=1
    ):

        print(
            f"\n[{number:03d}/{len(cases):03d}] {case_id}",
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

            result = measure_case(

                case_id,

                stl_path,

                centerline_path,

                args.step_mm,

                args.save_sections,

                section_root
            )


            results.append(
                result
            )


            print(
                f"  Valid sections      : "
                f"{result['valid_sections']}",
                flush=True
            )

            print(
                f"  MAX FERET           : "
                f"{result['max_feret_diameter_mm']:.3f} mm",
                flush=True
            )

            print(
                f"  Maximum area        : "
                f"{result['max_section_area_mm2']:.3f} mm^2",
                flush=True
            )

            print(
                f"  Equivalent diameter : "
                f"{result['max_section_equivalent_diameter_mm']:.3f} mm",
                flush=True
            )


        except Exception as e:

            print(
                f"  FAILED: "
                f"{type(e).__name__}: {e}",
                flush=True
            )


            failures.append(
                {
                    "case_id":
                        case_id,

                    "error":
                        (
                            f"{type(e).__name__}: "
                            f"{e}"
                        ),

                    "traceback":
                        traceback.format_exc()
                }
            )


    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    if results:

        result_df = pd.DataFrame(
            results
        )


        result_file = (
            output_root /
            "AAA_true_centerline_diameter_v3.csv"
        )


        result_df.to_csv(
            result_file,
            index=False
        )


        print(
            "\nResults saved:",
            result_file,
            flush=True
        )


    if failures:

        failure_file = (
            output_root /
            "AAA_true_centerline_diameter_v3_failures.csv"
        )


        pd.DataFrame(
            failures
        ).to_csv(
            failure_file,
            index=False
        )


        print(
            "Failures saved:",
            failure_file,
            flush=True
        )


    print(
        "\n==========================================",
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

    print(
        "==========================================",
        flush=True
    )


if __name__ == "__main__":

    main()

