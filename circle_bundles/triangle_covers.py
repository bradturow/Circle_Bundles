"""
triangle_covers.py

Utilities for:
- Triangulations of S^2 (octahedron + barycentric subdivision)
- Building an RP^2 triangulation as an antipodal quotient of a triangulated S^2
- Veronese embedding of RP^2 into R^6
- Barycentric coordinates / point-to-triangle assignment
- Star subcomplexes and induced membership matrices

Conventions
-----------
- We represent triangulations using gudhi.SimplexTree.
- vertex_coords: dict[int, np.ndarray] mapping vertex id -> coordinate in R^3 (or R^6 after mapping).
- Triangle lists are tuples of 3 vertex ids sorted increasingly.

Notes / Fixes vs old notebook code
----------------------------------
- Uses gd.SimplexTree consistently (your snippet used SimplexTree without importing it).
- Unifies barycentric coordinate functions (triangle-specialized and general simplex).
- project_to_veronese() now reliably extracts triangles and uses correct bary function.
- get_simplices(st, dim) implemented here and used consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import gudhi as gd
except ImportError as e:
    raise ImportError("triangle_covers.py requires gudhi (import gudhi as gd).") from e


# ----------------------------
# Basic simplex-tree helpers
# ----------------------------

def get_simplices(st: gd.SimplexTree, dim: int) -> List[Tuple[int, ...]]:
    """
    Return a sorted list of dim-simplices as tuples of vertex ids.

    dim=0 -> vertices (length 1)
    dim=1 -> edges (length 2)
    dim=2 -> triangles (length 3)
    """
    k = dim + 1
    out: List[Tuple[int, ...]] = []
    for s, _ in st.get_simplices():
        if len(s) == k:
            out.append(tuple(sorted(s)))
    out.sort()
    return out


def insert_triangle_complex(st: gd.SimplexTree, triangles: Iterable[Sequence[int]]) -> gd.SimplexTree:
    """
    Insert triangles into simplex tree. Gudhi will automatically include faces,
    but we keep it simple and just insert triangles.
    """
    for tri in triangles:
        st.insert(list(tri))
    return st


# ----------------------------
# S^2 triangulation: octahedron
# ----------------------------

def initialize_octahedron() -> Tuple[gd.SimplexTree, Dict[int, np.ndarray]]:
    """
    Octahedron triangulation of S^2 with 6 vertices and 8 triangular faces.

    Returns
    -------
    K : gd.SimplexTree
        Simplex tree containing the 2D complex.
    vertex_coords : dict[int, np.ndarray]
        Vertex coordinates (unit sphere).
    """
    vertex_coords: Dict[int, np.ndarray] = {
        0: np.array([ 1.0, 0.0, 0.0]),
        1: np.array([-1.0, 0.0, 0.0]),
        2: np.array([ 0.0, 1.0, 0.0]),
        3: np.array([ 0.0,-1.0, 0.0]),
        4: np.array([ 0.0, 0.0, 1.0]),
        5: np.array([ 0.0, 0.0,-1.0]),
    }

    faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
        (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),
    ]

    K = gd.SimplexTree()
    insert_triangle_complex(K, faces)
    return K, vertex_coords


def sphere_to_octahedron(points: np.ndarray) -> np.ndarray:
    """
    Radial projection from S^2 to the inscribed octahedron surface
    by L1-normalization: x -> x / (|x|_1).

    points: (N,3) assumed nonzero rows.
    """
    points = np.asarray(points, dtype=float)
    denom = np.abs(points[:, 0]) + np.abs(points[:, 1]) + np.abs(points[:, 2])
    denom = np.maximum(denom, 1e-15)
    return points / denom[:, None]


# ----------------------------
# Barycentric subdivision (triangle complex)
# ----------------------------

def barycentric_refinement(
    K: gd.SimplexTree,
    vertex_coords: Dict[int, np.ndarray],
) -> Tuple[gd.SimplexTree, Dict[int, np.ndarray]]:
    """
    One barycentric subdivision step for a 2D triangle complex.
    We create:
      - midpoint vertices for each edge
      - barycenter vertices for each triangle
    and replace each triangle with 6 smaller triangles.

    Returns
    -------
    K_new, vertex_coords_new
    """
    # Collect triangles and edges from K
    triangles = get_simplices(K, 2)
    edges = get_simplices(K, 1)

    K_new = gd.SimplexTree()
    vertex_coords_new: Dict[int, np.ndarray] = dict(vertex_coords)

    next_vid = (max(vertex_coords_new.keys()) + 1) if vertex_coords_new else 0

    edge_mid: Dict[Tuple[int, int], int] = {}
    tri_bary: Dict[Tuple[int, int, int], int] = {}

    # Edge midpoints
    for e in edges:
        a, b = e
        mid = 0.5 * (vertex_coords[a] + vertex_coords[b])
        edge_mid[e] = next_vid
        vertex_coords_new[next_vid] = mid
        next_vid += 1

    # Triangle barycenters
    for tri in triangles:
        a, b, c = tri
        bar = (vertex_coords[a] + vertex_coords[b] + vertex_coords[c]) / 3.0
        tri_bary[tri] = next_vid
        vertex_coords_new[next_vid] = bar
        next_vid += 1

    # Insert subdivided triangles
    for (a, b, c) in triangles:
        m_ab = edge_mid[tuple(sorted((a, b)))]
        m_bc = edge_mid[tuple(sorted((b, c)))]
        m_ac = edge_mid[tuple(sorted((a, c)))]
        t_bar = tri_bary[tuple(sorted((a, b, c)))]

        # 6 triangles around barycenter
        K_new.insert([a,   m_ab, t_bar])
        K_new.insert([m_ab, b,   t_bar])
        K_new.insert([b,   m_bc, t_bar])
        K_new.insert([m_bc, c,   t_bar])
        K_new.insert([c,   m_ac, t_bar])
        K_new.insert([m_ac, a,   t_bar])

    return K_new, vertex_coords_new


def get_sd(
    K: gd.SimplexTree,
    n_sd: int,
    vertex_coords: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[gd.SimplexTree, Dict[int, np.ndarray]]:
    """
    Apply n_sd barycentric subdivisions.

    If vertex_coords is None, creates random coords (mainly for debugging).
    """
    if vertex_coords is None:
        verts = get_simplices(K, 0)
        vertex_coords = {v[0]: np.random.rand(3) for v in verts}

    K_ref = K
    vc = dict(vertex_coords)

    for _ in range(n_sd):
        K_ref, vc = barycentric_refinement(K_ref, vc)

    return K_ref, vc


# ----------------------------
# RP^2 triangulation from S^2 triangulation
# ----------------------------

def build_rp2_simplex_tree(
    K_s2: gd.SimplexTree,
    vertex_coords_s2: Dict[int, np.ndarray],
    *,
    atol: float = 1e-8,
) -> Tuple[gd.SimplexTree, Dict[int, np.ndarray], Dict[int, int]]:
    """
    Build an RP^2 triangulation by quotienting S^2 via antipodal identification.

    Strategy (close to your notebook logic):
    - Pair antipodal vertices via np.allclose(p, -q).
    - Choose a representative for each antipodal pair:
        - If z > 0: keep that vertex
        - If z ~ 0: choose smaller index among the antipodal pair
        - If z < 0: map to antipode's representative
    - Relabel each S^2 triangle by mapping its vertices through old_to_new,
      and keep those that remain 3-distinct.

    Returns
    -------
    K_rp2 : gd.SimplexTree
    vertex_coords_rp2 : dict[new_vid -> coord in R^3] (chosen representatives)
    old_to_new : dict[old_vid -> new_vid]
    """
    # Build list for matching
    vertex_list = list(vertex_coords_s2.keys())
    coords = np.array([vertex_coords_s2[v] for v in vertex_list])

    # Find antipodal pairs
    antipodal: Dict[int, int] = {}
    used = set()
    for i, vi in enumerate(vertex_list):
        if vi in used:
            continue
        pi = coords[i]
        found = False
        for j in range(i + 1, len(vertex_list)):
            vj = vertex_list[j]
            if vj in used:
                continue
            pj = coords[j]
            if np.allclose(pi, -pj, atol=atol):
                antipodal[vi] = vj
                antipodal[vj] = vi
                used.add(vi)
                used.add(vj)
                found = True
                break
        if not found:
            raise ValueError(f"Vertex {vi} has no antipodal partner (within atol={atol}).")

    old_to_new: Dict[int, int] = {}
    vertex_coords_rp2: Dict[int, np.ndarray] = {}
    next_new = 0
    visited = set()

    for v, p in vertex_coords_s2.items():
        if v in visited:
            continue
        a = antipodal[v]
        pa = vertex_coords_s2[a]

        zv = p[2]
        za = pa[2]

        if zv > 0:
            # keep v as representative
            old_to_new[v] = next_new
            old_to_new[a] = next_new
            vertex_coords_rp2[next_new] = p
            next_new += 1
            visited.add(v); visited.add(a)

        elif np.isclose(zv, 0.0, atol=atol):
            # equator: choose smaller index of (v,a)
            chosen = min(v, a)
            if chosen not in old_to_new:
                old_to_new[chosen] = next_new
                vertex_coords_rp2[next_new] = vertex_coords_s2[chosen]
                next_new += 1
            old_to_new[v] = old_to_new[chosen]
            old_to_new[a] = old_to_new[chosen]
            visited.add(v); visited.add(a)

        else:
            # zv < 0: map to antipode's representative; ensure antipode handled
            if za < 0 and not np.isclose(za, 0.0, atol=atol):
                raise ValueError(f"Both {v} and antipode {a} have z<0. Unexpected with antipodal pairing.")
            # process antipode first by continuing loop on it if needed
            if a not in visited:
                # Force processing antipode by handling it now (mirror logic)
                if za > 0:
                    old_to_new[v] = next_new
                    old_to_new[a] = next_new
                    vertex_coords_rp2[next_new] = pa
                    next_new += 1
                    visited.add(v); visited.add(a)
                else:
                    # za ~ 0 equator
                    chosen = min(v, a)
                    if chosen not in old_to_new:
                        old_to_new[chosen] = next_new
                        vertex_coords_rp2[next_new] = vertex_coords_s2[chosen]
                        next_new += 1
                    old_to_new[v] = old_to_new[chosen]
                    old_to_new[a] = old_to_new[chosen]
                    visited.add(v); visited.add(a)
            else:
                old_to_new[v] = old_to_new[a]
                visited.add(v)

    # Build RP2 simplex tree by relabeling triangles
    K_rp2 = gd.SimplexTree()
    for tri in get_simplices(K_s2, 2):
        rel = tuple(sorted({old_to_new[v] for v in tri}))
        if len(rel) == 3:
            K_rp2.insert(list(rel))

    return K_rp2, vertex_coords_rp2, old_to_new


# ----------------------------
# Veronese embedding RP^2 -> R^6
# ----------------------------

def veronese_map(x: np.ndarray) -> np.ndarray:
    """
    Veronese embedding v: S^2 -> R^6 factoring through RP^2:
      (x,y,z) -> (x^2, y^2, z^2, sqrt(2)xy, sqrt(2)xz, sqrt(2)yz)

    Works on shape (..., 3) and returns (..., 6).
    """
    x = np.asarray(x, dtype=float)
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    rt2 = np.sqrt(2.0)
    return np.stack([X**2, Y**2, Z**2, rt2*X*Y, rt2*X*Z, rt2*Y*Z], axis=-1)


# ----------------------------
# Barycentric coordinates / point-to-triangle
# ----------------------------

def barycentric_coords_triangle(points: np.ndarray, tri_vertices: np.ndarray) -> np.ndarray:
    """
    Barycentric coordinates of points w.r.t. a triangle in R^D.

    points: (N, D)
    tri_vertices: (3, D)

    Returns: (N, 3) bary coords (u,v,w) s.t. u+v+w=1.
    Uses a stable least-squares solve of the augmented system.
    """
    points = np.asarray(points, dtype=float)
    tri_vertices = np.asarray(tri_vertices, dtype=float)

    N, D = points.shape
    assert tri_vertices.shape == (3, D)

    # Solve [V^T; 1 1 1] * bary = [p^T; 1] for each p
    A = np.vstack([tri_vertices.T, np.ones((1, 3))])  # (D+1, 3)
    B = np.hstack([points, np.ones((N, 1))])          # (N, D+1)

    # least squares for all points at once
    bary, *_ = np.linalg.lstsq(A, B.T, rcond=None)    # (3, N)
    return bary.T                                     # (N, 3)


def points_to_simplices(
    points: np.ndarray,
    K: gd.SimplexTree,
    vertex_coords: Mapping[int, np.ndarray],
    *,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """
    Determine which triangle each point lies in (possibly multiple if on edges),
    and record barycentric coords for a chosen containing triangle.

    Returns
    -------
    V : (n_tri, n_pts) int {0,1}
        V[t,i]=1 if point i is in triangle t (up to tol).
    bary_coords : (n_pts, 3)
        barycentric coords for each point w.r.t. *one* triangle that contains it
        (last one found; good enough for your usage with bary_extend / interpolation).
    triangles : list of triangle tuples in same order as V rows.
    """
    points = np.asarray(points, dtype=float)
    triangles = get_simplices(K, 2)

    n_tri = len(triangles)
    n_pts = points.shape[0]

    V = np.zeros((n_tri, n_pts), dtype=int)
    bary_coords = np.zeros((n_pts, 3), dtype=float)

    assigned = np.zeros(n_pts, dtype=bool)

    for t, tri in enumerate(triangles):
        tri_xyz = np.array([vertex_coords[tri[0]], vertex_coords[tri[1]], vertex_coords[tri[2]]], dtype=float)
        bc = barycentric_coords_triangle(points, tri_xyz)  # (n_pts,3)

        in_tri = (np.abs(bc.sum(axis=1) - 1.0) <= tol) & (bc >= -tol).all(axis=1)
        if np.any(in_tri):
            V[t, in_tri] = 1
            # Only write bary coords for points not yet assigned (avoid overwriting)
            new_pts = in_tri & (~assigned)
            bary_coords[new_pts] = bc[new_pts]
            assigned[new_pts] = True

    return V, bary_coords, triangles


def project_to_veronese(
    points_r3: np.ndarray,
    vertex_coords_r3: Mapping[int, np.ndarray],
    K_r3: gd.SimplexTree,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Map 3D points lying on a triangulated surface (in R^3) to R^6 by:
    - finding a containing triangle in K_r3
    - computing barycentric coords in that triangle
    - interpolating the Veronese images of the triangle vertices in R^6

    Returns: (N,6)
    """
    points_r3 = np.asarray(points_r3, dtype=float)
    N = points_r3.shape[0]

    proj = np.zeros((N, 6), dtype=float)
    assigned = np.zeros(N, dtype=bool)

    triangles = get_simplices(K_r3, 2)

    for tri in triangles:
        tri_xyz = np.array([vertex_coords_r3[tri[0]], vertex_coords_r3[tri[1]], vertex_coords_r3[tri[2]]], dtype=float)
        bc = barycentric_coords_triangle(points_r3, tri_xyz)

        in_tri = (np.abs(bc.sum(axis=1) - 1.0) <= tol) & (bc >= -tol).all(axis=1) & (~assigned)
        if not np.any(in_tri):
            continue

        tri_v6 = veronese_map(tri_xyz)            # (3,6)
        proj[in_tri] = bc[in_tri] @ tri_v6        # (m,3)@(3,6)->(m,6)
        assigned[in_tri] = True

    # Optional: you might want to assert all assigned unless you expect boundary cases
    # if not np.all(assigned):
    #     missing = np.where(~assigned)[0]
    #     raise ValueError(f"{len(missing)} points were not assigned to any triangle (tol={tol}).")

    return proj


# ----------------------------
# Star cover
# ----------------------------

def get_star_cover(K: gd.SimplexTree) -> Dict[int, gd.SimplexTree]:
    """
    Return the open star subcomplexes (combinatorial stars) of vertices in K.

    Output:
        star_cover[v] is a simplex tree containing all simplices that contain v.
    """
    stars: Dict[int, gd.SimplexTree] = {}
    vertices = [v[0] for v in get_simplices(K, 0)]

    all_simplices = [(tuple(s), f) for s, f in K.get_simplices()]

    for v in vertices:
        stv = gd.SimplexTree()
        for s, _ in all_simplices:
            if v in s:
                stv.insert(list(s))
        stars[v] = stv

    return stars


# ----------------------------
# Induced membership U from subcomplexes
# ----------------------------

def get_U_from_subcomplexes(
    V: np.ndarray,
    subcomplexes: Mapping[int, gd.SimplexTree],
    K: gd.SimplexTree,
) -> np.ndarray:
    """
    Build U[j, i] = 1 if sample i lies in (at least one triangle of) subcomplex j.

    Assumes:
    - V has shape (n_triangles_in_K, n_samples)
    - subcomplexes is keyed by the same indices you want for rows of U
      (typically vertices of the nerve, etc.)

    Returns:
        U: (len(subcomplexes), n_samples) int {0,1}
    """
    triangles_K = get_simplices(K, 2)
    index_map = {tri: t for t, tri in enumerate(triangles_K)}

    keys = list(subcomplexes.keys())
    n_samples = V.shape[1]
    U = np.zeros((len(keys), n_samples), dtype=int)

    for j, key in enumerate(keys):
        tris_sub = get_simplices(subcomplexes[key], 2)
        tri_inds = [index_map[tri] for tri in tris_sub if tri in index_map]
        if tri_inds:
            U[j] = np.minimum(np.sum(V[tri_inds], axis=0), 1)

    return U


# ----------------------------
# Barycentric extension (kept close to your usage)
# ----------------------------

@dataclass
class CoverLike:
    """
    Minimal interface needed by bary_extend().
    This mirrors what your TriangulationStarCover probably already stores.
    """
    base_points: np.ndarray
    K: gd.SimplexTree
    nerve: gd.SimplexTree
    subcomplexes: Mapping[int, gd.SimplexTree]
    V: np.ndarray
    bary_coords: np.ndarray


def bary_extend(cover: CoverLike, beta_L: np.ndarray, dim: int = 1) -> np.ndarray:
    """
    Extend vertex-defined values beta_L to all base_points via barycentric interpolation
    triangle-by-triangle, restricted to subcomplexes of the cover.

    Parameters
    ----------
    cover : CoverLike
        Must include fields: base_points, K, nerve, subcomplexes, V, bary_coords
    beta_L : (n_sets, n_vertices) array
        Values at vertices of K for each set (dim=0) or each intersection (dim=1)
    dim : 0 or 1
        Whether beta_L is indexed by cover sets or by double intersections.

    Returns
    -------
    beta : (n_sets_or_inters, n_points) array
    """
    all_triangles = get_simplices(cover.K, 2)

    U_simplices = get_simplices(cover.nerve, dim)
    beta = np.zeros((beta_L.shape[0], len(cover.base_points)), dtype=float)

    if dim == 0:
        for j, simplex in enumerate(U_simplices):
            # simplex is (u,) for dim=0
            u = simplex[0]
            Uj_tris = set(get_simplices(cover.subcomplexes[u], 2))

            for t, tri in enumerate(all_triangles):
                if tri not in Uj_tris:
                    continue

                point_inds = (cover.V[t] == 1)
                if not np.any(point_inds):
                    continue

                bc = cover.bary_coords[point_inds]  # (m,3)
                v1, v2, v3 = tri
                beta[j, point_inds] = bc @ beta_L[j, [v1, v2, v3]]

    elif dim == 1:
        for jk, edge in enumerate(U_simplices):
            # edge is (u, v) for dim=1
            u, v = edge
            tris_u = set(get_simplices(cover.subcomplexes[u], 2))
            tris_v = set(get_simplices(cover.subcomplexes[v], 2))
            inter_tris = tris_u & tris_v

            for t, tri in enumerate(all_triangles):
                if tri not in inter_tris:
                    continue

                point_inds = (cover.V[t] == 1)
                if not np.any(point_inds):
                    continue

                bc = cover.bary_coords[point_inds]
                v1, v2, v3 = tri
                beta[jk, point_inds] = bc @ beta_L[jk, [v1, v2, v3]]

    else:
        raise ValueError("bary_extend only supports dim=0 or dim=1.")

    return beta
