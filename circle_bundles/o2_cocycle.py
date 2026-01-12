# circle_bundles/o2_cocycle.py
"""
o2_cocycle.py

Utilities for estimating and working with O(2) cocycles coming from local angle
data on a cover.

Conventions
-----------
- U has shape (n_sets, n_samples) bool.
- f has shape (n_sets, n_samples) radians, only meaningful where U[j] is True.
- Estimated transitions stored on edges with j<k:
      Omega[(j,k)] ∈ O(2) maps k-coordinates -> j-coordinates.

Decomposition convention:
      Omega = R(theta) r
where r = I if det=+1, or r = reflection about axis ref_angle if det=-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Mat2 = np.ndarray

from .combinatorics import Edge, canon_edge

__all__ = [
    "Mat2",
    "angles_to_unit",
    "rotation_matrix",
    "reflection_axis_matrix",
    "r_from_det",
    "project_to_O2",
    "det_sign",
    "decompose_O2_as_R_times_r",
    "TransitionReport",
    "O2Cocycle",
    "estimate_transitions",
    "complete_edge_orientations",
]


# ----------------------------
# Basic helpers
# ----------------------------

def angles_to_unit(theta: np.ndarray) -> np.ndarray:
    """(n,) angles -> (n,2) unit vectors."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1:
        theta = theta.reshape(-1)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


# ----------------------------
# O(2) helpers
# ----------------------------

def rotation_matrix(theta: float) -> Mat2:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def reflection_axis_matrix(ref_angle: float) -> Mat2:
    """
    Reflection across the line through the origin making angle ref_angle with +x axis:
        r_a = R(a) diag(1,-1) R(-a)
    """
    a = float(ref_angle)
    c2, s2 = np.cos(2 * a), np.sin(2 * a)
    return np.array([[ c2,  s2],
                     [ s2, -c2]], dtype=float)


def r_from_det(det: int, ref_angle: float) -> Mat2:
    if det not in (+1, -1):
        raise ValueError("det must be ±1")
    return np.eye(2) if det == 1 else reflection_axis_matrix(ref_angle)


def project_to_O2(M: Mat2) -> Mat2:
    """Project a near-orthogonal 2x2 matrix to O(2) via SVD."""
    M = np.asarray(M, dtype=float)
    if M.shape != (2, 2):
        raise ValueError("M must be 2x2.")
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def det_sign(O: Mat2) -> int:
    d = float(np.linalg.det(O))
    return 1 if d >= 0 else -1


def decompose_O2_as_R_times_r(O: Mat2, ref_angle: float = 0.0) -> Tuple[float, int]:
    """
    Decompose O ∈ O(2) as O = R(theta) r(ref_angle), with theta in [0,2π).
    """
    O = np.asarray(O, dtype=float)
    if O.shape != (2, 2):
        raise ValueError("O must be 2x2.")

    # Keep your convention: require approximately orthogonal.
    if not np.allclose(O.T @ O, np.eye(2), atol=1e-7):
        raise ValueError("O must be orthogonal (approximately).")

    det = det_sign(O)
    r = r_from_det(det, ref_angle)     # r^{-1} = r
    R = O @ r                          # O = R r => R = O r
    theta = float(np.arctan2(R[1, 0], R[0, 0])) % (2 * np.pi)
    return theta, det


# ----------------------------
# Transition estimation
# ----------------------------

@dataclass
class TransitionReport:
    n_sets: int
    n_samples: int
    min_points: int
    n_edges_requested: int
    n_edges_estimated: int
    missing_edges: List[Edge]

    overlap_sizes: Dict[Edge, int]
    rms_angle_err: Dict[Edge, float]

    max_rms_angle_err: float
    mean_rms_angle_err: float


@dataclass
class O2Cocycle:
    Omega: Dict[Edge, Mat2]                 # (j,k) -> 2x2 O(2) matrix
    theta: Dict[Edge, float]                # (j,k) -> angle in [0,2π)
    det: Dict[Edge, int]                    # (j,k) -> ±1
    err: Optional[Dict[Edge, float]] = None
    ref_angle: float = 0.0

    def omega_Z2(self) -> Dict[Edge, int]:
        """(+1)->0, (-1)->1."""
        return {e: (1 - int(self.det[e])) // 2 for e in self.det}

    def omega_O1(self) -> Dict[Edge, int]:
        return dict(self.det)

    def theta_normalized(self) -> Dict[Edge, float]:
        """
        Return theta as an R/Z-valued cochain represented in [0,1).
        Our stored theta is in radians in [0,2π).
        """
        out: Dict[Edge, float] = {}
        for e, th in self.theta.items():
            e2 = canon_edge(*e)
            out[e2] = (float(th) / (2.0 * np.pi)) % 1.0
        return out

    def restrict(self, edges: Iterable[Edge]) -> "O2Cocycle":
        edges2 = {canon_edge(int(a), int(b)) for (a, b) in edges}
        return O2Cocycle(
            Omega={e: self.Omega[e] for e in edges2 if e in self.Omega},
            theta={e: self.theta[e] for e in edges2 if e in self.theta},
            det={e: self.det[e] for e in edges2 if e in self.det},
            err={e: self.err[e] for e in edges2} if self.err else None,
            ref_angle=self.ref_angle,
        )

    def complete_orientations(self) -> "O2Cocycle":
        Omega_full, det_full, theta_full, err_full = complete_edge_orientations(
            self.Omega,
            ref_angle=self.ref_angle,
            dets=self.det,
            thetas=self.theta,
            errs=self.err,
        )
        return O2Cocycle(
            Omega=Omega_full,
            det=det_full,
            theta=theta_full,
            err=err_full,
            ref_angle=self.ref_angle,
        )

    def orient_if_possible(
        self,
        edges: Iterable[Edge],
        *,
        n_vertices: Optional[int] = None,
        require_all_edges_present: bool = True,
    ) -> Tuple[bool, "O2Cocycle", np.ndarray]:
        """
        Try to "orient" the O(2) cocycle on the given 1-skeleton.

        We look for a vertex assignment phi_j ∈ {+1,-1} such that for every edge {j,k},
            phi_j * phi_k = det_{jk},
        where det_{jk} is det(Omega_{jk}) viewed on the undirected edge.

        If such phi exists, we gauge-transform by choosing g_j = I if phi_j=+1 and
        g_j = r(ref_angle) if phi_j=-1, and set
            Omega'_{jk} = g_j * Omega_{jk} * g_k^{-1}  (here g_k^{-1}=g_k),
        which forces det(Omega'_{jk}) = +1 on those edges.
        """
        E = [canon_edge(int(a), int(b)) for (a, b) in edges if a != b]
        E = sorted(set(E))

        if n_vertices is None:
            if len(E) == 0:
                n_vertices = 0
            else:
                n_vertices = 1 + max(max(j, k) for (j, k) in E)

        adj: List[List[Tuple[int, int]]] = [[] for _ in range(n_vertices)]
        for (j, k) in E:
            if (j, k) not in self.det:
                if require_all_edges_present:
                    phi = np.ones(n_vertices, dtype=int)
                    return False, self, phi
                else:
                    continue
            s = int(self.det[(j, k)])
            if s not in (+1, -1):
                raise ValueError(f"det[(j,k)] must be ±1; got {s} on edge {(j,k)}")
            adj[j].append((k, s))
            adj[k].append((j, s))

        phi = np.zeros(n_vertices, dtype=int)  # 0 means unassigned
        ok = True

        for start in range(n_vertices):
            if phi[start] != 0:
                continue
            phi[start] = 1
            stack = [start]
            while stack and ok:
                u = stack.pop()
                for (v, s_uv) in adj[u]:
                    want = s_uv * phi[u]
                    if phi[v] == 0:
                        phi[v] = want
                        stack.append(v)
                    elif phi[v] != want:
                        ok = False
                        break
            if not ok:
                break

        phi_pm1 = phi.copy()
        phi_pm1[phi_pm1 == 0] = 1

        if not ok:
            return False, self, phi_pm1

        R = reflection_axis_matrix(self.ref_angle)
        G: List[Mat2] = [np.eye(2) if phi_pm1[j] == 1 else R for j in range(n_vertices)]

        Omega_new: Dict[Edge, Mat2] = {}
        det_new: Dict[Edge, int] = {}
        theta_new: Dict[Edge, float] = {}
        err_new: Optional[Dict[Edge, float]] = dict(self.err) if self.err is not None else None

        for (j, k), O in self.Omega.items():
            jj, kk = int(j), int(k)
            if jj < n_vertices and kk < n_vertices:
                O2 = G[jj] @ O @ G[kk]  # inverse = itself
                O2 = project_to_O2(O2)
                th2, d2 = decompose_O2_as_R_times_r(O2, ref_angle=self.ref_angle)
                Omega_new[(j, k)] = O2
                det_new[(j, k)] = int(d2)
                theta_new[(j, k)] = float(th2)
            else:
                Omega_new[(j, k)] = O
                det_new[(j, k)] = int(self.det[(j, k)])
                theta_new[(j, k)] = float(self.theta[(j, k)])

        coc_oriented = O2Cocycle(
            Omega=Omega_new,
            theta=theta_new,
            det=det_new,
            err=err_new,
            ref_angle=self.ref_angle,
        )
        return True, coc_oriented, phi_pm1


def estimate_transitions(
    U: np.ndarray,
    f: np.ndarray,
    *,
    edges: Optional[Iterable[Edge]] = None,
    weights: Optional[np.ndarray] = None,
    min_points: int = 5,
    ref_angle: float = 0.0,
    fail_fast_missing: bool = False,
) -> Tuple[O2Cocycle, TransitionReport]:
    """
    Estimate O(2) transitions Omega_{jk} from local angle data.

    Returns Omega on edges (j,k) with j<k.
    """
    U = np.asarray(U, dtype=bool)
    f = np.asarray(f, dtype=float)

    n_sets, n_samples = U.shape
    if f.shape != (n_sets, n_samples):
        raise ValueError(f"f must have shape {(n_sets, n_samples)}; got {f.shape}.")

    if weights is None:
        weights = np.ones(n_samples, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_samples,):
            raise ValueError(f"weights must have shape {(n_samples,)}; got {weights.shape}.")

    if edges is None:
        edge_list = [(j, k) for j in range(n_sets) for k in range(j + 1, n_sets)]
        requested = False
    else:
        edge_list = sorted(set(canon_edge(a, b) for (a, b) in edges if a != b))
        requested = True

    Omega: Dict[Edge, Mat2] = {}
    dets: Dict[Edge, int] = {}
    thetas: Dict[Edge, float] = {}
    errs: Dict[Edge, float] = {}
    overlap_sizes: Dict[Edge, int] = {}
    missing_edges: List[Edge] = []

    for (j, k) in edge_list:
        overlap = U[j] & U[k]
        idx = np.where(overlap)[0]
        m = int(idx.size)
        overlap_sizes[(j, k)] = m

        if m < min_points:
            missing_edges.append((j, k))
            if fail_fast_missing and requested:
                raise ValueError(f"Edge {(j,k)} has overlap {m} < min_points={min_points}.")
            continue

        # vj is target, vk is source
        vj = angles_to_unit(f[j, idx])
        vk = angles_to_unit(f[k, idx])

        w = weights[idx].astype(float)
        wsum = float(w.sum())
        w = w / (wsum if wsum > 0 else 1.0)

        # Weighted orthogonal Procrustes: minimize || (O vk) - vj ||
        # (Keep your exact convention / formula.)
        H = (vk * w[:, None]).T @ vj
        U_svd, _, Vt_svd = np.linalg.svd(H)
        O = Vt_svd.T @ U_svd.T

        # ensure in O(2) numerically
        O = project_to_O2(O)

        theta, det = decompose_O2_as_R_times_r(O, ref_angle=ref_angle)

        # RMS angular error on overlap
        vk_trans = (O @ vk.T).T
        dot = np.sum(vj * vk_trans, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        ang_err = np.arccos(dot)
        err = float(np.sqrt(np.mean(ang_err**2)))

        Omega[(j, k)] = O
        dets[(j, k)] = det
        thetas[(j, k)] = theta
        errs[(j, k)] = err

    err_vals = list(errs.values())
    report = TransitionReport(
        n_sets=n_sets,
        n_samples=n_samples,
        min_points=min_points,
        n_edges_requested=len(edge_list),
        n_edges_estimated=len(Omega),
        missing_edges=missing_edges,
        overlap_sizes=overlap_sizes,
        rms_angle_err=errs,
        max_rms_angle_err=float(np.max(err_vals)) if err_vals else 0.0,
        mean_rms_angle_err=float(np.mean(err_vals)) if err_vals else 0.0,
    )

    cocycle = O2Cocycle(
        Omega=Omega,
        det=dets,
        theta=thetas,
        err=errs,
        ref_angle=ref_angle,
    )
    return cocycle, report


# ----------------------------
# Complete reversed orientations
# ----------------------------

def complete_edge_orientations(
    Omega: Dict[Edge, Mat2],
    *,
    ref_angle: float = 0.0,
    dets: Optional[Dict[Edge, int]] = None,
    thetas: Optional[Dict[Edge, float]] = None,
    errs: Optional[Dict[Edge, float]] = None,
) -> Tuple[Dict[Edge, Mat2], Dict[Edge, int], Dict[Edge, float], Optional[Dict[Edge, float]]]:
    """
    Given a dict Omega that may contain only canonical edges (j<k) or may already
    contain some directed edges, return a full dict containing BOTH directions:
      (j,k) and (k,j) with transpose.

    Also returns det/theta/err dictionaries aligned with the returned Omega_full.

    Conventions are unchanged:
      If Omega[(j,k)] maps k -> j, then Omega[(k,j)] = Omega[(j,k)]^T maps j -> k.
    """
    Omega_full: Dict[Edge, Mat2] = {}
    det_full: Dict[Edge, int] = {}
    theta_full: Dict[Edge, float] = {}
    err_full: Optional[Dict[Edge, float]] = {} if errs is not None else None

    # Process a deterministic set of base edges to avoid double work if Omega already had both directions.
    base_edges: List[Edge] = []
    for (j, k) in Omega.keys():
        jj, kk = int(j), int(k)
        if jj == kk:
            continue
        base_edges.append(canon_edge(jj, kk))
    base_edges = sorted(set(base_edges))

    for (j, k) in base_edges:
        # Prefer stored canonical direction if present; else use whichever is present.
        if (j, k) in Omega:
            O = np.asarray(Omega[(j, k)], dtype=float)
            O_dir = (j, k)
        elif (k, j) in Omega:
            O = np.asarray(Omega[(k, j)], dtype=float).T  # convert to (j,k) direction
            O_dir = (j, k)
        else:
            continue

        Omega_full[(j, k)] = O

        d = int(dets[(j, k)]) if (dets is not None and (j, k) in dets) else det_sign(O)
        det_full[(j, k)] = d

        if thetas is not None and (j, k) in thetas:
            th = float(thetas[(j, k)]) % (2 * np.pi)
        else:
            th = decompose_O2_as_R_times_r(O, ref_angle)[0]
        theta_full[(j, k)] = float(th)

        if err_full is not None:
            if errs is not None and (j, k) in errs:
                err_full[(j, k)] = float(errs[(j, k)])
            elif errs is not None and (k, j) in errs:
                err_full[(j, k)] = float(errs[(k, j)])
            else:
                err_full[(j, k)] = np.nan

        # Reverse
        O_rev = O.T
        Omega_full[(k, j)] = O_rev
        th_rev, d_rev = decompose_O2_as_R_times_r(O_rev, ref_angle=ref_angle)
        det_full[(k, j)] = int(d_rev)
        theta_full[(k, j)] = float(th_rev)
        if err_full is not None:
            err_full[(k, j)] = err_full[(j, k)]

    return Omega_full, det_full, theta_full, err_full
