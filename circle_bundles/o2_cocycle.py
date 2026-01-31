# circle_bundles/o2_cocycle.py
"""
Utilities for estimating and working with O(2)-valued cocycles from local angle data.

This module turns local angular coordinates on a cover into estimated transition maps
on overlaps, represented as 2×2 matrices in O(2).

Core data model
--------------
Let U be a cover membership matrix and f be local angles:

- U has shape (n_sets, n_samples) with boolean entries.
  U[j, s] == True means sample s lies in the j-th cover set.
- f has shape (n_sets, n_samples) in radians.
  f[j, s] is only meaningful when U[j, s] is True.

Transitions are stored on *directed* edges (j, k) with j < k by default:

    Omega[(j, k)] ∈ O(2)  maps k-coordinates -> j-coordinates.

So, if v_k is a unit-vector encoding of an angle in chart k, then Omega[(j,k)] @ v_k
is the corresponding unit-vector in chart j (up to estimation error).

Decomposition convention
------------------------
We decompose O ∈ O(2) using a fixed reflection axis angle `ref_angle`:

    O = R(theta) * r(ref_angle),

where:
- det(O) = +1  => r = I
- det(O) = -1  => r is reflection across the line through the origin at angle `ref_angle`
- theta is returned in [0, 2π).

This convention is used consistently throughout the codebase (including when
constructing Z₂ and O(1) cochains from determinants).
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Mat2 = np.ndarray

from .nerve.combinatorics import Edge, canon_edge

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
    Decompose an orthogonal 2×2 matrix into a rotation times a fixed-axis reflection.

    We write an input matrix O ∈ O(2) in the form

        O = R(theta) * r(ref_angle),

where:
- R(theta) is a counterclockwise rotation by theta,
- r(ref_angle) is either the identity (if det(O)=+1) or the reflection across the line
  making angle `ref_angle` with the +x-axis (if det(O)=-1).

Parameters
    ----------
    O:
        A 2×2 (approximately) orthogonal matrix.
    ref_angle:
        Reflection axis angle in radians. Only affects the decomposition when det(O)=-1.

Returns
    -------
    theta:
        Rotation angle in radians, normalized to the interval [0, 2π).
    det:
        The determinant sign of O, returned as +1 or -1.

Raises
    ------
    ValueError
        If O is not 2×2 or is not approximately orthogonal.

Notes
    -----
    This is a *convention choice*: many decompositions are possible for det=-1, but by
    fixing a reflection axis via `ref_angle`, the decomposition becomes deterministic and
    compatible across the library.
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
    """
    Summary statistics and diagnostics for transition estimation.

    A TransitionReport is returned alongside an :class:`O2Cocycle` by
    :func:`estimate_transitions`. It records basic sizes, missing overlaps, and
    per-edge fit errors.

    Attributes
    ----------
    n_sets:
        Number of cover sets (rows of U).
    n_samples:
        Number of samples (columns of U).
    min_points:
        Minimum overlap size required to estimate a transition on an edge.
    n_edges_requested:
        Number of edges requested for estimation. If no explicit edge list is provided,
        this equals n_sets choose 2.
    n_edges_estimated:
        Number of edges for which a transition was successfully estimated.
    missing_edges:
        List of edges (j, k) that were requested but skipped due to insufficient overlap.
    overlap_sizes:
        Dict mapping each requested edge (j, k) to the number of overlap samples |U_j ∩ U_k|.
    rms_angle_err:
        Dict mapping each estimated edge (j, k) to an RMS angular error (radians) measured
        on the overlap points.
    max_rms_angle_err:
        Maximum RMS angular error among estimated edges (0 if no edges estimated).
    mean_rms_angle_err:
        Mean RMS angular error among estimated edges (0 if no edges estimated).

    Notes
    -----
    The RMS angular error is computed after applying the estimated transition to unit
    vectors derived from angles, i.e. it measures discrepancy on S¹ (via arccos of dot products).
    """
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
    """
    An estimated O(2)-valued 1-cochain (typically a cocycle) on a cover nerve.

    The primary field is ``Omega``, a dictionary mapping directed edges to 2×2 matrices:

        Omega[(j, k)] maps k-coordinates -> j-coordinates.

    For convenience and downstream characteristic class computations, we also store
    determinant signs and rotation angles from a fixed-axis decomposition.

    Attributes
    ----------
    Omega:
        Dict mapping edges (j, k) to 2×2 matrices in O(2).
    theta:
        Dict mapping edges (j, k) to angles theta ∈ [0, 2π) such that
        Omega[(j,k)] = R(theta) * r(ref_angle) under this module's convention.
    det:
        Dict mapping edges (j, k) to det(Omega[(j,k)]) ∈ {+1, -1}.
    err:
        Optional dict mapping edges (j, k) to an RMS angular error (radians) on overlaps.
    ref_angle:
        Reflection axis angle (radians) used when decomposing det=-1 matrices.

    Notes
    -----
    - By default, :func:`estimate_transitions` stores only canonical edges (j<k).
      Use :meth:`complete_orientations` (or :func:`complete_edge_orientations`) if you want
      both directions (j,k) and (k,j).
    - The edge convention is important: Omega[(j,k)] is designed so that it takes
      vectors in chart k to vectors in chart j.

    Examples
    --------
    Typical usage is via a bundle construction pipeline, but you can also estimate
    transitions directly from local angles::

        cocycle, report = estimate_transitions(U, f, min_points=10)
        Omega_full = cocycle.complete_orientations().Omega
        omega_Z2 = cocycle.omega_Z2()

    See Also
    --------
    estimate_transitions :
        Fit O(2) transitions from overlap data.
    complete_edge_orientations :
        Low-level helper to fill reversed edges by transposition.
    """

    Omega: Dict[Edge, Mat2]                  # (j,k) -> 2x2 O(2) matrix, maps k -> j
    theta: Dict[Edge, float]                 # (j,k) -> angle in [0, 2π)
    det: Dict[Edge, int]                     # (j,k) -> ±1
    err: Optional[Dict[Edge, float]] = None  # (j,k) -> RMS angular error (radians), optional
    ref_angle: float = 0.0

    def omega_Z2(self) -> Dict[Edge, int]:
        """
        Convert determinants to a Z₂-valued 1-cochain.

        Returns
        -------
        omega:
            Dict mapping edges to values in {0,1}, using the convention:
            det=+1 ↦ 0 and det=-1 ↦ 1.
        """
        return {e: (1 - int(self.det[e])) // 2 for e in self.det}

    def omega_O1(self) -> Dict[Edge, int]:
        """
        Return the determinant-sign cochain as an O(1) (i.e. {±1}) valued 1-cochain.

        Returns
        -------
        omega:
            Dict mapping edges to ±1.
        """
        return dict(self.det)

    def theta_normalized(self) -> Dict[Edge, float]:
        """
        Return the rotation angles as an R/Z-valued cochain represented in [0, 1).

        The stored ``theta`` values are in radians in [0, 2π). This method converts to
        fractions of a full turn by dividing by 2π and reducing modulo 1.

        Returns
        -------
        theta01:
            Dict mapping edges to floats in [0,1).
        """
        out: Dict[Edge, float] = {}
        for e, th in self.theta.items():
            e2 = canon_edge(*e)
            out[e2] = (float(th) / (2.0 * np.pi)) % 1.0
        return out

    def restrict(self, edges: Iterable[Edge]) -> "O2Cocycle":
        """
        Restrict this cocycle to a specified set of edges.

        Parameters
        ----------
        edges:
            Iterable of edges. Edges are canonicalized to (min, max) form.

        Returns
        -------
        cocycle_restricted:
            A new :class:`O2Cocycle` containing only entries whose edges appear in `edges`.

        Notes
        -----
        Any edge not present in the underlying dicts is silently skipped.
        """
        edges2 = {canon_edge(int(a), int(b)) for (a, b) in edges}
        return O2Cocycle(
            Omega={e: self.Omega[e] for e in edges2 if e in self.Omega},
            theta={e: self.theta[e] for e in edges2 if e in self.theta},
            det={e: self.det[e] for e in edges2 if e in self.det},
            err={e: self.err[e] for e in edges2} if self.err else None,
            ref_angle=self.ref_angle,
        )

    def complete_orientations(self) -> "O2Cocycle":
        """
        Return an equivalent cocycle with both edge orientations filled in.

        For each undirected edge {j,k}, this ensures both directed entries are present:

            Omega[(j,k)] maps k->j,
            Omega[(k,j)] = Omega[(j,k)]^T maps j->k.

        Determinants, angles, and errors are also populated for the reversed edges in a way
        consistent with this module's decomposition convention.
        """
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
        Attempt to gauge-transform the cocycle so that det=+1 on a chosen 1-skeleton.

        This is an "orientability test" for the determinant-sign cochain on the graph
        given by `edges`. We look for an assignment φ_j ∈ {+1, -1} to vertices such that
        for each undirected edge {j,k}:

            φ_j * φ_k = det_{jk},

        where det_{jk} is det(Omega_{jk}) on that edge.

        If such a φ exists, we build a gauge g_j ∈ O(2) using the fixed reflection axis:

            g_j = I            if φ_j = +1
            g_j = r(ref_angle) if φ_j = -1

        and transform transitions by

            Omega'_{jk} = g_j * Omega_{jk} * g_k^{-1},

        which forces det(Omega'_{jk}) = +1 on all edges in the given 1-skeleton.

        Parameters
        ----------
        edges:
            The set of edges defining the 1-skeleton on which to test/orient.
        n_vertices:
            Number of vertices in the graph. If None, inferred as 1 + max index seen in `edges`.
        require_all_edges_present:
            If True, returns (False, self, phi) immediately if any requested edge is missing
            from this cocycle's determinant dict. If False, missing edges are ignored.

        Returns
        -------
        ok:
            True if the orientation assignment exists.
        cocycle_oriented:
            If ok is True, the gauge-transformed cocycle; otherwise the original cocycle.
        phi:
            An integer array of shape (n_vertices,) with entries in {+1,-1} giving φ.

        Notes
        -----
        This is useful when you want to reduce an O(2)-bundle problem to an S¹-bundle problem
        on a subcomplex where the bundle is orientable.
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
    Estimate O(2) transition maps on overlaps from local angle data.

    This routine fits, for each requested edge (j,k), an orthogonal matrix Omega[(j,k)] ∈ O(2)
    that best maps unit vectors derived from chart k to unit vectors derived from chart j
    over the overlap samples U[j] ∩ U[k].

    The fit is a (weighted) orthogonal Procrustes problem on S¹-embedded unit vectors:
    each angle θ is converted to v(θ) = (cos θ, sin θ), and we choose Omega minimizing

        Σ_s w_s || Omega v_k(s) - v_j(s) ||²

    over overlap samples s.

    Parameters
    ----------
    U:
        Boolean membership matrix of shape (n_sets, n_samples).
    f:
        Local angles in radians, shape (n_sets, n_samples). Values are only used where U is True.
    edges:
        Optional iterable of edges on which to estimate transitions. If None, attempts all pairs j<k.
        Each edge is canonicalized to (min, max).
    weights:
        Optional per-sample nonnegative weights of shape (n_samples,). If None, uniform weights are used.
        Weights are renormalized on each overlap to sum to 1.
    min_points:
        Minimum number of overlap samples required to estimate a transition on an edge.
    ref_angle:
        Reflection axis angle (radians) used in the decomposition convention
        Omega = R(theta) r(ref_angle) when det=-1.
    fail_fast_missing:
        If True and `edges` is provided explicitly, raise an error as soon as an edge has overlap
        size < min_points. If False, record such edges in the report and continue.

    Returns
    -------
    cocycle:
        An :class:`O2Cocycle` containing Omega, determinant signs, angles, and per-edge RMS errors.
        By default, transitions are stored only on canonical edges (j<k).
    report:
        A :class:`TransitionReport` with overlap sizes, missing edges, and summary error statistics.

    Notes
    -----
    - The stored convention is: Omega[(j,k)] maps k-coordinates -> j-coordinates.
    - If you want transitions on both orientations (j,k) and (k,j), call
      :meth:`O2Cocycle.complete_orientations` (or :func:`complete_edge_orientations`).
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
    Fill in both directed orientations of each edge by transposition.

    Given a dictionary Omega containing some set of edges (often only canonical edges j<k),
    this function returns a new dictionary Omega_full that contains *both* directions:

        Omega_full[(j,k)] and Omega_full[(k,j)] = Omega_full[(j,k)]^T.

    The determinant, angle, and error dictionaries are also completed to match.

    Parameters
    ----------
    Omega:
        Dict mapping edges to 2×2 matrices. Entries may include only canonical edges or a mix
        of directions.
    ref_angle:
        Reflection axis angle used when recomputing decompositions for reversed edges.
    dets:
        Optional determinant-sign dict aligned with Omega.
        If not provided for an edge, the determinant is computed from the matrix.
    thetas:
        Optional angle dict aligned with Omega.
        If not provided for an edge, the angle is computed via :func:`decompose_O2_as_R_times_r`.
    errs:
        Optional RMS-error dict aligned with Omega. If provided, reversed edges inherit the same
        error as the forward edge.

    Returns
    -------
    Omega_full:
        Dict containing both orientations of each edge.
    det_full:
        Determinant signs aligned with Omega_full.
    theta_full:
        Angles aligned with Omega_full (in radians, in [0, 2π) under this module's convention).
    err_full:
        Errors aligned with Omega_full, or None if `errs` was None.

    Notes
    -----
    Edge conventions are preserved: if Omega[(j,k)] maps k->j, then Omega[(k,j)] maps j->k.
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
