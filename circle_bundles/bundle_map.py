# circle_bundles/bundle_map.py
"""
bundle_map.py

Global bundle coordinatization pipeline.

This version includes:
- the PSC/subspace-PCA "frame reduction" hook (delegated to frame_reduction.py)
- the IMPORTANT gluing fix: anchored principal-branch lift for angle averaging
  (chart-equivariant; avoids π-flips caused by largest-gap branch cuts)

Core idea for gluing:
For each sample s and each active chart j, define
    F_j(s) = Phi_true[j,s] @ unit(mean_theta_j(s))
where mean_theta_j is computed by transporting local angles f_k(s) into chart j,
then taking an *anchored* weighted mean (strict semicircle by default).

If semicircle hypothesis holds, these chart formulas glue exactly on overlaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np

from .combinatorics import Edge, canon_edge
from .o2_cocycle import angles_to_unit

from .frame_reduction import (
    FrameReducerConfig,
    FrameReductionReport,
    reduce_frames_subspace_pca,
    reduce_frames_psc,
)

Mat2 = np.ndarray

SummaryMode = Literal["auto", "text", "latex", "both"]



# ============================================================
# Angle utilities (anchored strict semicircle averaging)
# ============================================================

def _wrap_to_pi(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float(((theta + np.pi) % (2.0 * np.pi)) - np.pi)


def _circle_dist(a: float, b: float) -> float:
    """Geodesic distance on S^1 between angles a,b (radians)."""
    return abs(_wrap_to_pi(a - b))


def weighted_angle_mean_anchored(
    angles: np.ndarray,
    weights: np.ndarray,
    *,
    anchor: float,
    tol: float = 1e-8,
) -> float:
    """
    Weighted mean of angles using a principal-branch lift anchored at `anchor`.

    Lift each angle to be within (-pi, pi] of anchor:
        diff_k = wrap_to_pi(a_k - anchor)
        a_lift_k = anchor + diff_k

    Semicircle test (anchored):
        max_k |diff_k| < pi  (up to tol)

    Returns:
      mean angle in (-pi, pi]
    """
    angles = np.asarray(angles, dtype=float).reshape(-1) % (2.0 * np.pi)
    w = np.asarray(weights, dtype=float).reshape(-1)

    if angles.shape != w.shape:
        raise ValueError("angles and weights must have the same shape.")
    if angles.size == 0:
        raise ValueError("Need at least one angle to average.")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative.")
    ws = float(w.sum())
    if ws <= 0:
        raise ValueError("weights must sum to a positive value.")
    w = w / ws

    anchor = float(anchor) % (2.0 * np.pi)

    diffs = ((angles - anchor + np.pi) % (2.0 * np.pi)) - np.pi  # (-pi,pi]
    max_abs = float(np.max(np.abs(diffs)))

    if max_abs >= np.pi - tol:
        raise ValueError(
            f"Angles are not contained in an open semicircle around the anchor "
            f"(max_abs={max_abs:.6f} >= pi)."
        )

    lifted = anchor + diffs
    mean = float(np.sum(w * lifted))
    return _wrap_to_pi(mean)


# ============================================================
# Linear algebra helpers (rank-2 projector + Stiefel polar)
# ============================================================

def project_to_rank2_projection(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    As = (A + A^T)/2
    Take top-2 eigenspace of As, return P = V2 V2^T and dist = ||As - P||_F.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    As = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(As)  # ascending
    order = np.argsort(vals)[::-1]
    V2 = vecs[:, order[:2]]  # (d,2)
    P = V2 @ V2.T
    dist = float(np.linalg.norm(As - P, ord="fro"))
    return P, dist


def polar_stiefel_projection(P: np.ndarray, A: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Given projection P (d,d) and A (d,2), return polar factor of PA:
      U = (PA) ((PA)^T (PA))^{-1/2} ∈ V(2,d)
    """
    P = np.asarray(P, dtype=float)
    A = np.asarray(A, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square.")
    d = P.shape[0]
    if A.shape != (d, 2):
        raise ValueError(f"A must have shape {(d,2)}; got {A.shape}.")

    PA = P @ A
    AtA = PA.T @ PA
    eigvals, eigvecs = np.linalg.eigh(AtA)
    eigvals = np.clip(eigvals, eps, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return PA @ inv_sqrt


# ============================================================
# Core construction pieces
# ============================================================

def infer_edges_from_U(U: np.ndarray) -> List[Edge]:
    """Edges (j,k) where U_j ∩ U_k contains at least one sample."""
    U = np.asarray(U, dtype=bool)
    n_sets, _ = U.shape
    E: List[Edge] = []
    for j in range(n_sets):
        for k in range(j + 1, n_sets):
            if np.any(U[j] & U[k]):
                E.append((j, k))
    return E


def get_local_frames(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    Omega: Dict[Edge, Mat2],
    edges: Iterable[Edge],
) -> np.ndarray:
    """
    Build approximate local frames Phi[j,s] ∈ R^{D×2}, D=2*n_sets:

      Phi_j(s) = [ sqrt(rho_0(s)) Omega_{0<-j} ;
                   ...
                   sqrt(rho_{n-1}(s)) Omega_{n-1<-j} ]

    Convention:
      Omega[(a,b)] for a<b maps b -> a.
      Directed Omega_{i<-j} is:
        - I if i==j
        - Omega[(i,j)] if i<j
        - Omega[(j,i)]^T if i>j
    """
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    n_sets, n_samples = U.shape
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou must have same shape as U.")

    edge_set = {canon_edge(a, b) for (a, b) in edges if a != b}
    sqrt_pou = np.sqrt(np.clip(pou, 0.0, None))

    Omega_lookup = np.zeros((n_sets, n_sets, 2, 2), dtype=float)
    for i in range(n_sets):
        Omega_lookup[i, i] = np.eye(2)

    for (a, b) in edge_set:
        if (a, b) not in Omega:
            raise ValueError(f"Omega missing edge {(a,b)} required by edges input.")
        Oab = np.asarray(Omega[(a, b)], dtype=float)
        if Oab.shape != (2, 2):
            raise ValueError(f"Omega[{(a,b)}] must be 2x2.")
        Omega_lookup[a, b] = Oab
        Omega_lookup[b, a] = Oab.T

    D = 2 * n_sets
    Phi = np.zeros((n_sets, n_samples, D, 2), dtype=float)
    for j in range(n_sets):
        mask_j = U[j].astype(float)[:, None, None]  # (n_samples,1,1)
        for i in range(n_sets):
            Oij = Omega_lookup[i, j][None, :, :]          # (1,2,2)
            w = sqrt_pou[i, :, None, None]                # (n_samples,1,1)
            Phi[j, :, 2 * i : 2 * i + 2, :] = (w * Oij) * mask_j
    return Phi


def get_classifying_map(
    Phi: np.ndarray,
    pou: np.ndarray,
    *,
    project: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    cl[s] = sum_j pou[j,s] * Phi[j,s] Phi[j,s]^T  (D×D).

    If project=True, replace by nearest rank-2 projection via top-2 eigenspace.
    """
    Phi = np.asarray(Phi, dtype=float)
    pou = np.asarray(pou, dtype=float)

    n_sets, n_samples, D, two = Phi.shape
    if two != 2:
        raise ValueError("Phi must have last dim = 2.")
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou shape mismatch.")

    cl = np.zeros((n_samples, D, D), dtype=float)
    for s in range(n_samples):
        for j in range(n_sets):
            w = float(pou[j, s])
            if w == 0.0:
                continue
            A = Phi[j, s]  # (D,2)
            cl[s] += w * (A @ A.T)

    if not project:
        return cl, 0.0

    P = np.zeros_like(cl)
    max_dist = 0.0
    for s in range(n_samples):
        Ps, dist = project_to_rank2_projection(cl[s])
        P[s] = Ps
        max_dist = max(max_dist, dist)
    return P, float(max_dist)


def project_frames_to_stiefel(
    *,
    U: np.ndarray,
    Phi: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Project Phi[j,s] onto V(2,D) over the plane P[s] via polar factor."""
    U = np.asarray(U, dtype=bool)
    Phi = np.asarray(Phi, dtype=float)
    P = np.asarray(P, dtype=float)

    n_sets, n_samples, D, _ = Phi.shape
    if P.shape != (n_samples, D, D):
        raise ValueError("P shape mismatch with Phi.")

    out = np.zeros_like(Phi)
    max_dist = 0.0
    for j in range(n_sets):
        for s in range(n_samples):
            if not U[j, s]:
                continue
            A = Phi[j, s]
            Uproj = polar_stiefel_projection(P[s], A)
            out[j, s] = Uproj
            max_dist = max(max_dist, float(np.linalg.norm(Uproj - A, ord="fro")))
    return out, float(max_dist)


def cocycle_from_frames(
    *,
    Phi_true: np.ndarray,
    edges: Iterable[Edge],
) -> Dict[Edge, np.ndarray]:
    """
    Omega_true[(j,k)][s] = Phi_true[j,s]^T Phi_true[k,s] (2×2), for each undirected edge j<k.
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    _, n_samples, _, _ = Phi_true.shape

    E = sorted({canon_edge(a, b) for (a, b) in edges if a != b})
    Omega_true: Dict[Edge, np.ndarray] = {}

    for (j, k) in E:
        arr = np.zeros((n_samples, 2, 2), dtype=float)
        for s in range(n_samples):
            arr[s] = Phi_true[j, s].T @ Phi_true[k, s]
        Omega_true[(j, k)] = arr

    return Omega_true


def cocycle_projection_distance(
    *,
    U: np.ndarray,
    Omega_simplicial: Dict[Edge, Mat2],
    Omega_true: Dict[Edge, np.ndarray],
    edges: Iterable[Edge],
) -> float:
    """max over overlaps of ||Omega_simplicial - Omega_true(s)||_F."""
    U = np.asarray(U, dtype=bool)
    E = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    max_diff = 0.0
    for (j, k) in E:
        if (j, k) not in Omega_simplicial or (j, k) not in Omega_true:
            continue
        O = np.asarray(Omega_simplicial[(j, k)], dtype=float)
        Ot = Omega_true[(j, k)]
        overlap = U[j] & U[k]
        idx = np.where(overlap)[0]
        for s in idx:
            max_diff = max(max_diff, float(np.linalg.norm(O - Ot[s], ord="fro")))
    return float(max_diff)


def witness_error(
    *,
    U: np.ndarray,
    f: np.ndarray,
    Omega_true: Dict[Edge, np.ndarray],
    edges: Iterable[Edge],
) -> float:
    """
    sup over overlaps of circular distance:
        dist( f[j,s], arg( Omega_{j<-k}(s) * unit(f[k,s]) ) )
    """
    U = np.asarray(U, dtype=bool)
    f = np.asarray(f, dtype=float)
    n_sets, n_samples = U.shape
    if f.shape != (n_sets, n_samples):
        raise ValueError("f shape mismatch with U.")

    E = sorted({canon_edge(a, b) for (a, b) in edges if a != b})
    max_err = 0.0

    for (j, k) in E:
        if (j, k) not in Omega_true:
            continue
        Ojk = Omega_true[(j, k)]  # (n_samples,2,2) mapping k->j when (j,k) is canonical
        overlap = U[j] & U[k]
        idx = np.where(overlap)[0]
        for s in idx:
            v = angles_to_unit(np.array([f[k, s]], dtype=float))[0]
            w = Ojk[s] @ v
            th = float(np.arctan2(w[1], w[0]))
            max_err = max(max_err, _circle_dist(float(f[j, s]), th))
    return float(max_err)


# ============================================================
# Bundle map construction 
# ============================================================

def build_bundle_map(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    f: np.ndarray,
    Phi_true: np.ndarray,
    Omega_true: Dict[Edge, np.ndarray],
    edges: Iterable[Edge],
    strict_semicircle: bool = True,
    semicircle_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a global map F: samples -> R^D.

    For each sample s:
      - For EACH active chart j, compute chartwise output F_j(s):
          transport all local angles into chart j,
          anchored semicircle average (default),
          output Phi_true[j,s] @ unit(mean_theta).
      - Global F[s] is chosen as F_{j0}(s) for a canonical chart j0 (first active).

    Returns:
      F:     (n_samples, D)
      pre_F: (n_sets, n_samples, D) storing all F_j(s) for active charts (zeros otherwise)
    """
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    f = np.asarray(f, dtype=float)
    Phi_true = np.asarray(Phi_true, dtype=float)

    n_sets, n_samples = U.shape
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou shape mismatch.")
    if f.shape != (n_sets, n_samples):
        raise ValueError("f shape mismatch.")
    if Phi_true.shape[:2] != (n_sets, n_samples) or Phi_true.shape[3] != 2:
        raise ValueError("Phi_true must have shape (n_sets,n_samples,D,2).")

    D = int(Phi_true.shape[2])
    Eset = {canon_edge(a, b) for (a, b) in edges if a != b}

    def omega_dir(j: int, k: int, s: int) -> Optional[np.ndarray]:
        """Omega_{j<-k}(s) from undirected store (a<b)."""
        if j == k:
            return np.eye(2)
        a, b = canon_edge(j, k)
        if (a, b) not in Omega_true:
            return None
        Oab = Omega_true[(a, b)][s]
        return Oab if (j == a and k == b) else Oab.T

    pre_F = np.zeros((n_sets, n_samples, D), dtype=float)
    F = np.zeros((n_samples, D), dtype=float)

    for s in range(n_samples):
        active = np.where(U[:, s])[0]
        if active.size == 0:
            continue

        # Compute all chartwise formulas F_j(s)
        for j_ in active:
            j = int(j_)
            transported_angles: List[float] = []
            weights: List[float] = []

            for k in range(n_sets):
                w = float(pou[k, s])
                if w <= 0.0 or (not U[k, s]):
                    continue

                O = omega_dir(j, k, s)
                if O is None:
                    continue

                # Apply Omega to the unit vector (this is the “S^1 action”),
                # then take arg/log (angle) of the resulting point on S^1.
                v = angles_to_unit(np.array([f[k, s]], dtype=float))[0]
                u = O @ v
                th = float(np.arctan2(u[1], u[0]))

                transported_angles.append(th)
                weights.append(w)

            if len(weights) == 0:
                continue

            ang = np.asarray(transported_angles, dtype=float)
            wts = np.asarray(weights, dtype=float)

            if strict_semicircle:
                # Deterministic anchor: transported angle corresponding to largest weight
                idx_max = int(np.argmax(wts))
                anchor = float(ang[idx_max])
                mean_theta = weighted_angle_mean_anchored(ang, wts, anchor=anchor, tol=semicircle_tol)
            else:
                z = np.sum((wts / wts.sum()) * np.exp(1j * ang))
                mean_theta = float(np.angle(z))

            out = Phi_true[j, s] @ np.array([np.cos(mean_theta), np.sin(mean_theta)], dtype=float)
            pre_F[j, s] = out

        # Define global F using the first active chart
        j0 = int(active[0])
        F[s] = pre_F[j0, s]

    return F, pre_F


# ============================================================
# Diagnostics: chart disagreement
# ============================================================

@dataclass
class ChartDisagreementStats:
    max: float

    def to_text(self, *, decimals: int = 3) -> str:
        r = int(decimals)
        return (
            "Chart disagreement:\n"
            "  Δ = max_{j,j'∈J_s} ||F_j(s) - F_{j'}(s)||_2\n"
            f"  max ≈ {self.max:.{r}f}\n"
        )


def chart_disagreement_stats(*, U: np.ndarray, pre_F: np.ndarray) -> ChartDisagreementStats:
    """Compute only max Δ(s) over samples."""
    U = np.asarray(U, dtype=bool)
    pre_F = np.asarray(pre_F, dtype=float)

    n_sets, n_samples = U.shape
    if pre_F.shape[0] != n_sets or pre_F.shape[1] != n_samples:
        raise ValueError("pre_F shape mismatch with U.")

    max_delta = 0.0

    for s in range(n_samples):
        active = np.where(U[:, s])[0]
        if active.size <= 1:
            continue
        V = pre_F[active, s, :]
        mx = 0.0
        for a in range(V.shape[0]):
            for b in range(a + 1, V.shape[0]):
                mx = max(mx, float(np.linalg.norm(V[a] - V[b])))
        max_delta = max(max_delta, mx)

    return ChartDisagreementStats(max=float(max_delta))


# ============================================================
# Public pipeline (refactored)
# ============================================================

@dataclass
class TrueFramesResult:
    Phi: np.ndarray               # (n_sets,n_samples,D,2) approximate frames
    cl: np.ndarray                # (n_samples,D,D) projected rank-2 projectors
    Phi_true: np.ndarray          # (n_sets,n_samples,D,2) Stiefel frames
    Omega_true: Dict[Edge, np.ndarray]
    cl_proj_dist: float
    frame_proj_dist: float


def build_true_frames(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    Omega: Dict[Edge, Mat2],
    edges: Optional[Iterable[Edge]] = None,
) -> TrueFramesResult:
    """Phi -> classifying map -> rank-2 projectors -> Stiefel projection -> Omega_true."""
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)

    if edges is None:
        edges = infer_edges_from_U(U)
    edges_list = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    Phi = get_local_frames(U=U, pou=pou, Omega=Omega, edges=edges_list)
    P, cl_proj_dist = get_classifying_map(Phi, pou, project=True)
    Phi_true, frame_proj_dist = project_frames_to_stiefel(U=U, Phi=Phi, P=P)
    Omega_true = cocycle_from_frames(Phi_true=Phi_true, edges=edges_list)

    return TrueFramesResult(
        Phi=Phi,
        cl=P,
        Phi_true=Phi_true,
        Omega_true=Omega_true,
        cl_proj_dist=float(cl_proj_dist),
        frame_proj_dist=float(frame_proj_dist),
    )


def apply_frame_reduction(
    *,
    Phi_true: np.ndarray,
    U: np.ndarray,
    reducer: FrameReducerConfig,
) -> Tuple[np.ndarray, Optional[FrameReductionReport]]:
    """Delegate to frame_reduction.py."""
    if reducer.method == "none":
        return Phi_true, None

    if int(reducer.d) <= 0:
        raise ValueError("FrameReducerConfig.d must be set for reduction.")

    if reducer.method == "subspace_pca":
        Phi_red, rep, _B = reduce_frames_subspace_pca(
            Phi_true=Phi_true,
            U=U,
            d=int(reducer.d),
            max_frames=reducer.max_frames,
            rng_seed=reducer.rng_seed,
        )
        return Phi_red, rep

    if reducer.method == "psc":
        Phi_red, rep, _alpha = reduce_frames_psc(
            Phi_true=Phi_true,
            U=U,
            d=int(reducer.d),
            max_frames=reducer.max_frames,
            rng_seed=reducer.rng_seed,
            verbosity=int(getattr(reducer, "psc_verbosity", 0)),
        )
        return Phi_red, rep

    raise ValueError(f"Unknown reducer.method: {reducer.method}")


@dataclass
class BundleMapReport:
    cl_proj_dist: float
    frame_proj_dist: float
    cocycle_proj_dist: float
    witness_err: float

    # optional extras
    chart_disagreement_max: Optional[float] = None
    reduction_mean_proj_err: Optional[float] = None
    reduction_method: Optional[str] = None
    reduction_d_out: Optional[int] = None

    def to_text(self, r: int = 3) -> str:
        IND = "  "
        LABEL_W = 30

        def _tline(label: str, content: str) -> str:
            return f"{IND}{label:<{LABEL_W}} {content}"

        lines: List[str] = []
        lines.append("=== Coordinatization ===")
        lines.append(_tline("cocycle proj dist:", f"{float(self.cocycle_proj_dist):.{r}f}"))
        lines.append(_tline("witness error (S^1):", f"{float(self.witness_err):.{r}f} rad"))
        lines.append(_tline("Grassmann proj dist:", f"{float(self.cl_proj_dist):.{r}f}"))
        lines.append(_tline("Stiefel proj dist:", f"{float(self.frame_proj_dist):.{r}f}"))

        if self.chart_disagreement_max is not None:
            lines.append(_tline("chart disagreement max:", f"{float(self.chart_disagreement_max):.{r}f}"))

        if self.reduction_mean_proj_err is not None:
            meth = self.reduction_method or "reduction"
            d_out = self.reduction_d_out
            if d_out is None:
                lines.append(_tline(f"{meth} mean proj err:", f"{float(self.reduction_mean_proj_err):.{r}f}"))
            else:
                lines.append(_tline(f"{meth} mean proj err:", f"{float(self.reduction_mean_proj_err):.{r}f}  (d={int(d_out)})"))

        return "\n".join(lines)

    def to_markdown(self, r: int = 3) -> str:
        md = (
            "### Coordinatization\n\n"
            r"**Cocycle projection distance**  " "\n"
            r"$d_{\infty}(\Omega,\Pi(\Omega))"
            r"=\sup_{(jk)}\sup_{b\in\pi(X)\cap(U_j\cap U_k)}"
            r"\ \|\Omega_{jk}(b)-\Pi(\Omega)_{jk}(b)\|_{F}$"
            f"\n\n- $\\approx {float(self.cocycle_proj_dist):.{r}f}$\n\n"
            r"**Witness quality**  " "\n"
            r"$\sup_{(jk)}\sup_{x:\ \pi(x)\in U_j\cap U_k}"
            r"\ d_{\mathbb{S}^{1}}\!\left(f_j(x),\ \Omega_{jk}(\pi(x))f_k(x)\right)$"
            f"\n\n- $\\approx {float(self.witness_err):.{r}f}$ (rad)\n\n"
            r"**Auxiliary diagnostics**"
            f"\n\n- Grassmann projection distance $\\approx {float(self.cl_proj_dist):.{r}f}$"
            f"\n- Stiefel projection distance $\\approx {float(self.frame_proj_dist):.{r}f}$\n"
        )

        if self.chart_disagreement_max is not None:
            md += (
                "\n"
                r"**Chart disagreement (max)**  " "\n"
                r"$\max_{s}\ \Delta(s),\quad \Delta(s)=\max_{j,j'\in J_s}\|F_j(s)-F_{j'}(s)\|_2$"
                f"\n\n- $\\approx {float(self.chart_disagreement_max):.{r}f}$\n"
            )

        if self.reduction_mean_proj_err is not None:
            meth = self.reduction_method or "reduction"
            d_out = self.reduction_d_out
            if d_out is None:
                md += (
                    "\n"
                    f"**Frame reduction** (`{meth}`)\n\n"
                    r"- Mean projection error $\mathbb{E}\,\|Y - BB^{\mathsf{T}}Y\|_{F}$"
                    f" $\\approx {float(self.reduction_mean_proj_err):.{r}f}$\n"
                )
            else:
                md += (
                    "\n"
                    f"**Frame reduction** (`{meth}`, $d={int(d_out)}$)\n\n"
                    r"- Mean projection error $\mathbb{E}\,\|Y - BB^{\mathsf{T}}Y\|_{F}$"
                    f" $\\approx {float(self.reduction_mean_proj_err):.{r}f}$\n"
                )

        return md


    
def show_bundle_map_summary(
    report: BundleMapReport,
    *,
    show: bool = True,
    mode: SummaryMode = "auto",
    rounding: int = 3,
) -> str:
    """
    Pretty summary, matching characteristic-class `show_summary`.

    mode:
      - "auto": show LaTeX if available, else print text
      - "text": print text only
      - "latex": show LaTeX only (best-effort; falls back to text)
      - "both": show LaTeX (if available) AND print text
    """
    text = report.to_text(r=rounding)

    if not show:
        return text

    did_latex = False
    if mode in {"latex", "auto", "both"}:
        did_latex = _display_bundle_map_summary_latex(report, rounding=rounding)

    if mode == "both" or mode == "text" or (mode == "auto" and not did_latex) or (mode == "latex" and not did_latex):
        print("\n" + text + "\n")

    return text


def _display_bundle_map_summary_latex(report: BundleMapReport, *, rounding: int = 3) -> bool:
    """Best-effort Math display; returns True iff succeeded."""
    try:
        from IPython.display import display, Markdown  # type: ignore
    except Exception:
        return False

    try:
        display(Markdown(report.to_markdown(r=rounding)))
        return True
    except Exception:
        return False
    

def get_bundle_map(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    f: np.ndarray,
    Omega: Dict[Edge, Mat2],
    edges: Optional[Iterable[Edge]] = None,
    strict_semicircle: bool = True,
    semicircle_tol: float = 1e-8,
    reducer: Optional[FrameReducerConfig] = None,
    show_summary: bool = True,
    compute_chart_disagreement: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[Edge, np.ndarray], np.ndarray, BundleMapReport]:
    """
    End-to-end coordinatization.

    Stages:
      (A) build_true_frames(): Phi, Π(classifying map), Phi_true, Omega_true
      (B) diagnostics in ambient space
      (C) optional frame reduction Phi_true -> Phi_used (and rebuild Omega_true on reduced frames)
      (D) build_bundle_map() to produce F and pre_F (with anchored averaging)

    Returns:
      F, pre_F, Omega_used, Phi_used, report
    """
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    f = np.asarray(f, dtype=float)

    n_sets, n_samples = U.shape
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou shape mismatch with U.")
    if f.shape != (n_sets, n_samples):
        raise ValueError("f shape mismatch with U.")

    if edges is None:
        edges = infer_edges_from_U(U)
    edges_list = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    # (A) true frames
    tf = build_true_frames(U=U, pou=pou, Omega=Omega, edges=edges_list)

    # (B) diagnostics in ambient space
    cocycle_proj_dist = cocycle_projection_distance(
        U=U, Omega_simplicial=Omega, Omega_true=tf.Omega_true, edges=edges_list
    )
    wit_err = witness_error(U=U, f=f, Omega_true=tf.Omega_true, edges=edges_list)

    # (C) optional reduction
    Phi_used = tf.Phi_true
    Omega_used = tf.Omega_true
    reduction_report: Optional[FrameReductionReport] = None

    if reducer is not None and reducer.method != "none":
        Phi_used, reduction_report = apply_frame_reduction(Phi_true=tf.Phi_true, U=U, reducer=reducer)
        Omega_used = cocycle_from_frames(Phi_true=Phi_used, edges=edges_list)

    # (D) bundle map (with anchored averaging)
    F, pre_F = build_bundle_map(
        U=U,
        pou=pou,
        f=f,
        Phi_true=Phi_used,
        Omega_true=Omega_used,
        edges=edges_list,
        strict_semicircle=bool(strict_semicircle),
        semicircle_tol=float(semicircle_tol),
    )

    # ----------------------------
    # (E) chart disagreement summary 
    # ----------------------------
    cd_max: Optional[float] = None
    if compute_chart_disagreement:
        cd = chart_disagreement_stats(U=U, pre_F=pre_F)
        cd_max = float(cd.max)

    # ----------------------------
    # (F) reduction summary fields 
    # ----------------------------
    red_mean: Optional[float] = None
    red_method: Optional[str] = None
    red_d: Optional[int] = None

    if reduction_report is not None:
        # Be robust to whatever attribute names your FrameReductionReport uses
        red_mean = getattr(reduction_report, "mean_proj_err", None)
        if red_mean is None:
            red_mean = getattr(reduction_report, "mean_projection_error", None)

        red_method = getattr(reduction_report, "method", None)
        if red_method is None:
            red_method = getattr(reduction_report, "name", None)

        red_d = getattr(reduction_report, "d_out", None)
        if red_d is None:
            red_d = getattr(reduction_report, "d", None)

        if red_mean is not None:
            red_mean = float(red_mean)
        if red_d is not None:
            red_d = int(red_d)

    report = BundleMapReport(
        cl_proj_dist=float(tf.cl_proj_dist),
        frame_proj_dist=float(tf.frame_proj_dist),
        cocycle_proj_dist=float(cocycle_proj_dist),
        witness_err=float(wit_err),
        chart_disagreement_max=cd_max,
        reduction_mean_proj_err=red_mean,
        reduction_method=red_method,
        reduction_d_out=red_d,
    )

    if show_summary:
        show_bundle_map_summary(report, show=True, mode="auto", rounding=3)

    return F, pre_F, Omega_used, Phi_used, report
