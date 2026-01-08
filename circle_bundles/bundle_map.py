# circle_bundles/bundle_map.py
"""
bundle_map.py

Global bundle coordinatization pipeline.

Inputs
------
- Cover membership U:      (n_sets, n_samples) bool
- Partition of unity pou:  (n_sets, n_samples) float, columns sum to 1 on covered samples
- Local angles f:          (n_sets, n_samples) float radians, meaningful where U[j,s]=True
- Discrete O(2) cocycle Omega on nerve edges (j<k):
      Omega[(j,k)] ∈ O(2) maps k-coordinates -> j-coordinates

Outputs
-------
- F:      (n_samples, D)  global coordinates (D = 2*n_sets by default)
- pre_F:  (n_sets, n_samples, D) chartwise outputs (mostly zero)
- Omega_true: dict (j,k)->(n_samples,2,2) pointwise induced cocycle from true frames
- Phi_true:  (n_sets, n_samples, D, 2) Stiefel frames in V(2,D)
- report: diagnostics summary

Key features
------------
1) Strict semicircle averaging (principal-branch log): by default we require transported
   angles to lie in an open semicircle; otherwise we raise.

2) Grassmann projection + Stiefel polar projection:
   - Build approximate frames Phi from POU + Omega
   - Build classifying map cl(s) = Σ_j ρ_j(s) Phi_j(s) Phi_j(s)^T
   - Project cl(s) to nearest rank-2 projector Π(cl(s)) via top-2 eigenspace
   - Project frames to Stiefel on that plane via polar factor

3) Optional equivariant dimensionality reduction on frames (PSC hook):
   We expose a clean "reduce frames to V(2,d)" stage, with:
   - an always-available, O(2)-equivariant subspace PCA baseline (method="subspace_pca")
   - an optional PSC-package hook (method="psc") if you want to wire in PSC later.
     (I keep it as a hook because PSC’s public API in the wild is easy to mismatch.)

If you don’t pass a reducer config, the pipeline is exactly the full ambient one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np

from .combinatorics import Edge, canon_edge
from .o2_cocycle import angles_to_unit

Mat2 = np.ndarray


# ============================================================
# Angle utilities (strict semicircle averaging)
# ============================================================

def _wrap_to_pi(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float(((theta + np.pi) % (2.0 * np.pi)) - np.pi)


def _circle_dist(a: float, b: float) -> float:
    """Geodesic distance on S^1 between angles a,b (radians)."""
    return abs(_wrap_to_pi(a - b))


def weighted_angle_mean_with_semicircle_check(
    angles: np.ndarray,
    weights: np.ndarray,
    *,
    tol: float = 1e-8,
) -> float:
    """
    Weighted mean of angles using a consistent principal-branch lift,
    but ONLY if the angles lie in an open semicircle.

    Semicircle test:
      Let max_gap be the largest circular gap between sorted angles.
      A set is contained in an open semicircle  <=> max_gap > pi.

    Returns:
      mean angle in (-pi, pi]

    Raises:
      ValueError if not contained in an open semicircle.
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

    idx = np.argsort(angles)
    a = angles[idx]
    w = w[idx]

    gaps = np.diff(a, append=a[0] + 2.0 * np.pi)
    max_gap = float(np.max(gaps))

    if max_gap <= np.pi + tol:
        raise ValueError(
            f"Angles are not contained in an open semicircle (max_gap={max_gap:.6f} <= pi)."
        )

    # Choose branch cut inside the largest empty arc (midpoint of that gap)
    k = int(np.argmax(gaps))
    cut = float((a[k] + 0.5 * gaps[k]) % (2.0 * np.pi))

    # Unwrap around cut into (-pi, pi]
    a_unwrapped = ((a - cut + np.pi) % (2.0 * np.pi)) - np.pi
    mean_unwrapped = float(np.sum(w * a_unwrapped))

    return _wrap_to_pi(mean_unwrapped)


# ============================================================
# Linear algebra helpers (rank-2 projector + Stiefel polar)
# ============================================================

def _top2_eigvecs_sym(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return top-2 eigenvalues/eigenvectors of a symmetric matrix A using numpy.linalg.eigh.
    (Full eigendecomposition; robust.)
    """
    vals, vecs = np.linalg.eigh(A)  # ascending
    order = np.argsort(vals)[::-1]
    vals2 = vals[order[:2]]
    vecs2 = vecs[:, order[:2]]
    return vals2, vecs2


def project_to_rank2_projection(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Your construction Π(A):

      As = (A + A^T)/2
      As = Λ D Λ^T (eigs in decreasing order)
      Π(A) = Λ J2 Λ^T  where J2 has first two diagonal entries = 1

    Returns:
      P (d,d) rank-2 orthogonal projection
      dist = ||As - P||_F
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    As = 0.5 * (A + A.T)
    _, V2 = _top2_eigvecs_sym(As)     # (d,2) top-2 eigenvectors
    P = V2 @ V2.T                    # same as Λ J2 Λ^T
    dist = float(np.linalg.norm(As - P, ord="fro"))
    return P, dist


def polar_stiefel_projection(P: np.ndarray, A: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Stiefel projection via polar factor:

      Given (P, A) with rank(PA)=2, define
          U = (PA) ( (PA)^T (PA) )^{-1/2}
      Then U has orthonormal columns (lies in V(2,d)) and spans range(PA).

    Inputs:
      P: (d,d) projection
      A: (d,2)

    Returns:
      U: (d,2)
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
    U = PA @ inv_sqrt
    return U


# ============================================================
# Core construction pieces
# ============================================================

def infer_edges_from_U(U: np.ndarray) -> List[Edge]:
    """Default 1-skeleton: edges (j,k) where U_j ∩ U_k contains at least one sample."""
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
    Build approximate local frames Phi[j, s] in R^{2n x 2}:

      Phi_j(s) = [ sqrt(rho_0(s)) Omega_{0j} ;
                   ...
                   sqrt(rho_{n-1}(s)) Omega_{n-1,j} ]

    Convention for Omega:
      Omega[(a,b)] for a<b maps b -> a.
      For directed use, Omega_{i<-j} is:
        if i==j: I
        if i<j:  Omega[(i,j)]
        if i>j:  Omega[(j,i)]^T

    Returns:
      Phi: (n_sets, n_samples, 2*n_sets, 2)
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

    Phi = np.zeros((n_sets, n_samples, 2 * n_sets, 2), dtype=float)
    for j in range(n_sets):
        mask_j = U[j].astype(float)[:, None, None]  # (n_samples,1,1)
        for i in range(n_sets):
            Oij = Omega_lookup[i, j][None, :, :]            # (1,2,2)
            w = sqrt_pou[i, :, None, None]                  # (n_samples,1,1)
            Phi[j, :, 2 * i : 2 * i + 2, :] = (w * Oij) * mask_j

    return Phi


def get_classifying_map(
    Phi: np.ndarray,
    pou: np.ndarray,
    *,
    project: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    cl[s] = sum_j pou[j,s] * Phi[j,s] Phi[j,s]^T  in R^{D x D}, D=2n.

    If project=True, apply Π(cl[s]) (rank-2 projector from top-2 eigenspace).

    Returns:
      P: (n_samples, D, D)  projection matrices (if project=True) else raw cl
      max_dist: max ||sym(cl[s]) - P[s]||_F if project=True, else 0
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
    """
    For each (j,s) with U[j,s]=True, project Phi[j,s] -> polar_stiefel_projection(P[s], Phi[j,s]).
    Else keep zeros.

    Returns:
      Phi_true: same shape as Phi
      max_proj_dist: max ||Phi_true - Phi||_F over (j,s) where U[j,s]=True
    """
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
    Omega_true[(j,k)][s] = Phi_true[j,s]^T Phi_true[k,s]  (2x2),
    for each undirected edge (j,k) with j<k.

    Returns:
      dict mapping (j,k) -> (n_samples, 2, 2)
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    n_sets, n_samples, _, _ = Phi_true.shape

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
    """
    max over edges (j,k) and samples s in overlap:
      || Omega_simplicial[(j,k)] - Omega_true[(j,k)][s] ||_F
    """
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
        if idx.size == 0:
            continue
        diffs = np.linalg.norm(Ot[idx] - O[None, :, :], axis=(1, 2))
        max_diff = max(max_diff, float(np.max(diffs)))
    return float(max_diff)


def witness_error(
    *,
    U: np.ndarray,
    f: np.ndarray,
    Omega_true: Dict[Edge, np.ndarray],
    edges: Iterable[Edge],
) -> float:
    """
    sup over edges (j,k) and samples s in overlap of circular distance:
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
        Ojk = Omega_true[(j, k)]
        overlap = U[j] & U[k]
        idx = np.where(overlap)[0]
        for s in idx:
            v = angles_to_unit(np.array([f[k, s]], dtype=float))[0]
            w = Ojk[s] @ v
            th = float(np.arctan2(w[1], w[0]))
            max_err = max(max_err, _circle_dist(float(f[j, s]), th))
    return float(max_err)


# ============================================================
# Optional equivariant dimensionality reduction on frames
# ============================================================

ReduceMethod = Literal["none", "subspace_pca", "psc"]


@dataclass
class FrameReducerConfig:
    """
    Optional frame reduction stage:
      Phi_true (D,2)  ->  Phi_red (d,2) with O(2)-equivariance.

    method:
      - "subspace_pca": always-available, O(2)-equivariant baseline:
          compute mean projector M = avg(Y Y^T), take top-d eigenspace B,
          project frames by B^T Y and re-orthonormalize (polar).
      - "psc": hook for PSC package (arXiv:2309.10775). If PSC is not importable
        or the expected API doesn't match, we raise with a helpful message.
      - "none": skip

    d:
      target ambient dimension after reduction (d >= 2).
    """
    method: ReduceMethod = "none"
    d: int = 0
    # If you want to reduce using only a subset of frames (for speed), set max_frames.
    max_frames: Optional[int] = None
    rng_seed: int = 0


@dataclass
class FrameReductionReport:
    method: ReduceMethod
    D_in: int
    d_out: int
    mean_recon_err: float
    p95_recon_err: float
    max_recon_err: float

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("\n" + "=" * 12 + " Frame Reduction " + "=" * 12 + "\n")
        lines.append(f"method = {self.method}")
        lines.append(f"ambient: D_in = {self.D_in},  d_out = {self.d_out}")
        lines.append("Reconstruction error ||Y - BB^T Y||_F (over used frames):")
        lines.append(f"  mean ≈ {self.mean_recon_err:.6g}")
        lines.append(f"  p95  ≈ {self.p95_recon_err:.6g}")
        lines.append(f"  max  ≈ {self.max_recon_err:.6g}")
        lines.append("\n" + "=" * 40 + "\n")
        return "\n".join(lines)


def _collect_frames_as_list(
    Phi_true: np.ndarray,
    U: np.ndarray,
    *,
    max_frames: Optional[int] = None,
    rng_seed: int = 0,
) -> List[np.ndarray]:
    """
    Collect a list of frames Y = Phi_true[j,s] (D,2) for which U[j,s]=True.
    Optionally subsample to at most max_frames.
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    U = np.asarray(U, dtype=bool)
    n_sets, n_samples, D, two = Phi_true.shape
    if two != 2:
        raise ValueError("Expected frames in V(2,D): last dim must be 2.")

    Js, Ss = np.where(U)
    frames = [Phi_true[int(j), int(s)] for (j, s) in zip(Js, Ss)]

    if max_frames is not None and len(frames) > max_frames:
        rng = np.random.default_rng(int(rng_seed))
        idx = rng.choice(len(frames), size=int(max_frames), replace=False)
        frames = [frames[int(i)] for i in idx]

    return frames


def subspace_pca_fit(
    frames: Sequence[np.ndarray],
    *,
    d: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit O(2)-equivariant subspace PCA on frames in V(2,D).

    Returns:
      B: (D,d) orthonormal basis of chosen subspace
      evals: (d,) top eigenvalues of mean projector M
    """
    if len(frames) == 0:
        raise ValueError("Need at least one frame to fit a reducer.")
    Y0 = np.asarray(frames[0], dtype=float)
    if Y0.ndim != 2 or Y0.shape[1] != 2:
        raise ValueError("Each frame must have shape (D,2).")
    D = int(Y0.shape[0])
    if not (2 <= d <= D):
        raise ValueError(f"Need 2 <= d <= D. Got d={d}, D={D}.")

    # Mean projector M = avg(Y Y^T). This is invariant under Y -> YQ for Q∈O(2).
    M = np.zeros((D, D), dtype=float)
    for Y in frames:
        Y = np.asarray(Y, dtype=float)
        M += Y @ Y.T
    M /= float(len(frames))

    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    B = vecs[:, :d]          # (D,d) orthonormal
    evals = vals[:d]
    return B, evals


def subspace_pca_transform_frame(B: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Project a frame Y (D,2) to d-dim: Yd = B^T Y (d,2), then re-orthonormalize columns.
    """
    B = np.asarray(B, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("Y must have shape (D,2).")
    if B.ndim != 2:
        raise ValueError("B must be (D,d).")
    if B.shape[0] != Y.shape[0]:
        raise ValueError("B and Y have incompatible ambient dimensions.")

    Yd = B.T @ Y  # (d,2)

    # Polar orthonormalization in R^d (same formula as before, with P=I_d):
    AtA = Yd.T @ Yd
    eigvals, eigvecs = np.linalg.eigh(AtA)
    eigvals = np.clip(eigvals, 1e-12, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    Ud = Yd @ inv_sqrt
    return Ud


def reduce_frames_subspace_pca(
    *,
    Phi_true: np.ndarray,
    U: np.ndarray,
    d: int,
    max_frames: Optional[int] = None,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, FrameReductionReport, np.ndarray]:
    """
    Reduce all frames Phi_true[j,s] (D,2) -> Phi_red[j,s] (d,2) using subspace PCA.

    Returns:
      Phi_red: (n_sets, n_samples, d, 2)
      report
      B: (D,d) basis used
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    U = np.asarray(U, dtype=bool)
    n_sets, n_samples, D, two = Phi_true.shape
    if two != 2:
        raise ValueError("Phi_true last dim must be 2.")
    if not (2 <= d <= D):
        raise ValueError(f"Need 2 <= d <= D. Got d={d}, D={D}.")

    frames_fit = _collect_frames_as_list(Phi_true, U, max_frames=max_frames, rng_seed=rng_seed)
    B, _ = subspace_pca_fit(frames_fit, d=d)

    Phi_red = np.zeros((n_sets, n_samples, d, 2), dtype=float)

    # recon stats (on all used frames)
    recon_errs: List[float] = []
    for j in range(n_sets):
        for s in range(n_samples):
            if not U[j, s]:
                continue
            Y = Phi_true[j, s]                # (D,2)
            Yproj = (B @ (B.T @ Y))           # (D,2)
            recon_errs.append(float(np.linalg.norm(Y - Yproj, ord="fro")))
            Phi_red[j, s] = subspace_pca_transform_frame(B, Y)

    if len(recon_errs) == 0:
        raise ValueError("No frames were available to reduce (U has no True entries?).")

    errs = np.asarray(recon_errs, dtype=float)
    rep = FrameReductionReport(
        method="subspace_pca",
        D_in=D,
        d_out=int(d),
        mean_recon_err=float(np.mean(errs)),
        p95_recon_err=float(np.quantile(errs, 0.95)),
        max_recon_err=float(np.max(errs)),
    )
    return Phi_red, rep, B


def reduction_curve_subspace_pca(
    *,
    Phi_true: np.ndarray,
    U: np.ndarray,
    dims: Sequence[int],
    max_frames: Optional[int] = None,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a simple "explained variance" style curve for subspace PCA:
      for each d in dims, fit B_d and compute mean reconstruction error
        mean ||Y - B_d B_d^T Y||_F^2  over sampled frames.
    Returns:
      dims_arr, mean_sq_err
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    U = np.asarray(U, dtype=bool)

    frames_fit = _collect_frames_as_list(Phi_true, U, max_frames=max_frames, rng_seed=rng_seed)
    if len(frames_fit) == 0:
        raise ValueError("No frames available for curve computation.")

    D = int(frames_fit[0].shape[0])
    dims_arr = np.asarray(list(dims), dtype=int)
    if np.any(dims_arr < 2) or np.any(dims_arr > D):
        raise ValueError(f"All dims must satisfy 2 <= d <= D={D}.")

    mean_sq_err = np.zeros_like(dims_arr, dtype=float)

    for t, d in enumerate(dims_arr):
        B, _ = subspace_pca_fit(frames_fit, d=int(d))
        se: List[float] = []
        for Y in frames_fit:
            Yproj = B @ (B.T @ Y)
            se.append(float(np.linalg.norm(Y - Yproj, ord="fro") ** 2))
        mean_sq_err[t] = float(np.mean(se))

    return dims_arr, mean_sq_err


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
    Build global map F in ambient dimension D:

    For each sample s:
      - choose a chart j with U[j,s]=True (first such j),
      - transport all local angles f[k,s] into chart j using Omega_true[j<-k](s),
      - compute a weighted mean angle using weights pou[k,s],
        with strict semicircle check by default,
      - output Phi_true[j,s] @ unit(mean_angle) in R^D.

    Returns:
      F:     (n_samples, D)
      pre_F: (n_sets, n_samples, D) (mostly zero; filled at chosen charts)
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
    if Phi_true.shape[:2] != (n_sets, n_samples):
        raise ValueError("Phi_true shape mismatch.")

    D = int(Phi_true.shape[2])
    if Phi_true.shape[3] != 2:
        raise ValueError("Phi_true must have shape (n_sets,n_samples,D,2).")

    E = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    # directed cocycle at (s): omega(j<-k)
    def omega_dir(j: int, k: int, s: int) -> Optional[np.ndarray]:
        if j == k:
            return np.eye(2)
        a, b = canon_edge(j, k)
        if (a, b) not in Omega_true:
            return None
        Oab = Omega_true[(a, b)][s]
        if j == a and k == b:
            return Oab
        else:
            return Oab.T

    pre_F = np.zeros((n_sets, n_samples, D), dtype=float)
    F = np.zeros((n_samples, D), dtype=float)

    for s in range(n_samples):
        js = np.where(U[:, s])[0]
        if js.size == 0:
            continue
        j = int(js[0])

        transported_angles: List[float] = []
        weights: List[float] = []

        for k in range(n_sets):
            w = float(pou[k, s])
            if w <= 0.0:
                continue
            if not U[k, s]:
                continue

            O = omega_dir(j, k, s)
            if O is None:
                continue

            v = angles_to_unit(np.array([f[k, s]], dtype=float))[0]
            u = O @ v
            th = float(np.arctan2(u[1], u[0]))
            transported_angles.append(th)
            weights.append(w)

        if len(weights) == 0:
            continue

        transported_angles_arr = np.asarray(transported_angles, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)

        if strict_semicircle:
            mean_theta = weighted_angle_mean_with_semicircle_check(
                transported_angles_arr,
                weights_arr,
                tol=semicircle_tol,
            )
        else:
            z = np.sum((weights_arr / weights_arr.sum()) * np.exp(1j * transported_angles_arr))
            mean_theta = float(np.angle(z))

        out = Phi_true[j, s] @ np.array([np.cos(mean_theta), np.sin(mean_theta)], dtype=float)
        pre_F[j, s] = out
        F[s] = out

    return F, pre_F


# ============================================================
# Summaries / diagnostics
# ============================================================

@dataclass
class BundleMapReport:
    cl_proj_dist: float
    frame_proj_dist: float
    cocycle_proj_dist: float
    witness_err: float

    # optional reduction info
    reduction: Optional[FrameReductionReport] = None

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append("\n" + "=" * 12 + " Coordinatization " + "=" * 12 + "\n")
        lines.append(
            "Cocycle projection distance:\n"
            "  d_∞(Ω, Π(Ω)) = sup_(jk) sup_(b in π(X)∩(U_j∩U_k)) ||Ω_jk(b) - Π(Ω)_jk(b)||_F\n"
            f"  ≈ {self.cocycle_proj_dist:.6g}"
        )
        lines.append("")
        lines.append(
            "Witness quality (S^1 error):\n"
            "  sup_(jk) sup_(x with π(x)∈U_j∩U_k) d_{S^1}( f_j(x), Ω_jk(π(x)) f_k(x) )\n"
            f"  ≈ {self.witness_err:.6g}  (radians)"
        )
        lines.append("")
        lines.append(
            "Auxiliary diagnostics:\n"
            f"  classifying-map projection dist (Grassmann) ≈ {self.cl_proj_dist:.6g}\n"
            f"  frame projection dist (polar/Stiefel)      ≈ {self.frame_proj_dist:.6g}"
        )
        if self.reduction is not None:
            lines.append(self.reduction.to_text())
        lines.append("\n" + "=" * 38 + "\n")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        md = (
            "### Coordinatization\n\n"
            r"**Cocycle projection distance:**  "
            r"$d_{\infty}(\Omega,\Pi(\Omega))"
            r"= \sup_{(jk)\in\mathcal{N}(\mathcal{U})^1}\ \sup_{b\in \pi(X)\cap(U_j\cap U_k)}"
            r"\ \|\Omega_{jk}(b)-\Pi(\Omega)_{jk}(b)\|_{F}$"
            f"\n\n- $\\approx {self.cocycle_proj_dist:.6g}$\n\n"
            r"**Witness quality:**  "
            r"$\sup_{(jk)}\sup_{x:\ \pi(x)\in U_j\cap U_k}\ d_{\mathbb{S}^{1}}\!\big(f_j(x),\ \Omega_{jk}(\pi(x))f_k(x)\big)$"
            f"\n\n- $\\approx {self.witness_err:.6g}$ (radians)\n\n"
            r"**Auxiliary diagnostics:**"
            f"\n\n- Classifying-map projection distance $\\approx {self.cl_proj_dist:.6g}$"
            f"\n- Frame projection distance $\\approx {self.frame_proj_dist:.6g}$\n"
        )
        if self.reduction is not None:
            md += "\n---\n\n" + "#### Frame Reduction\n\n"
            md += (
                f"- method: `{self.reduction.method}`\n"
                f"- $D_\\mathrm{{in}}={self.reduction.D_in}$, $d_\\mathrm{{out}}={self.reduction.d_out}$\n"
                f"- mean recon err $\\approx {self.reduction.mean_recon_err:.6g}$\n"
                f"- p95 recon err $\\approx {self.reduction.p95_recon_err:.6g}$\n"
                f"- max recon err $\\approx {self.reduction.max_recon_err:.6g}$\n"
            )
        return md


def show_bundle_map_summary(
    report: BundleMapReport,
    *,
    latex: str | bool = "auto",   # "auto" | True | False
    verbose: bool = True,
) -> None:
    if not verbose:
        return

    want_latex = (latex is True) or (latex == "auto")
    if want_latex and latex is not False:
        try:
            from IPython.display import display, Markdown  # type: ignore
            display(Markdown(report.to_markdown()))
            return
        except Exception:
            pass

    print(report.to_text())


# ============================================================
# Public pipeline (refactored)
# ============================================================

@dataclass
class TrueFramesResult:
    """
    Convenient breakpoint in the pipeline (this is where PSC/reduction can happen).
    """
    Phi: np.ndarray               # (n_sets,n_samples,D,2) approximate
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
    """
    Build frames up through the "true frames" stage:
      Phi -> classifying map -> rank-2 projectors -> Stiefel projection -> Omega_true.
    """
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    n_sets, _ = U.shape

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
    """
    Apply optional reduction to Phi_true (D,2)->(d,2).
    Returns reduced frames and a reduction report (or None if no reduction).
    """
    if reducer.method == "none":
        return Phi_true, None

    if reducer.method == "subspace_pca":
        if reducer.d is None or int(reducer.d) <= 0:
            raise ValueError("FrameReducerConfig.d must be set for reduction.")
        Phi_red, rep, _B = reduce_frames_subspace_pca(
            Phi_true=Phi_true,
            U=U,
            d=int(reducer.d),
            max_frames=reducer.max_frames,
            rng_seed=reducer.rng_seed,
        )
        return Phi_red, rep

    if reducer.method == "psc":
        # Hook: you can wire in PSC here once you settle the exact interface for k=2.
        # I’m deliberately failing loudly rather than silently doing the wrong thing.
        raise NotImplementedError(
            "PSC hook not wired yet for k=2 frames. "
            "Once you confirm PSC’s expected tensor shape for V(2,D) data, we’ll drop it in here "
            "(and keep subspace_pca as a baseline)."
        )

    raise ValueError(f"Unknown reducer.method: {reducer.method}")


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
    latex: str | bool = "auto",
) -> Tuple[np.ndarray, np.ndarray, Dict[Edge, np.ndarray], np.ndarray, BundleMapReport]:
    """
    End-to-end coordinatization.

    Stages:
      (A) build_true_frames(): Phi, Π(classifying map), Phi_true, Omega_true
      (B) diagnostics: cocycle_proj_dist, witness_err
      (C) optional frame reduction Phi_true -> Phi_red (and rebuild Omega_true on reduced frames)
      (D) build_bundle_map() to produce F and pre_F

    Returns:
      F, pre_F, Omega_true, Phi_true_used, report
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

    # (B) diagnostics in the ambient space
    cocycle_proj_dist = cocycle_projection_distance(
        U=U, Omega_simplicial=Omega, Omega_true=tf.Omega_true, edges=edges_list
    )
    wit_err = witness_error(U=U, f=f, Omega_true=tf.Omega_true, edges=edges_list)

    # (C) optional reduction
    reduction_report: Optional[FrameReductionReport] = None
    Phi_used = tf.Phi_true
    Omega_used = tf.Omega_true

    if reducer is not None and reducer.method != "none":
        Phi_used, reduction_report = apply_frame_reduction(Phi_true=tf.Phi_true, U=U, reducer=reducer)
        # rebuild induced cocycle in reduced ambient space
        Omega_used = cocycle_from_frames(Phi_true=Phi_used, edges=edges_list)

    # (D) bundle map
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

    report = BundleMapReport(
        cl_proj_dist=float(tf.cl_proj_dist),
        frame_proj_dist=float(tf.frame_proj_dist),
        cocycle_proj_dist=float(cocycle_proj_dist),
        witness_err=float(wit_err),
        reduction=reduction_report,
    )
    if show_summary:
        show_bundle_map_summary(report, latex=latex, verbose=True)

    return F, pre_F, Omega_used, Phi_used, report
