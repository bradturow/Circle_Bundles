# circle_bundles/frame_reduction.py
"""
frame_reduction.py

Optional O(2)-equivariant dimensionality reduction for Stiefel frames Y ∈ V(2, D).

This module is intentionally separate from bundle_map.py so that:
- bundle_map.py stays focused on the bundle/gluing math
- PSC is an optional dependency (imported only inside PSC functions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Literal

import numpy as np

ReduceMethod = Literal["none", "subspace_pca", "psc"]



# ============================================================
# Reports / config
# ============================================================

@dataclass
class FrameReducerConfig:
    """
    method:
      - "none": no reduction
      - "subspace_pca": always-available, O(2)-equivariant baseline
      - "psc": PSC package hook (HarlinLee/PSC)

    d:
      target ambient dimension after reduction (2 <= d <= D).

    max_frames:
      optional subsampling for speed (fit reducer on at most max_frames frames).
    """
    method: ReduceMethod = "none"
    d: int = 0
    max_frames: Optional[int] = None
    rng_seed: int = 0
    psc_verbosity: int = 0


@dataclass
class FrameReductionReport:
    method: ReduceMethod
    D_in: int
    d_out: int
    mean_proj_err: float  # renamed / clarified

    def to_text(self, *, decimals: int = 3) -> str:
        r = int(decimals)
        return (
            "\n"
            + "=" * 12
            + " Frame Reduction "
            + "=" * 12
            + "\n\n"
            + f"Method: {self.method}\n"
            + f"Ambient dim: D={self.D_in} → d={self.d_out}\n"
            + f"Mean projection error (||Y - BB^T Y||_F): {self.mean_proj_err:.{r}f}\n"
            + "\n"
            + "=" * 40
            + "\n"
        )

# ============================================================
# Helpers
# ============================================================

def _polar_orthonormalize_2cols(Yd: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Nearest Stiefel frame to Yd (d,2) via polar factor."""
    Yd = np.asarray(Yd, dtype=float)
    if Yd.ndim != 2 or Yd.shape[1] != 2:
        raise ValueError("Yd must have shape (d,2).")
    AtA = Yd.T @ Yd
    eigvals, eigvecs = np.linalg.eigh(AtA)
    eigvals = np.clip(eigvals, eps, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return Yd @ inv_sqrt


def _collect_frames_as_list(
    Phi_true: np.ndarray,
    U: np.ndarray,
    *,
    max_frames: Optional[int] = None,
    rng_seed: int = 0,
) -> List[np.ndarray]:
    """
    Collect frames Y = Phi_true[j,s] (D,2) for which U[j,s]=True.
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


# ============================================================
# Subspace PCA (equivariant baseline)
# ============================================================

def subspace_pca_fit(frames: Sequence[np.ndarray], *, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit O(2)-equivariant subspace PCA on frames in V(2,D).

    Mean projector M = avg(Y Y^T) is invariant under Y -> YQ for Q∈O(2).
    Take top-d eigenspace of M.

    Returns:
      B: (D,d) orthonormal basis
      evals: (d,) top eigenvalues
    """
    if len(frames) == 0:
        raise ValueError("Need at least one frame to fit a reducer.")
    Y0 = np.asarray(frames[0], dtype=float)
    if Y0.ndim != 2 or Y0.shape[1] != 2:
        raise ValueError("Each frame must have shape (D,2).")
    D = int(Y0.shape[0])
    if not (2 <= d <= D):
        raise ValueError(f"Need 2 <= d <= D. Got d={d}, D={D}.")

    M = np.zeros((D, D), dtype=float)
    for Y in frames:
        Y = np.asarray(Y, dtype=float)
        M += Y @ Y.T
    M /= float(len(frames))

    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    B = vecs[:, :d]
    evals = vals[:d]
    return B, evals


def subspace_pca_transform_frame(B: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Y (D,2) -> (d,2): B^T Y, then polar re-orthonormalize."""
    B = np.asarray(B, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("Y must have shape (D,2).")
    if B.ndim != 2:
        raise ValueError("B must be (D,d).")
    if B.shape[0] != Y.shape[0]:
        raise ValueError("B and Y have incompatible ambient dimensions.")
    return _polar_orthonormalize_2cols(B.T @ Y)


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
      B: (D,d)
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    U = np.asarray(U, dtype=bool)

    n_sets, n_samples, D, two = Phi_true.shape
    if two != 2:
        raise ValueError("Phi_true last dim must be 2.")
    if not (2 <= d <= D):
        raise ValueError(f"Need 2 <= d <= D. Got d={d}, D={D}.")

    frames_fit = _collect_frames_as_list(Phi_true, U, max_frames=max_frames, rng_seed=rng_seed)
    if len(frames_fit) == 0:
        raise ValueError("No frames were available to reduce (U has no True entries?).")

    B, _ = subspace_pca_fit(frames_fit, d=int(d))

    Phi_red = np.zeros((n_sets, n_samples, d, 2), dtype=float)    
    recon_errs: List[float] = []

    for j in range(n_sets):
        for s in range(n_samples):
            if not U[j, s]:
                continue
            Y = Phi_true[j, s]
            Yproj = B @ (B.T @ Y)
            recon_errs.append(float(np.linalg.norm(Y - Yproj, ord="fro")))
            Phi_red[j, s] = subspace_pca_transform_frame(B, Y)

    errs = np.asarray(recon_errs, dtype=float)
    rep = FrameReductionReport(
        method="subspace_pca",
        D_in=int(D),
        d_out=int(d),
        mean_proj_err=float(np.mean(errs)),
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
    For each d in dims, fit B_d and compute mean ||Y - B_d B_d^T Y||_F^2
    over sampled frames.
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
        se = []
        for Y in frames_fit:
            Yproj = B @ (B.T @ Y)
            se.append(float(np.linalg.norm(Y - Yproj, ord="fro") ** 2))
        mean_sq_err[t] = float(np.mean(se))

    return dims_arr, mean_sq_err


# ============================================================
# PSC (optional dependency)
# ============================================================

def reduce_frames_psc(
    *,
    Phi_true: np.ndarray,
    U: np.ndarray,
    d: int,
    max_frames: Optional[int] = None,
    rng_seed: int = 0,
    verbosity: int = 0,
) -> Tuple[np.ndarray, FrameReductionReport, np.ndarray]:
    """
    Reduce frames using the PSC package (HarlinLee/PSC):

      - Collect frames Y ∈ R^{D×2} with orthonormal columns.
      - Stack into ys with shape (s, D, 2).
      - Fit alpha ∈ R^{D×d} (orthonormal cols) via PSC.PCA + manopt_alpha.
      - Transform each frame by Yd = alpha^T Y and polar re-orthonormalize.

    Returns:
      Phi_red: (n_sets,n_samples,d,2)
      report
      alpha: (D,d)
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    U = np.asarray(U, dtype=bool)

    n_sets, n_samples, D, two = Phi_true.shape
    if two != 2:
        raise ValueError("PSC reducer expects k=2 frames (last dim must be 2).")
    if not (2 <= d <= D):
        raise ValueError(f"Need 2 <= d <= D. Got d={d}, D={D}.")

    frames_fit = _collect_frames_as_list(Phi_true, U, max_frames=max_frames, rng_seed=rng_seed)
    if len(frames_fit) == 0:
        raise ValueError("No frames available to fit PSC reducer (U has no True entries?).")

    ys = np.stack(frames_fit, axis=0)  # (s, D, 2)

    try:
        from PSC.projections import PCA as PSC_PCA, manopt_alpha
    except Exception as e:
        raise ImportError(
            "PSC package not importable. Install it so that "
            "`from PSC.projections import PCA, manopt_alpha` works."
        ) from e

    alpha_init = PSC_PCA(ys, int(d))  # (D,d)
    alpha = manopt_alpha(ys, alpha_init, verbosity=int(verbosity))
    alpha = np.asarray(alpha, dtype=float)

    if alpha.shape != (D, d):
        raise ValueError(f"PSC returned alpha with shape {alpha.shape}, expected {(D,d)}.")

    Phi_red = np.zeros((n_sets, n_samples, d, 2), dtype=float)
    recon_errs: List[float] = []

    for j in range(n_sets):
        for s in range(n_samples):
            if not U[j, s]:
                continue
            Y = Phi_true[j, s]                      # (D,2)
            Yproj = alpha @ (alpha.T @ Y)           # (D,2)
            recon_errs.append(float(np.linalg.norm(Y - Yproj, ord="fro")))
            Phi_red[j, s] = _polar_orthonormalize_2cols(alpha.T @ Y)

    errs = np.asarray(recon_errs, dtype=float)
    rep = FrameReductionReport(
        method="psc",
        D_in=int(D),
        d_out=int(d),
        mean_proj_err=float(np.mean(errs)),
    )
    return Phi_red, rep, alpha


def reduction_curve_psc(
    *,
    Phi_true: np.ndarray,
    U: np.ndarray,
    dims: Sequence[int],
    max_frames: Optional[int] = None,
    rng_seed: int = 0,
    psc_verbosity: int = 0,
    use_manopt: bool = True,
    plot: bool = False,          # NEW
):
    """
    PSC reconstruction curve:
      For each d in dims:
        fit alpha_d via PSC,
        compute mean squared projection error:
            mean ||Y - alpha_d alpha_d^T Y||_F^2  over sampled frames.

    If plot=True, produces a simple matplotlib plot.
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    U = np.asarray(U, dtype=bool)

    frames_fit = _collect_frames_as_list(Phi_true, U, max_frames=max_frames, rng_seed=rng_seed)
    if len(frames_fit) == 0:
        raise ValueError("No frames available for PSC curve computation.")

    D = int(frames_fit[0].shape[0])
    dims_arr = np.asarray(list(dims), dtype=int)
    if np.any(dims_arr < 2) or np.any(dims_arr > D):
        raise ValueError(f"All dims must satisfy 2 <= d <= D={D}.")

    ys = np.stack(frames_fit, axis=0)  # (s, D, 2)

    from PSC.projections import PCA as PSC_PCA, manopt_alpha

    mean_sq_err = np.zeros_like(dims_arr, dtype=float)

    for t, d in enumerate(dims_arr):
        alpha = PSC_PCA(ys, int(d))
        if use_manopt:
            alpha = manopt_alpha(ys, alpha, verbosity=int(psc_verbosity))
        alpha = np.asarray(alpha, dtype=float)

        se = []
        for Y in frames_fit:
            Yproj = alpha @ (alpha.T @ Y)
            se.append(float(np.linalg.norm(Y - Yproj, ord="fro") ** 2))
        mean_sq_err[t] = float(np.mean(se))

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(dims_arr, mean_sq_err, marker="o")
        plt.xlabel("Target dimension d")
        plt.ylabel("Mean squared projection error")
        plt.title("PSC projection error curve")
        plt.show()

    return dims_arr, mean_sq_err