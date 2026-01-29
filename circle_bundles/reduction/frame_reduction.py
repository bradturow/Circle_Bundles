# circle_bundles/frame_reduction.py
"""
O(2)-equivariant frame reduction for Stiefel frames Y ∈ V(2, D).

This module provides optional dimensionality reduction for collections of 2-frames
(`(D,2)` matrices with orthonormal columns), used in the bundle pipeline when the
ambient feature dimension D is large.

Design goals
------------
- Keep bundle/gluing logic out of this file (this is a preprocessing utility).
- Preserve the right O(2)-action: Y ↦ YQ for Q ∈ O(2) (i.e., equivariant reduction).
- Provide two reducers:
    * subspace_pca : always available, fast, deterministic baseline
    * psc          : optional dependency (HarlinLee/PSC), higher-quality but heavier

Conventions
-----------
- Frames are represented as real arrays of shape (D,2).
- A "projection" Π(Y) refers to projecting Y into a learned d-dimensional subspace:
      Π(Y) = B B^T Y   or   Π(Y) = α α^T Y
  with B/α having orthonormal columns in R^D.
- After projecting, we re-orthonormalize columns via the polar factor to return to V(2,d).

Where it is used
----------------
This is intended to be called either:
- "pre_classifying": reduce raw Stiefel frames before building the classifying map
  (paper-faithful), or
- "post_stiefel": reduce after a Stiefel projection step (legacy / experimental).

Optional dependency
-------------------
The PSC reducer imports PSC lazily inside PSC functions so that PSC is not required
for the rest of the package.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Literal

import numpy as np

from ..utils.status_utils import _status, _status_clear  # shared status helpers

from ..trivializations.gauge_canon import GaugeCanonConfig, compute_samplewise_gauge_from_o2_cocycle, apply_gauge_to_frames, GaugeCanonReport  # noqa


ReduceMethod = Literal["none", "subspace_pca", "psc"]


ReduceStage = Literal["pre_classifying", "post_stiefel"]


__all__ = [
    "FrameReducerConfig",
    "FrameReductionReport",
]


def canonicalize_frames_before_reduction(
    *,
    Phi_true: np.ndarray,
    Omega_true: dict,
    U: np.ndarray,
    pou: np.ndarray,
    edges: Optional[Sequence[Tuple[int,int]]] = None,
    cfg: Optional[GaugeCanonConfig] = None,
) -> Tuple[np.ndarray, GaugeCanonReport, np.ndarray, dict]:
    """
    Convenience: compute gauge from Omega_true, apply it to Phi_true.

    Returns:
      Phi_star: (n_sets,n_samples,D,2)
      report: GaugeCanonReport
      gauge: (n_sets,n_samples,2,2)
      Omega_star: dict edges -> (n_samples,2,2)
    """
    if cfg is None:
        cfg = GaugeCanonConfig(enabled=True)

    gauge, Omega_star, rep = compute_samplewise_gauge_from_o2_cocycle(
        Omega_true, U=U, pou=pou, edges=edges, cfg=cfg
    )
    Phi_star = apply_gauge_to_frames(Phi_true, gauge, U=U)
    return Phi_star, rep, gauge, Omega_star


# ============================================================
# Reports / config
# ============================================================

@dataclass
class FrameReducerConfig:
@dataclass
class FrameReducerConfig:
    """
    Configuration for O(2)-equivariant dimensionality reduction of Stiefel frames.

    Attributes
    ----------
    method : ReduceMethod
        - "none"        : no reduction
        - "subspace_pca": always-available, O(2)-equivariant baseline
        - "psc"         : PSC-based reducer (optional dependency)
    stage : ReduceStage
        - "pre_classifying": reduce the raw Stiefel point cloud before building the classifying map
        - "post_stiefel"   : reduce after a Stiefel projection step (legacy / experimental)
    d : int
        Target ambient dimension after reduction (must satisfy 2 <= d <= D at runtime).
        A value of 0 is treated as "unset" by callers (i.e., no reduction unless set).
    max_frames : Optional[int]
        Optional subsampling cap for fitting the reducer (speed control).
    rng_seed : int
        RNG seed used for subsampling.
    psc_verbosity : int
        Verbosity forwarded to PSC's optimization routine (if method="psc").
    """
    method: ReduceMethod = "none"
    stage: ReduceStage = "pre_classifying"
    d: int = 0
    max_frames: Optional[int] = None
    rng_seed: int = 0
    psc_verbosity: int = 0


@dataclass
class FrameReductionReport:    
    """
    Summary statistics for a fitted frame reduction map Π : V(2,D) -> V(2,d).

    Attributes
    ----------
    method : ReduceMethod
        Reduction method used.
    D_in : int
        Original ambient dimension D.
    d_out : int
        Reduced ambient dimension d.
    sup_proj_err : float
        Supremum (over processed frames) of the ambient projection error:
            sup ||Y - Π(Y)||_F
    mean_proj_err : float
        Mean (over processed frames) of the ambient projection error:
            mean ||Y - Π(Y)||_F
    """
    method: ReduceMethod
    D_in: int
    d_out: int
    sup_proj_err: float   # sup ||Y - Π(Y)||_F   (ε_red)
    mean_proj_err: float  # mean ||Y - Π(Y)||_F  (\bar{ε}_red)

    def to_text(self, *, decimals: int = 3) -> str:
        """
        Render a human-readable multi-line summary (for notebooks/logging).

        Parameters
        ----------
        decimals : int
            Number of decimal places used for numeric fields.

        Returns
        -------
        str
            A formatted text block describing the fitted reduction and its errors.
        """        
        r = int(decimals)
        return (
            "\n"
            + "=" * 12
            + " Frame Reduction "
            + "=" * 12
            + "\n\n"
            + f"Method: {self.method}\n"
            + f"Ambient dim: D={self.D_in} → d={self.d_out}\n"
            + f"Sup projection error (||Y - Π(Y)||_F): {self.sup_proj_err:.{r}f}\n"
            + f"Mean projection error (||Y - Π(Y)||_F): {self.mean_proj_err:.{r}f}\n"
            + "\n"
            + "=" * 40
            + "\n"
        )



# ============================================================
# Progress helpers
# ============================================================

def _maybe_status(progress: bool, msg: str) -> None:
    if progress:
        _status(msg)


def _maybe_clear(progress: bool) -> None:
    if progress:
        _status_clear()


def _auto_every(n: int, *, target_updates: int = 25, min_every: int = 200) -> int:
    """
    Choose an update frequency that doesn't spam clear_output in notebooks.
    """
    n = int(n)
    if n <= 0:
        return 1
    every = max(int(min_every), int(np.ceil(n / max(1, int(target_updates)))))
    return max(1, every)


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
    U: Optional[np.ndarray] = None,
    *,
    max_frames: Optional[int] = None,
    rng_seed: int = 0,
) -> List[np.ndarray]:
    """
    Collect frames as a Python list of (D,2) arrays.

    Accepts either:
      - Phi_true with shape (n_sets,n_samples,D,2) plus U (n_sets,n_samples), OR
      - Phi_true with shape (m,D,2) (packed dataset), in which case U is ignored.
    """
    Phi_true = np.asarray(Phi_true, dtype=float)

    # Case A: packed dataset (m,D,2)
    if Phi_true.ndim == 3:
        m, D, two = Phi_true.shape
        if two != 2:
            raise ValueError("Packed frames must have shape (m,D,2).")
        frames = [Phi_true[i] for i in range(m)]

    # Case B: grid of frames (n_sets,n_samples,D,2)
    elif Phi_true.ndim == 4:
        if U is None:
            raise ValueError("U must be provided when Phi_true has shape (n_sets,n_samples,D,2).")
        U = np.asarray(U, dtype=bool)

        n_sets, n_samples, D, two = Phi_true.shape
        if two != 2:
            raise ValueError("Expected frames in V(2,D): last dim must be 2.")
        if U.shape != (n_sets, n_samples):
            raise ValueError("U shape mismatch with Phi_true.")

        Js, Ss = np.where(U)
        frames = [Phi_true[int(j), int(s)] for (j, s) in zip(Js, Ss)]

    else:
        raise ValueError(f"Phi_true must have ndim 3 or 4. Got shape {Phi_true.shape}.")

    if max_frames is not None and len(frames) > int(max_frames):
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
        if Y.shape != (D, 2):
            raise ValueError("All frames must share the same ambient dimension D and have shape (D,2).")
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
    progress: bool = False,
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

    if Phi_true.ndim != 4:
        raise ValueError(f"Expected Phi_true with shape (n_sets,n_samples,D,2). Got {Phi_true.shape}.")

    n_sets, n_samples, D, two = Phi_true.shape
    if two != 2:
        raise ValueError("Phi_true last dim must be 2.")
    if U.shape != (n_sets, n_samples):
        raise ValueError("U shape mismatch with Phi_true.")
    if not (2 <= int(d) <= int(D)):
        raise ValueError(f"Need 2 <= d <= D. Got d={d}, D={D}.")

    _maybe_status(progress, f"[frame reduction | subspace_pca] collecting frames (max_frames={max_frames})...")
    frames_fit = _collect_frames_as_list(Phi_true, U, max_frames=max_frames, rng_seed=rng_seed)
    if len(frames_fit) == 0:
        _maybe_clear(progress)
        raise ValueError("No frames were available to reduce (U has no True entries?).")

    _maybe_status(progress, f"[frame reduction | subspace_pca] fitting basis B (D={D} → d={d}) on {len(frames_fit)} frames...")
    B, _ = subspace_pca_fit(frames_fit, d=int(d))

    Phi_red = np.zeros((n_sets, n_samples, int(d), 2), dtype=float)
    recon_errs: List[float] = []

    total = int(np.count_nonzero(U))
    every = _auto_every(total, target_updates=25, min_every=200)
    done = 0
    _maybe_status(progress, f"[frame reduction | subspace_pca] transforming frames: 0/{total}")

    for j in range(n_sets):
        for s in range(n_samples):
            if not U[j, s]:
                continue
            Y = Phi_true[j, s]
            Yproj = B @ (B.T @ Y)  # Π(Y) in ambient space
            recon_errs.append(float(np.linalg.norm(Y - Yproj, ord="fro")))
            Phi_red[j, s] = subspace_pca_transform_frame(B, Y)

            done += 1
            if progress and (done % every == 0 or done == total):
                _status(f"[frame reduction | subspace_pca] transforming frames: {done}/{total}")

    errs = np.asarray(recon_errs, dtype=float)
    sup_err = float(np.max(errs)) if errs.size else 0.0
    mean_err = float(np.mean(errs)) if errs.size else 0.0

    rep = FrameReductionReport(
        method="subspace_pca",
        D_in=int(D),
        d_out=int(d),
        sup_proj_err=sup_err,
        mean_proj_err=mean_err,
    )
    _maybe_clear(progress)
    return Phi_red, rep, B


def reduction_curve_subspace_pca(
    *,
    Phi_true: np.ndarray,
    U: np.ndarray,
    dims: Sequence[int],
    max_frames: Optional[int] = None,
    rng_seed: int = 0,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a subspace PCA reconstruction-error curve across multiple target dimensions.

    For each d in `dims`, this fits a subspace basis B_d (using the same sampled frames)
    and reports the mean squared reconstruction error:
        mean ||Y - B_d B_d^T Y||_F^2.

    Returns
    -------
    dims_arr : np.ndarray
        Target dimensions as an int array.
    mean_sq_err : np.ndarray
        Mean squared reconstruction errors corresponding to dims_arr.
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    U = np.asarray(U, dtype=bool)

    _maybe_status(progress, f"[frame reduction | subspace_pca] collecting frames for curve (max_frames={max_frames})...")
    frames_fit = _collect_frames_as_list(Phi_true, U, max_frames=max_frames, rng_seed=rng_seed)
    if len(frames_fit) == 0:
        _maybe_clear(progress)
        raise ValueError("No frames available for curve computation.")

    D = int(frames_fit[0].shape[0])
    dims_arr = np.asarray(list(dims), dtype=int)
    if np.any(dims_arr < 2) or np.any(dims_arr > D):
        _maybe_clear(progress)
        raise ValueError(f"All dims must satisfy 2 <= d <= D={D}.")

    mean_sq_err = np.zeros_like(dims_arr, dtype=float)

    every = _auto_every(len(dims_arr), target_updates=25, min_every=1)
    _maybe_status(progress, f"[frame reduction | subspace_pca] curve: 0/{len(dims_arr)}")

    for t, dd in enumerate(dims_arr):
        B, _ = subspace_pca_fit(frames_fit, d=int(dd))
        se = []
        for Y in frames_fit:
            Yproj = B @ (B.T @ Y)
            se.append(float(np.linalg.norm(Y - Yproj, ord="fro") ** 2))
        mean_sq_err[t] = float(np.mean(se)) if len(se) else 0.0

        if progress and ((t + 1) % every == 0 or (t + 1) == len(dims_arr)):
            _status(f"[frame reduction | subspace_pca] curve: {t+1}/{len(dims_arr)}")

    _maybe_clear(progress)
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
    progress: bool = False,
) -> Tuple[np.ndarray, FrameReductionReport, np.ndarray]:
    """
    Reduce frames using the PSC package (HarlinLee/PSC).

    Returns:
      Phi_red: (n_sets,n_samples,d,2)
      report
      alpha: (D,d)
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    U = np.asarray(U, dtype=bool)

    if Phi_true.ndim != 4:
        raise ValueError(f"Expected Phi_true with shape (n_sets,n_samples,D,2). Got {Phi_true.shape}.")

    n_sets, n_samples, D, two = Phi_true.shape
    if two != 2:
        raise ValueError("PSC reducer expects k=2 frames (last dim must be 2).")
    if U.shape != (n_sets, n_samples):
        raise ValueError("U shape mismatch with Phi_true.")
    if not (2 <= int(d) <= int(D)):
        raise ValueError(f"Need 2 <= d <= D. Got d={d}, D={D}.")

    _maybe_status(progress, f"[frame reduction | psc] collecting frames (max_frames={max_frames})...")
    frames_fit = _collect_frames_as_list(Phi_true, U, max_frames=max_frames, rng_seed=rng_seed)
    if len(frames_fit) == 0:
        _maybe_clear(progress)
        raise ValueError("No frames available to fit PSC reducer (U has no True entries?).")

    ys = np.stack(frames_fit, axis=0)  # (s, D, 2)

    try:
        from PSC.projections import PCA as PSC_PCA, manopt_alpha  # type: ignore
    except Exception as e:
        _maybe_clear(progress)
        raise ImportError(
            "PSC package not importable. Install it so that "
            "`from PSC.projections import PCA, manopt_alpha` works."
        ) from e

    _maybe_status(progress, f"[frame reduction | psc] PSC_PCA init (D={D} → d={d}) on {len(frames_fit)} frames...")
    alpha_init = PSC_PCA(ys, int(d))  # (D,d)

    _maybe_status(progress, f"[frame reduction | psc] manopt_alpha (verbosity={int(verbosity)})...")
    alpha = manopt_alpha(ys, alpha_init, verbosity=int(verbosity))
    alpha = np.asarray(alpha, dtype=float)

    if alpha.shape != (D, int(d)):
        _maybe_clear(progress)
        raise ValueError(f"PSC returned alpha with shape {alpha.shape}, expected {(D,int(d))}.")

    Phi_red = np.zeros((n_sets, n_samples, int(d), 2), dtype=float)
    recon_errs: List[float] = []

    total = int(np.count_nonzero(U))
    every = _auto_every(total, target_updates=25, min_every=200)
    done = 0
    _maybe_status(progress, f"[frame reduction | psc] transforming frames: 0/{total}")

    for j in range(n_sets):
        for s in range(n_samples):
            if not U[j, s]:
                continue
            Y = Phi_true[j, s]                      # (D,2)
            Yproj = alpha @ (alpha.T @ Y)           # Π(Y) in ambient space
            recon_errs.append(float(np.linalg.norm(Y - Yproj, ord="fro")))
            Phi_red[j, s] = _polar_orthonormalize_2cols(alpha.T @ Y)

            done += 1
            if progress and (done % every == 0 or done == total):
                _status(f"[frame reduction | psc] transforming frames: {done}/{total}")

    errs = np.asarray(recon_errs, dtype=float)
    sup_err = float(np.max(errs)) if errs.size else 0.0
    mean_err = float(np.mean(errs)) if errs.size else 0.0

    rep = FrameReductionReport(
        method="psc",
        D_in=int(D),
        d_out=int(d),
        sup_proj_err=sup_err,
        mean_proj_err=mean_err,
    )
    _maybe_clear(progress)
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
    plot: bool = False,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a PSC reconstruction-error curve across multiple target dimensions.

    For each d in `dims`, this fits α_d via PSC (optionally refined with manopt) and reports:
        mean ||Y - α_d α_d^T Y||_F^2.

    Parameters
    ----------
    use_manopt : bool
        If True, refine the PCA initializer with PSC's manifold optimization routine.
    plot : bool
        If True, display a simple matplotlib curve in the current environment.

    Returns
    -------
    dims_arr : np.ndarray
        Target dimensions as an int array.
    mean_sq_err : np.ndarray
        Mean squared reconstruction errors corresponding to dims_arr.

    Raises
    ------
    ImportError
        If PSC is not importable.
    """

    Phi_true = np.asarray(Phi_true, dtype=float)
    U = np.asarray(U, dtype=bool)

    _maybe_status(progress, f"[frame reduction | psc] collecting frames for curve (max_frames={max_frames})...")
    frames_fit = _collect_frames_as_list(Phi_true, U, max_frames=max_frames, rng_seed=rng_seed)
    if len(frames_fit) == 0:
        _maybe_clear(progress)
        raise ValueError("No frames available for PSC curve computation.")

    D = int(frames_fit[0].shape[0])
    dims_arr = np.asarray(list(dims), dtype=int)
    if np.any(dims_arr < 2) or np.any(dims_arr > D):
        _maybe_clear(progress)
        raise ValueError(f"All dims must satisfy 2 <= d <= D={D}.")

    ys = np.stack(frames_fit, axis=0)  # (s, D, 2)

    try:
        from PSC.projections import PCA as PSC_PCA, manopt_alpha  # type: ignore
    except Exception as e:
        _maybe_clear(progress)
        raise ImportError(
            "PSC package not importable. Install it so that "
            "`from PSC.projections import PCA, manopt_alpha` works."
        ) from e

    mean_sq_err = np.zeros_like(dims_arr, dtype=float)

    every = _auto_every(len(dims_arr), target_updates=25, min_every=1)
    _maybe_status(progress, f"[frame reduction | psc] curve: 0/{len(dims_arr)}")

    for t, dd in enumerate(dims_arr):
        alpha = PSC_PCA(ys, int(dd))
        if use_manopt:
            alpha = manopt_alpha(ys, alpha, verbosity=int(psc_verbosity))
        alpha = np.asarray(alpha, dtype=float)

        se = []
        for Y in frames_fit:
            Yproj = alpha @ (alpha.T @ Y)
            se.append(float(np.linalg.norm(Y - Yproj, ord="fro") ** 2))
        mean_sq_err[t] = float(np.mean(se)) if len(se) else 0.0

        if progress and ((t + 1) % every == 0 or (t + 1) == len(dims_arr)):
            _status(f"[frame reduction | psc] curve: {t+1}/{len(dims_arr)}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(dims_arr, mean_sq_err, marker="o")
        plt.xlabel("Target dimension d")
        plt.ylabel("Mean squared projection error")
        plt.title("PSC projection error curve")
        plt.show()

    _maybe_clear(progress)
    return dims_arr, mean_sq_err
