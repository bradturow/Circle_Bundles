# circle_bundles/gauge_canon.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Literal, List

import numpy as np

Edge = Tuple[int, int]


@dataclass(frozen=True)
class GaugeCanonConfig:
    """
    Samplewise gauge canonicalization for O(2) cocycles induced by Stiefel frames.

    This is designed as a *pre-PSC* step:
      - input: Omega_true[(j,k)][s] = Phi_true[j,s].T @ Phi_true[k,s]
      - output: gauge[j,s] in SO(2)
      - frames update: Phi_star[j,s] = Phi_true[j,s] @ gauge[j,s].T

    Notes
    -----
    - This is currently a "flatten the rotational part" canonicalization:
        minimize Σ w_{jk}(s) * |(phi_j - phi_k) - theta_{jk}(s)|^2
      with one chart pinned per sample to remove the global-rotation ambiguity.
    - It preserves det-pattern automatically (we only solve for SO(2) gauge).
    """
    enabled: bool = False

    edge_weight: Literal["pou_product", "uniform"] = "pou_product"
    anchor: Literal["max_pou", "first_active"] = "max_pou"

    # numerical / safety
    det_threshold: float = 0.0  # det>=threshold treated as +1
    inactive_identity: bool = True  # set gauge=I on inactive entries


@dataclass(frozen=True)
class GaugeCanonReport:
    n_sets: int
    n_samples: int
    n_edges: int
    mean_abs_rot_angle_before: float
    mean_abs_rot_angle_after: float

    def to_text(self, *, decimals: int = 3) -> str:
        r = int(decimals)
        return (
            "\n"
            + "=" * 12
            + " Gauge Canonicalization "
            + "=" * 12
            + "\n\n"
            + f"Edges: {self.n_edges}\n"
            + f"Mean |rot angle| before: {self.mean_abs_rot_angle_before:.{r}f} rad\n"
            + f"Mean |rot angle| after:  {self.mean_abs_rot_angle_after:.{r}f} rad\n"
            + "\n"
            + "=" * 44
            + "\n"
        )


# ------------------------------------------------------------
# O(2) helpers (your r(±1) convention)
# ------------------------------------------------------------

def reflection_matrix(sign: int) -> np.ndarray:
    """r(+1)=I, r(-1)=reflection across x-axis."""
    if sign == +1:
        return np.eye(2)
    if sign == -1:
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    raise ValueError(f"sign must be ±1, got {sign}")


def det_sign(M: np.ndarray, *, threshold: float = 0.0) -> int:
    d = float(np.linalg.det(M))
    return +1 if d >= float(threshold) else -1


def so2_angle(R: np.ndarray) -> float:
    """Angle of an SO(2) matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def so2_from_angle(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def split_o2(O: np.ndarray, *, det_threshold: float = 0.0) -> Tuple[np.ndarray, int]:
    """
    Decompose O in O(2) as O = R @ r(omega), omega=det(O) in {±1}, R in SO(2).
    """
    omega = det_sign(O, threshold=det_threshold)
    r = reflection_matrix(omega)
    R = O @ r  # since r^{-1}=r
    # tiny numeric cleanup: enforce det(R)>0 if drifted
    if np.linalg.det(R) < 0:
        R = R @ reflection_matrix(-1)
    return R, omega


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def infer_edges_from_U(U: np.ndarray) -> Tuple[Edge, ...]:
    U = np.asarray(U, dtype=bool)
    n_sets, _ = U.shape
    edges: List[Edge] = []
    for j in range(n_sets):
        for k in range(j + 1, n_sets):
            if np.any(U[j] & U[k]):
                edges.append((j, k))
    return tuple(edges)


def _edge_weight(cfg: GaugeCanonConfig, pou: np.ndarray, j: int, k: int, s: int) -> float:
    if cfg.edge_weight == "uniform":
        return 1.0
    if cfg.edge_weight == "pou_product":
        return float(pou[j, s] * pou[k, s])
    raise ValueError(f"Unknown edge_weight={cfg.edge_weight!r}")


# ------------------------------------------------------------
# Core: samplewise gauge solve
# ------------------------------------------------------------

def compute_samplewise_gauge_from_o2_cocycle(
    Omega: Dict[Edge, np.ndarray],
    *,
    U: np.ndarray,
    pou: np.ndarray,
    edges: Optional[Iterable[Edge]] = None,
    cfg: Optional[GaugeCanonConfig] = None,
) -> Tuple[np.ndarray, Dict[Edge, np.ndarray], GaugeCanonReport]:
    """
    Compute samplewise SO(2) gauge g_j(s) that flattens the SO(2) part
    of an O(2) cocycle Omega_{jk}(s).

    Returns
    -------
    gauge:
        (n_sets,n_samples,2,2) SO(2) matrices (I on inactive entries if cfg.inactive_identity)
    Omega_star:
        dict edges -> (n_samples,2,2) O(2) cocycle after gauge normalization
    report:
        GaugeCanonReport with simple before/after rotation-angle stats
    """
    if cfg is None:
        cfg = GaugeCanonConfig(enabled=True)

    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    n_sets, n_samples = U.shape

    if edges is None:
        edges = infer_edges_from_U(U)
    edges = tuple((min(a, b), max(a, b)) for (a, b) in edges)

    # Precompute theta_{jk}(s) and omega_{jk}(s)
    n_edges = len(edges)
    theta = np.zeros((n_edges, n_samples), dtype=float)
    omega = np.ones((n_edges, n_samples), dtype=int)

    for e_idx, (j, k) in enumerate(edges):
        Om = Omega[(j, k)]  # (n_samples,2,2)
        if Om.shape != (n_samples, 2, 2):
            raise ValueError(f"Omega[{(j,k)}] expected (n_samples,2,2), got {Om.shape}")
        for s in range(n_samples):
            R, w = split_o2(Om[s], det_threshold=cfg.det_threshold)
            theta[e_idx, s] = so2_angle(R)
            omega[e_idx, s] = int(w)

    # Solve for phi_j(s) in least squares per sample, with one anchor pinned.
    phi = np.zeros((n_sets, n_samples), dtype=float)

    for s in range(n_samples):
        active = np.where(U[:, s])[0]
        if active.size <= 1:
            continue

        if cfg.anchor == "max_pou":
            anchor = int(active[np.argmax(pou[active, s])])
        elif cfg.anchor == "first_active":
            anchor = int(active[0])
        else:
            raise ValueError(f"Unknown anchor={cfg.anchor!r}")

        var_nodes = [int(j) for j in active if int(j) != anchor]
        idx = {j: t for t, j in enumerate(var_nodes)}
        n_var = len(var_nodes)

        rows = []
        rhs = []

        for e_idx, (j, k) in enumerate(edges):
            if not (U[j, s] and U[k, s]):
                continue
            w = _edge_weight(cfg, pou, j, k, s)
            if w <= 0.0:
                continue
            t = theta[e_idx, s]
            row = np.zeros((n_var,), dtype=float)
            if j != anchor:
                row[idx[j]] += 1.0
            if k != anchor:
                row[idx[k]] -= 1.0
            rows.append(np.sqrt(w) * row)
            rhs.append(np.sqrt(w) * t)

        if not rows:
            continue

        A = np.vstack(rows)
        b = np.asarray(rhs, dtype=float)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)

        for j in var_nodes:
            phi[j, s] = float(x[idx[j]])
        phi[anchor, s] = 0.0

    # Build gauge matrices
    gauge = np.zeros((n_sets, n_samples, 2, 2), dtype=float)
    for j in range(n_sets):
        for s in range(n_samples):
            if cfg.inactive_identity and (not U[j, s]):
                gauge[j, s] = np.eye(2)
            else:
                gauge[j, s] = so2_from_angle(phi[j, s])

    # Build Omega_star
    Omega_star: Dict[Edge, np.ndarray] = {}
    for e_idx, (j, k) in enumerate(edges):
        Om_star = np.zeros((n_samples, 2, 2), dtype=float)
        for s in range(n_samples):
            # corrected rotation angle
            t_star = theta[e_idx, s] - (phi[j, s] - phi[k, s])
            R_star = so2_from_angle(t_star)
            Om_star[s] = R_star @ reflection_matrix(int(omega[e_idx, s]))
        Omega_star[(j, k)] = Om_star

    # Simple before/after metric
    abs_before = float(np.mean(np.abs(theta)))
    # after: compute abs angle of rotation part of Omega_star
    abs_after_list = []
    for (j, k) in edges:
        Om = Omega_star[(j, k)]
        for s in range(n_samples):
            R, _ = split_o2(Om[s], det_threshold=cfg.det_threshold)
            abs_after_list.append(abs(so2_angle(R)))
    abs_after = float(np.mean(abs_after_list)) if abs_after_list else 0.0

    report = GaugeCanonReport(
        n_sets=int(n_sets),
        n_samples=int(n_samples),
        n_edges=int(n_edges),
        mean_abs_rot_angle_before=abs_before,
        mean_abs_rot_angle_after=abs_after,
    )

    return gauge, Omega_star, report


def apply_gauge_to_frames(Phi_true: np.ndarray, gauge: np.ndarray, *, U: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fast update:
        Phi_star[j,s] = Phi_true[j,s] @ gauge[j,s].T

    Phi_true: (n_sets,n_samples,D,2)
    gauge:    (n_sets,n_samples,2,2)

    If U is provided, inactive entries are left unchanged.
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    gauge = np.asarray(gauge, dtype=float)

    if Phi_true.ndim != 4 or Phi_true.shape[-1] != 2:
        raise ValueError(f"Phi_true expected (n_sets,n_samples,D,2), got {Phi_true.shape}")
    if gauge.shape[:2] != Phi_true.shape[:2] or gauge.shape[-2:] != (2, 2):
        raise ValueError(f"gauge expected (n_sets,n_samples,2,2), got {gauge.shape}")

    gT = np.swapaxes(gauge, -1, -2)
    Phi_star = Phi_true @ gT

    if U is not None:
        U = np.asarray(U, dtype=bool)
        mask = ~U[:, :, None, None]
        Phi_star = np.where(mask, Phi_true, Phi_star)

    return Phi_star
