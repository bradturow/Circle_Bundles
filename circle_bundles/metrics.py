# circle_bundles/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Union

import numpy as np

try:
    from scipy.spatial.distance import cdist as _cdist  # optional
except Exception:  # pragma: no cover
    _cdist = None


# ============================================================
# Vectorized metric objects 
# ============================================================

class Metric(Protocol):
    """Vectorized metric interface: returns full distance matrices."""
    name: str

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        ...


@dataclass(frozen=True)
class EuclideanMetric:
    name: str = "euclidean"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


@dataclass(frozen=True)
class S1AngleMetric:
    """Angles in radians; distance on S^1."""
    name: str = "S1_angle"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        t1 = np.asarray(X).reshape(-1)[:, None]
        t2 = t1.T if Y is None else np.asarray(Y).reshape(-1)[None, :]
        d = np.abs(t2 - t1)
        return np.minimum(d, 2 * np.pi - d)


@dataclass(frozen=True)
class RP1AngleMetric:
    """Angles in radians; theta ~ theta+pi."""
    name: str = "RP1_angle"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        t1 = np.asarray(X).reshape(-1)[:, None]
        t2 = t1.T if Y is None else np.asarray(Y).reshape(-1)[None, :]
        d = np.abs(t2 - t1)
        return np.minimum(d, np.pi - d)


@dataclass(frozen=True)
class S1UnitVectorMetric:
    """Unit vectors in R^2; geodesic distance arccos(<p,q>)."""
    name: str = "S1_unitvec"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        dots = np.clip(X @ Y.T, -1.0, 1.0)
        return np.arccos(dots)


@dataclass(frozen=True)
class RP1UnitVectorMetric:
    """Unit vectors in R^2 with antipodal ID; min(ang, pi-ang)."""
    name: str = "RP1_unitvec"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        dots = np.clip(X @ Y.T, -1.0, 1.0)
        ang = np.arccos(dots)
        return np.minimum(ang, np.pi - ang)


@dataclass(frozen=True)
class RP2UnitVectorMetric:
    """Unit vectors in R^3 with antipodal ID; min(||p-q||, ||p+q||)."""
    name: str = "RP2_unitvec"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)

        if _cdist is None:
            raise ImportError("SciPy not available, but RP2UnitVectorMetric uses scipy.spatial.distance.cdist.")
        Dpos = _cdist(X, Y)
        Dneg = _cdist(X, -Y)
        return np.minimum(Dpos, Dneg)


@dataclass(frozen=True)
class T2FlatMetric:
    """Flat torus distance for coords in [0,2pi)^2."""
    name: str = "T2_flat"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        diff = np.abs(X[:, None, :] - Y[None, :, :])
        torus_diff = np.minimum(diff, 2 * np.pi - diff)
        return np.linalg.norm(torus_diff, axis=-1)


@dataclass(frozen=True)
class SciPyCdistMetric:
    """Fallback wrapper for a scalar metric(p,q) using scipy.spatial.distance.cdist."""
    metric: Callable
    name: str = "scipy_cdist"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if _cdist is None:
            raise ImportError("SciPy not available: cannot use cdist fallback for custom metrics.")
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return _cdist(X, Y, metric=self.metric)


def as_metric(metric: Union["Metric", Callable, None]) -> "Metric":
    """Convert either a Metric object or a callable(p,q) into a Metric object."""
    if metric is None:
        return EuclideanMetric()
    # duck-typing: Metric objects have .pairwise
    if hasattr(metric, "pairwise"):
        return metric  # type: ignore[return-value]
    return SciPyCdistMetric(metric=metric, name=getattr(metric, "__name__", "custom_metric"))
    
    
# ============================================================
# Quotient metrics 
# ============================================================
    

ActionFn = Callable[[np.ndarray], np.ndarray]


def _pairwise_euclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


@dataclass(frozen=True)
class Z2QuotientMetricR5:
    """
    Z2 quotient metric induced from ambient Euclidean distance in R^5.

    Points are in R^5, intended as (v, Re z, Im z) with v in R^3, z on S^1.
    The Z2 action is: y ~ g(y), where g is an involution.

    Quotient distance:
        d([x],[y]) = min(||x - y||, ||x - g(y)||)
    """
    action: ActionFn
    name: str = "Z2QuotientMetricR5"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 5:
            raise ValueError(f"X must be (n,5). Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 5:
            raise ValueError(f"Y must be (m,5). Got {Y0.shape}.")

        Y1 = np.asarray(self.action(Y0), dtype=float)
        if Y1.shape != Y0.shape:
            raise ValueError(f"action returned shape {Y1.shape}, expected {Y0.shape}.")

        D0 = _pairwise_euclidean(X, Y0)
        D1 = _pairwise_euclidean(X, Y1)
        return np.minimum(D0, D1)

    def dist(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=float).reshape(5)
        q = np.asarray(q, dtype=float).reshape(5)
        q2 = np.asarray(self.action(q), dtype=float).reshape(5)
        return float(min(np.linalg.norm(p - q), np.linalg.norm(p - q2)))


def act_base_only(Y: np.ndarray) -> np.ndarray:
    """
    (v, a, b) -> (-v, a, b)  i.e. (v,z) -> (-v, z)
    """
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    out[..., :3] *= -1.0
    return out


def act_pi_twist(Y: np.ndarray) -> np.ndarray:
    """
    (v, a, b) -> (-v, -a, -b)  i.e. (v,z) -> (-v, -z)
    """
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    out[..., :3] *= -1.0
    out[..., 3:5] *= -1.0
    return out


def act_reflection_twist(Y: np.ndarray) -> np.ndarray:
    """
    (v, a, b) -> (-v, a, -b)  i.e. (v,z) -> (-v, conj(z))
    """
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    out[..., :3] *= -1.0
    out[..., 4] *= -1.0
    return out


def RP2_TrivialMetric() -> Z2QuotientMetricR5:
    """Trivial circle bundle over RP^2: (v,z)~(-v,z)."""
    return Z2QuotientMetricR5(action=act_base_only, name="RP2xS1")

def RP2_TwistMetric() -> Z2QuotientMetricR5:
    """Orientable nontrivial (monodromy -1): (v,z)~(-v,-z)."""
    return Z2QuotientMetricR5(action=act_pi_twist, name="RP2_Twist")

def RP2_FlipMetric() -> Z2QuotientMetricR5:
    """Non-orientable (reflection on fiber): (v,z)~(-v,conj z)."""
    return Z2QuotientMetricR5(action=act_reflection_twist, name="RP2_Klein")


# ============================================================
# Z_p quotient metric on S^3 compatible with hopf_projection
# ============================================================

def _safe_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return X / nrm


def _quat_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Hamilton product, vectorized.
    A: (...,4), B: (...,4) -> (...,4)
    Convention: q = (a,b,c,d) with a scalar.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    aw, ax, ay, az = A[..., 0], A[..., 1], A[..., 2], A[..., 3]
    bw, bx, by, bz = B[..., 0], B[..., 1], B[..., 2], B[..., 3]

    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)


def _s3_geodesic_pairwise(X: np.ndarray, Y: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Geodesic distance on S^3 via arccos(<x,y>) in R^4.
    X: (n,4), Y: (m,4) -> (n,m)
    """
    X = _safe_normalize_rows(X, eps=eps)
    Y = _safe_normalize_rows(Y, eps=eps)
    dots = np.clip(X @ Y.T, -1.0, 1.0)
    return np.arccos(dots)


def _s3_chordal_pairwise(X: np.ndarray, Y: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Chordal distance on S^3 induced from ambient R^4 Euclidean distance.
    X: (n,4), Y: (m,4) -> (n,m)
    """
    X = _safe_normalize_rows(X, eps=eps)
    Y = _safe_normalize_rows(Y, eps=eps)
    return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


def _unit_quat_from_axis_angle(v: np.ndarray, theta: float, *, eps: float = 1e-12) -> np.ndarray:
    """
    g = cos(theta) + (v_hat) sin(theta), where v is a 3-vector.
    Returns (4,) quaternion.
    """
    v = np.asarray(v, dtype=float).reshape(3,)
    nv = max(np.linalg.norm(v), eps)
    vhat = v / nv
    return np.array([np.cos(theta), *(np.sin(theta) * vhat)], dtype=float)


@dataclass(frozen=True)
class ZpHopfQuotientMetricS3:
    """
    Quotient metric on S^3 / <g>, where g = exp(v * 2pi/p) lies in the Hopf fiber circle S^1_v.

    This is compatible with hopf_projection(q, v=v_axis) because right-multiplication by g
    preserves q v q^{-1} (since g commutes with v).

    Distance:
        d([x],[y]) = min_{m=0..p-1} d_S3(x, y * g^m)

    Inputs are unit quaternions in R^4 with convention (a,b,c,d).
    """
    p: int
    v_axis: np.ndarray = np.array([1.0, 0.0, 0.0])  # matches your hopf_projection default v=e1
    base: str = "geodesic"  # "geodesic" or "chordal"
    name: str = "ZpHopfQuotientMetricS3"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if int(self.p) <= 0:
            raise ValueError(f"p must be positive. Got {self.p}.")
        if self.base not in ("geodesic", "chordal"):
            raise ValueError(f"base must be 'geodesic' or 'chordal'. Got {self.base}.")

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError(f"X must be (n,4) quaternions. Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 4:
            raise ValueError(f"Y must be (m,4) quaternions. Got {Y0.shape}.")

        # Normalize (important if user passes slightly non-unit quats)
        Xn = _safe_normalize_rows(X, eps=self.eps)
        Yn = _safe_normalize_rows(Y0, eps=self.eps)

        # Choose base distance on S^3
        if self.base == "geodesic":
            dist_fn = _s3_geodesic_pairwise
        else:
            dist_fn = _s3_chordal_pairwise

        # Precompute powers g^m (as unit quaternions)
        # g = exp(v * 2pi/p)
        theta0 = 2.0 * np.pi / float(self.p)
        gs = np.stack(
            [_unit_quat_from_axis_angle(self.v_axis, m * theta0, eps=self.eps) for m in range(self.p)],
            axis=0,
        )  # (p,4)

        # For each m, compute distance to Y * g^m and take min.
        Dmin = None
        for m in range(self.p):
            Ym = _quat_mul(Yn, gs[m][None, :])  # right-multiply each row of Y by g^m
            Dm = dist_fn(Xn, Ym, eps=self.eps)
            Dmin = Dm if Dmin is None else np.minimum(Dmin, Dm)

        return Dmin

def _pick_u_perp_v(v_axis: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Choose a unit 3-vector u ⟂ v_axis in a deterministic way.
    Returns u_vec in R^3 (not a quaternion).
    """
    v = np.asarray(v_axis, dtype=float).reshape(3,)
    nv = max(np.linalg.norm(v), eps)
    v = v / nv

    # pick a helper vector not parallel to v
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, v)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])

    u = np.cross(v, a)
    nu = max(np.linalg.norm(u), eps)
    u = u / nu
    return u


def _right_mul_by_u(Q: np.ndarray, u_axis: np.ndarray) -> np.ndarray:
    """
    Apply tau(q)=q*u where u is the pure-imag unit quaternion (0,u_axis).
    Q: (n,4) -> (n,4)
    """
    u_axis = np.asarray(u_axis, dtype=float).reshape(3,)
    u_quat = np.array([0.0, u_axis[0], u_axis[1], u_axis[2]], dtype=float)
    return _quat_mul(Q, u_quat[None, :])


@dataclass(frozen=True)
class Z2LensAntipodalQuotientMetricS3:
    """
    Metric on the quotient of the lens space L(p,1)=S^3/<g> by the Z2-action tau(q)=q*u
    that covers antipodal on S^2 under pi_v(q)=q v q^{-1}.

    Requires p even so that tau^2 is trivial on L(p,1).
    Distance:
        d([x],[y]) = min_{e in {id,tau}} min_{m=0..p-1} d_S3(x, e(y) * g^m)
    """
    p: int
    v_axis: np.ndarray = np.array([1.0, 0.0, 0.0])
    base: str = "geodesic"  # "geodesic" or "chordal"
    name: str = "Z2LensAntipodalQuotientMetricS3"
    eps: float = 1e-12
    u_axis: Optional[np.ndarray] = None  # if None, we pick a canonical u ⟂ v

    def __post_init__(self) -> None:
        p = int(self.p)
        if p <= 0:
            raise ValueError(f"p must be positive. Got {self.p}.")
        if p % 2 != 0:
            raise ValueError(f"This Z2 quotient covers antipodal and is well-defined only for even p. Got p={p}.")
        if self.base not in ("geodesic", "chordal"):
            raise ValueError(f"base must be 'geodesic' or 'chordal'. Got {self.base}.")

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError(f"X must be (n,4) quaternions. Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 4:
            raise ValueError(f"Y must be (m,4) quaternions. Got {Y0.shape}.")

        Xn = _safe_normalize_rows(X, eps=self.eps)
        Yn = _safe_normalize_rows(Y0, eps=self.eps)

        # base S^3 distance
        dist_fn = _s3_geodesic_pairwise if self.base == "geodesic" else _s3_chordal_pairwise

        # generator g = exp(v * 2pi/p)
        theta0 = 2.0 * np.pi / float(self.p)
        gs = np.stack(
            [_unit_quat_from_axis_angle(self.v_axis, m * theta0, eps=self.eps) for m in range(self.p)],
            axis=0,
        )  # (p,4)

        # choose u ⟂ v so that u v u^{-1} = -v
        u_axis = _pick_u_perp_v(self.v_axis, eps=self.eps) if self.u_axis is None else np.asarray(self.u_axis, dtype=float).reshape(3,)
        u_axis = u_axis / max(np.linalg.norm(u_axis), self.eps)

        # Two representatives for Y-orbit: Y and tau(Y)=Y*u
        Ytau = _right_mul_by_u(Yn, u_axis)

        def min_over_zp(Yrep: np.ndarray) -> np.ndarray:
            Dmin = None
            for m in range(self.p):
                Ym = _quat_mul(Yrep, gs[m][None, :])  # right-multiply by g^m
                Dm = dist_fn(Xn, Ym, eps=self.eps)
                Dmin = Dm if Dmin is None else np.minimum(Dmin, Dm)
            return Dmin

        D0 = min_over_zp(Yn)
        D1 = min_over_zp(Ytau)
        return np.minimum(D0, D1)
    

    
def S3QuotientMetric(
    p: int,
    *,
    v_axis: np.ndarray = np.array([1.0, 0.0, 0.0]),
    antipodal: bool = False,
    base: str = "geodesic",
    u_axis: Optional[np.ndarray] = None,
    name: Optional[str] = None,
) -> "Metric":
    """
    Unified constructor for S^3 quotient metrics compatible with hopf_projection.

    Parameters
    ----------
    p : int
        Lens parameter. Uses the cyclic subgroup <exp(v * 2pi/p)> acting by right-multiplication.
        p=1 means "no Z_p quotient" (still returns a valid Metric).
    v_axis : (3,) array
        Axis v defining hopf_projection(q,v)=q v q^{-1} and the Hopf circle subgroup S^1_v.
    antipodal : bool
        If True, additionally quotient by the Z2-action tau(q)=q*u covering antipodal on S^2.
        This is well-defined only when p is even (because tau^2 = (-1) which must lie in <g>).
    base : {"geodesic","chordal"}
        Underlying S^3 distance used before quotienting.
    u_axis : optional (3,) array
        Controls which u ⟂ v is used for the Z2 element. If None, a canonical perpendicular is chosen.
    name : optional str
        Override metric.name. Otherwise a reasonable default is used.

    Returns
    -------
    Metric
        Either ZpHopfQuotientMetricS3 or Z2LensAntipodalQuotientMetricS3.
    """
    p = int(p)
    if p <= 0:
        raise ValueError(f"p must be positive. Got {p}.")
    if base not in ("geodesic", "chordal"):
        raise ValueError(f"base must be 'geodesic' or 'chordal'. Got {base}.")

    v_axis = np.asarray(v_axis, dtype=float).reshape(3,)

    if antipodal:
        if p % 2 != 0:
            raise ValueError(f"antipodal=True requires even p (so -1 ∈ <exp(v*2pi/p)>). Got p={p}.")
        return Z2LensAntipodalQuotientMetricS3(
            p=p,
            v_axis=v_axis,
            base=base,
            u_axis=u_axis,
            name=name or f"S3/(Z_{p}⋊Z2)_antipodal",
        )

    return ZpHopfQuotientMetricS3(
        p=p,
        v_axis=v_axis,
        base=base,
        name=name or f"S3/Z_{p}",
    )
    
    
# ============================================================
# Old existing scalar / vector metric functions 
# ============================================================

def S1_dist(theta1, theta2):
    d = np.abs(theta2 - theta1)
    return np.minimum(d, 2 * np.pi - d)

def RP1_dist(theta1, theta2):
    d = np.abs(theta2 - theta1)
    return np.minimum(d, np.pi - d)

def S1_dist2(p, q):
    return np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))

def RP1_dist2(p, q):
    ang = np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))
    return np.minimum(ang, np.pi - ang)

def Euc_met(p, q):
    return np.linalg.norm(p - q)

def RP2_dist(p, q):
    return min(np.linalg.norm(p - q), np.linalg.norm(p + q))

def T2_dist(p, q):
    diff = np.abs(p - q)
    torus_diff = np.minimum(diff, 2 * np.pi - diff)
    return np.linalg.norm(torus_diff)


# ============================================================
# Distance matrices helper (backwards compatible)
# ============================================================

def get_dist_mat(data1, data2=None, metric=Euc_met):
    """
    Backwards-compatible distance matrix helper.

    metric can be:
      - one of the old scalar functions (Euc_met, S1_dist, RP1_dist, ...)
      - a new Metric object with .pairwise
      - an arbitrary callable(p,q) (requires SciPy for fallback)
    """
    X = np.asarray(data1)
    Y = X if data2 is None else np.asarray(data2)

    # Map known old metrics to fast vectorized Metric objects
    if metric is Euc_met:
        M = EuclideanMetric()
        return M.pairwise(X, None if data2 is None else Y)

    if metric is S1_dist:
        M = S1AngleMetric()
        return M.pairwise(X, None if data2 is None else Y)

    if metric is RP1_dist:
        M = RP1AngleMetric()
        return M.pairwise(X, None if data2 is None else Y)

    if metric is S1_dist2:
        M = S1UnitVectorMetric()
        return M.pairwise(X, None if data2 is None else Y)

    if metric is RP1_dist2:
        M = RP1UnitVectorMetric()
        return M.pairwise(X, None if data2 is None else Y)

    if metric is RP2_dist:
        M = RP2UnitVectorMetric()
        return M.pairwise(X, None if data2 is None else Y)

    if metric is T2_dist:
        M = T2FlatMetric()
        return M.pairwise(X, None if data2 is None else Y)

    # Otherwise accept Metric object or callable(p,q)
    M = as_metric(metric)
    return M.pairwise(X, None if data2 is None else Y)
