# circle_bundles/metrics.py
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial.distance import cdist as _cdist  # optional
except Exception:  # pragma: no cover
    _cdist = None


# ---------- scalar / vector metrics ----------

def S1_dist(theta1, theta2):
    """Distance on S^1 for angles in radians."""
    d = np.abs(theta2 - theta1)
    return np.minimum(d, 2 * np.pi - d)

def RP1_dist(theta1, theta2):
    """Distance on RP^1 for angles in radians (identify theta ~ theta+pi)."""
    d = np.abs(theta2 - theta1)
    return np.minimum(d, np.pi - d)

def S1_dist2(p, q):
    """Geodesic distance on unit circle embedded in R^2: arccos(<p,q>)."""
    return np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))

def RP1_dist2(p, q):
    """Geodesic distance on RP^1 using unit vectors in R^2 (p ~ -p)."""
    ang = np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))
    return np.minimum(ang, np.pi - ang)

def Euc_met(p, q):
    """Euclidean distance."""
    return np.linalg.norm(p - q)

def RP2_dist(p, q):
    """Distance on RP^2 via unit vectors in R^3 (p ~ -p)."""
    return min(np.linalg.norm(p - q), np.linalg.norm(p + q))

def T2_dist(p, q):
    """Flat torus distance for coordinates in [0, 2pi)^2."""
    diff = np.abs(p - q)
    torus_diff = np.minimum(diff, 2 * np.pi - diff)
    return np.linalg.norm(torus_diff)


# ---------- distance matrices ----------

def get_dist_mat(data1, data2=None, metric=Euc_met):
    """
    Vectorized distance matrix helper.

    Parameters
    ----------
    data1 : (n, d) or (n,) array
    data2 : (m, d) or (m,) array, optional
    metric : callable
        One of the metric functions above, or a SciPy-compatible metric.

    Returns
    -------
    D : (n, m) ndarray
    """
    X = np.asarray(data1)
    Y = X if data2 is None else np.asarray(data2)

    # Ensure 2D for vector metrics when appropriate
    # (Angle metrics expect 1D arrays; dot-product metrics expect 2D.)
    if metric in (Euc_met, RP2_dist, T2_dist, S1_dist2, RP1_dist2):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

    # ---- Fast paths ----
    if metric is Euc_met:
        return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)

    if metric is S1_dist:
        t1 = np.asarray(X).reshape(-1)[:, None]
        t2 = np.asarray(Y).reshape(-1)[None, :]
        d = np.abs(t2 - t1)
        return np.minimum(d, 2 * np.pi - d)

    if metric is RP1_dist:
        t1 = np.asarray(X).reshape(-1)[:, None]
        t2 = np.asarray(Y).reshape(-1)[None, :]
        d = np.abs(t2 - t1)
        return np.minimum(d, np.pi - d)

    if metric is S1_dist2:
        # X,Y are (n,2) unit vectors
        dots = np.clip(X @ Y.T, -1.0, 1.0)
        return np.arccos(dots)

    if metric is RP1_dist2:
        dots = np.clip(X @ Y.T, -1.0, 1.0)
        ang = np.arccos(dots)
        return np.minimum(ang, np.pi - ang)

    if metric is RP2_dist:
        if _cdist is None:
            raise ImportError("SciPy not available, but RP2_dist matrix uses scipy.spatial.distance.cdist.")
        Dpos = _cdist(X, Y)
        Dneg = _cdist(X, -Y)
        return np.minimum(Dpos, Dneg)

    # ---- Fallback: SciPy cdist ----
    if _cdist is None:
        raise ImportError("SciPy not available: cannot use fallback cdist for custom metrics.")
    return _cdist(X, Y, metric=metric)
