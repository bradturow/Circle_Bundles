# circle_bundles/metrics.py
from __future__ import annotations

from dataclasses import dataclass, field
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
    """
    Vectorized metric interface.

    A ``Metric`` represents a distance function that can be evaluated *in batch*
    on two collections of points.

    Implementations must provide:
    - a human-readable ``name`` attribute, and
    - a :meth:`pairwise` method returning a full distance matrix.

    This interface is intentionally minimal so metrics can be passed seamlessly
    to covers, bundle construction, and visualization utilities.

    Notes
    -----
    - The library assumes (but does not enforce) that distances satisfy the
      metric axioms.
    - All metrics operate on NumPy arrays and return NumPy arrays.

    See Also
    --------
    as_metric :
        Utility for converting scalar distance functions into vectorized metrics.
    """
    name: str

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute pairwise distances between two point clouds.

        Parameters
        ----------
        X :
            Array of shape ``(n, d)`` or ``(n,)`` representing ``n`` points.
        Y :
            Optional array of shape ``(m, d)`` or ``(m,)``.
            If omitted, distances are computed between rows of ``X``.

        Returns
        -------
        D :
            Distance matrix of shape ``(n, m)``, where
            ``D[i, j] = d(X[i], Y[j])``.
        """
        ...


@dataclass(frozen=True)
class EuclideanMetric:
    """
    Standard Euclidean metric on :math:`\\mathbb{R}^d`.

    This metric computes ordinary Euclidean distances between vectors
    using the ℓ² norm.

    It is the default metric used throughout the library whenever no
    other metric is specified.

    Attributes
    ----------
    name :
        Display name for the metric (default ``"euclidean"``).
    """
    name: str = "euclidean"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Euclidean pairwise distances.

        Parameters
        ----------
        X :
            Array of shape ``(n, d)`` or ``(n,)`` representing ``n`` points.
        Y :
            Optional array of shape ``(m, d)`` or ``(m,)``.
            If omitted, uses ``X``.

        Returns
        -------
        D :
            Euclidean distance matrix of shape ``(n, m)``.
        """
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


@dataclass(frozen=True)
class S1AngleMetric:
    r"""
    Geodesic distance on the circle :math:`\mathbb{S}^1` using angles.

    Points are represented by angles (in radians). The distance between
    two angles is the shorter arc length between them on the circle:

    .. math::
        d(\theta_1, \theta_2)
        = \min\left(|\theta_2 - \theta_1|,
                    2\pi - |\theta_2 - \theta_1|\right).

    This metric is appropriate when the base space is a circle and data
    are naturally parameterized by angles.

    Attributes
    ----------
    name :
        Metric identifier (default ``"S1_angle"``).
    base_name :
        Short name of the base space for plots/UI.
    base_name_latex :
        LaTeX symbol used in summaries and tables.
    """
    name: str = "S1_angle"
    base_name: str = "S^1"
    base_name_latex: str = r"\mathbb{S}^1"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute circular geodesic distances between angle arrays.

        Parameters
        ----------
        X :
            Array of angles (any real values), shape ``(n,)``.
        Y :
            Optional array of angles, shape ``(m,)``.
            If omitted, uses ``X``.

        Returns
        -------
        D :
            Distance matrix of shape ``(n, m)``.
        """
        t1 = np.asarray(X).reshape(-1)[:, None]
        t2 = t1.T if Y is None else np.asarray(Y).reshape(-1)[None, :]
        d = np.abs(t2 - t1)
        return np.minimum(d, 2 * np.pi - d)


@dataclass(frozen=True)
class RP1AngleMetric:
    r"""
    Geodesic distance on the real projective line :math:`\mathbb{RP}^1`
    using angular coordinates.

    The space :math:`\mathbb{RP}^1` can be viewed as a circle with antipodal
    points identified. Angles are therefore taken modulo :math:`\pi`.

    The distance between two angles is

    .. math::
        d(\theta_1, \theta_2)
        = \min\left(|\Delta|, \pi - |\Delta|\right),
        \quad \Delta = (\theta_2 - \theta_1) \bmod \pi.

    This metric is commonly used when the base variable represents
    *unoriented directions*.

    Attributes
    ----------
    name :
        Metric identifier (default ``"RP1_angle"``).
    base_name :
        Short name of the base space for plots/UI.
    base_name_latex :
        LaTeX symbol used in summaries and tables.
    """
    name: str = "RP1_angle"
    base_name: str = "RP^1"
    base_name_latex: str = r"\mathbb{RP}^1"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute projective geodesic distances between angle arrays.

        Parameters
        ----------
        X :
            Array of angles (any real values), shape ``(n,)``.
        Y :
            Optional array of angles, shape ``(m,)``.
            If omitted, uses ``X``.

        Returns
        -------
        D :
            Distance matrix of shape ``(n, m)``.
        """
        t1 = np.mod(np.asarray(X, dtype=float).reshape(-1), np.pi)[:, None]
        t2 = t1.T if Y is None else np.mod(
            np.asarray(Y, dtype=float).reshape(-1), np.pi
        )[None, :]
        d = np.abs(t2 - t1)
        return np.minimum(d, np.pi - d)


@dataclass(frozen=True)
class S1UnitVectorMetric:
    r"""
    Geodesic distance on :math:`\mathbb{S}^1` using unit vectors in :math:`\mathbb{R}^2`.

    Points are represented as (approximately) unit vectors ``p, q ∈ R^2`` lying on the
    unit circle. The geodesic distance is the angle between the vectors:

    .. math::
        d(p, q) = \arccos(\langle p, q \rangle),

    where the dot product is clamped to ``[-1, 1]`` for numerical stability.

    Use this metric when your base points are stored as 2D unit vectors
    (e.g. ``(cos θ, sin θ)``) rather than angles.

    Attributes
    ----------
    name :
        Metric identifier (default ``"S1_unitvec"``).
    base_name :
        Short name of the base space for plots/UI.
    base_name_latex :
        LaTeX symbol used in summaries and tables.

    Notes
    -----
    - This metric assumes inputs are unit vectors. If your vectors are not normalized,
      you should normalize them before calling :meth:`pairwise`, or use a different metric.
    - Values are in radians in the range ``[0, π]``.
    """
    name: str = "S1_unitvec"
    base_name: str = "S^1"
    base_name_latex: str = r"\mathbb{S}^1"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute geodesic distances on :math:`\mathbb{S}^1` between unit-vector samples.

        Parameters
        ----------
        X :
            Array of shape ``(n, 2)`` containing unit vectors in :math:`\mathbb{R}^2`.
        Y :
            Optional array of shape ``(m, 2)``. If omitted, uses ``X``.

        Returns
        -------
        D :
            Distance matrix of shape ``(n, m)`` with entries in radians.
        """
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        dots = np.clip(X @ Y.T, -1.0, 1.0)
        return np.arccos(dots)


@dataclass(frozen=True)
class RP1UnitVectorMetric:
    r"""
    Geodesic distance on :math:`\mathbb{RP}^1` using unit vectors in :math:`\mathbb{R}^2`.

    Points are represented by unit vectors in :math:`\mathbb{R}^2`, but with the antipodal
    identification:

    .. math::
        p \sim -p.

    Therefore the distance between classes ``[p]`` and ``[q]`` is:

    .. math::
        d([p],[q]) = \arccos(|\langle p, q \rangle|).

    This is the correct metric when the base variable represents *unoriented directions*
    (i.e. an axis rather than an arrow).

    Attributes
    ----------
    name :
        Metric identifier (default ``"RP1_unitvec"``).
    base_name :
        Short name of the base space for plots/UI.
    base_name_latex :
        LaTeX symbol used in summaries and tables.

    Notes
    -----
    - This implementation is robust to small deviations from unit norm: it normalizes
      input rows internally.
    - Distances are in radians in the range ``[0, π/2]`` for true unit inputs
      (because of antipodal identification).
    """
    name: str = "RP1_unitvec"
    base_name: str = "RP^1"
    base_name_latex: str = r"\mathbb{RP}^1"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute projective geodesic distances between unit-vector samples.

        Parameters
        ----------
        X :
            Array of shape ``(n, 2)`` representing vectors in :math:`\mathbb{R}^2`.
            Rows should be (approximately) unit length.
        Y :
            Optional array of shape ``(m, 2)``. If omitted, uses ``X``.

        Returns
        -------
        D :
            Distance matrix of shape ``(n, m)`` with entries in radians.

        Raises
        ------
        ValueError
            If ``X`` or ``Y`` does not have shape ``(*, 2)`` after reshaping 1D inputs.
        """
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)

        # allow (n,) -> (n,1) but RP1UnitVectorMetric really expects (n,2)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if X.shape[1] != 2 or Y.shape[1] != 2:
            raise ValueError(f"RP1UnitVectorMetric expects (n,2) arrays. Got X={X.shape}, Y={Y.shape}.")

        # normalize rows (robust if user passes slightly non-unit vectors)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)

        dots = np.clip(Xn @ Yn.T, -1.0, 1.0)
        return np.arccos(np.abs(dots))

    
    
@dataclass(frozen=True)
class RP2UnitVectorMetric:
    r"""
    Metric on :math:`\mathbb{RP}^2` using antipodal unit vectors in :math:`\mathbb{R}^3`.

    Points in :math:`\mathbb{RP}^2` can be represented by unit vectors
    ``p ∈ S^2 ⊂ R^3`` with the antipodal identification ``p ~ -p``.

    This implementation uses the *chordal* quotient distance induced from
    Euclidean distance in :math:`\mathbb{R}^3`:

    .. math::
        d([p],[q]) = \min(\|p - q\|,\; \|p + q\|).

    This is a common practical choice for embedding-based computations and
    is fast to compute in batch.

    Attributes
    ----------
    name :
        Metric identifier (default ``"RP2_unitvec"``).
    base_name :
        Short name of the base space for plots/UI.
    base_name_latex :
        LaTeX symbol used in summaries and tables.

    Notes
    -----
    - This is a *chordal* metric, not the intrinsic geodesic metric on :math:`\mathbb{RP}^2`.
      For most cover-building and neighborhood computations, chordal behavior is appropriate.
    - If you need the intrinsic projective geodesic distance, you’d implement a different class
      (e.g. based on ``arccos(|<p,q>|)``).
    """
    name: str = "RP2_unitvec"
    base_name: str = "RP^2"
    base_name_latex: str = r"\mathbb{RP}^2"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute chordal quotient distances on :math:`\mathbb{RP}^2`.

        Parameters
        ----------
        X :
            Array of shape ``(n, 3)`` representing vectors in :math:`\mathbb{R}^3`.
            Rows are ideally unit vectors on :math:`S^2`.
        Y :
            Optional array of shape ``(m, 3)``. If omitted, uses ``X``.

        Returns
        -------
        D :
            Distance matrix of shape ``(n, m)``.
        """
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        Dpos = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
        Dneg = np.linalg.norm(X[:, None, :] + Y[None, :, :], axis=-1)
        return np.minimum(Dpos, Dneg)


def _t2_flat_pairwise_angles(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    r"""
    Pairwise flat torus distance on :math:`\mathbb{T}^2 = (\mathbb{R}/2\pi\mathbb{Z})^2`
    using angle coordinates.

    This helper computes the standard *flat* (product) metric on the 2-torus by
    applying circular distance in each angular coordinate and then taking the
    Euclidean norm:

    .. math::
        d((\theta_1,\theta_2),(\phi_1,\phi_2))
        = \sqrt{ d_{S^1}(\theta_1,\phi_1)^2 + d_{S^1}(\theta_2,\phi_2)^2 },

    where
    :math:`d_{S^1}(a,b) = \min(|a-b|, 2\pi - |a-b|)`.

    Parameters
    ----------
    X :
        Array of shape ``(n, 2)`` containing angles in radians, interpreted modulo
        ``2π``.
    Y :
        Array of shape ``(m, 2)`` containing angles in radians.

    Returns
    -------
    D :
        Distance matrix of shape ``(n, m)``.

    Notes
    -----
    - This is the *flat* torus metric, not a quotient metric.
    - Used internally by several quotient constructions (Klein bottle,
      diagonal Z₂ quotients).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    diff = np.abs(X[:, None, :] - Y[None, :, :])
    torus_diff = np.minimum(diff, 2.0 * np.pi - diff)
    return np.linalg.norm(torus_diff, axis=-1)


@dataclass(frozen=True)
class T2FlatMetric:
    r"""
    Flat metric on the 2-torus :math:`\mathbb{T}^2`.

    Points are represented as angle pairs ``(θ₁, θ₂)`` in radians, interpreted
    modulo ``2π`` in each coordinate. The distance is computed using the product
    of circular distances in each factor:

    .. math::
        d(x,y) = \sqrt{ d_{S^1}(x_1,y_1)^2 + d_{S^1}(x_2,y_2)^2 }.

    This metric is appropriate when:
    - your base space is a genuine torus (no quotient identifications), and
    - coordinates are stored explicitly as angles.

    Attributes
    ----------
    name :
        Metric identifier (default ``"T2_flat"``).
    base_name :
        Short name of the base space for plots/UI.
    base_name_latex :
        LaTeX symbol used in summaries and tables.

    Notes
    -----
    - Input angles may lie outside ``[0, 2π)``; wrapping is handled implicitly.
    - This metric is frequently used as the *upstairs* metric before taking
      Z₂ quotients (e.g. Klein bottle, diagonal quotients).
    """
    name: str = "T2_flat"
    base_name: str = "T^2"
    base_name_latex: str = r"\mathbb{T}^2"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute flat torus distances between angle-coordinate samples.

        Parameters
        ----------
        X :
            Array of shape ``(n, 2)`` containing angles in radians.
        Y :
            Optional array of shape ``(m, 2)``. If omitted, uses ``X``.

        Returns
        -------
        D :
            Distance matrix of shape ``(n, m)``.

        Raises
        ------
        ValueError
            If ``X`` or ``Y`` does not have shape ``(*, 2)``.
        """
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X must be (n,2) angles. Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 2:
            raise ValueError(f"Y must be (m,2) angles. Got {Y0.shape}.")
        return _t2_flat_pairwise_angles(X, Y0)


# ============================================================
# Product metrics on concatenated (base | fiber) vectors
# ============================================================

@dataclass(frozen=True)
class ProductMetricConcat:
    """
    Product metric on concatenated vectors Z = [base | fiber].

    Distance:
        d(z,z')^2 = (base_weight * dB(base, base'))^2 + (fiber_weight * ||fiber-fiber'||)^2

    Notes
    -----
    - Uses the *cover/base metric* dB for the base block (can be torus, RP1, etc.).
    - Uses Euclidean distance for the fiber block.
    - Fully vectorized, returns an (n,m) matrix, satisfies Metric protocol.
    """
    base_metric: Metric
    base_dim: int
    base_weight: float = 1.0
    fiber_weight: float = 1.0
    name: str = "product_concat"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or Y0.ndim != 2:
            raise ValueError("ProductMetricConcat expects 2D arrays.")
        if X.shape[1] < self.base_dim or Y0.shape[1] < self.base_dim:
            raise ValueError(
                f"base_dim={self.base_dim} exceeds feature dim: X={X.shape}, Y={Y0.shape}"
            )

        XB, XF = X[:, : self.base_dim], X[:, self.base_dim :]
        YB, YF = Y0[:, : self.base_dim], Y0[:, self.base_dim :]

        DB = self.base_metric.pairwise(XB, YB)  # (n,m)
        DF = np.linalg.norm(XF[:, None, :] - YF[None, :, :], axis=-1)  # (n,m)

        bw = float(self.base_weight)
        fw = float(self.fiber_weight)
        return np.sqrt((bw * DB) ** 2 + (fw * DF) ** 2)
    
    

# ============================================================
# Flat Z2 quotient metrics on angle-coordinates (base-first!)
# ============================================================

@dataclass(frozen=True)
class KleinBottleFlatMetric:
    r"""
    Flat Klein bottle metric as a :math:`\mathbb{Z}_2` quotient of the flat torus.

    We represent points by angle coordinates ``(b, f)`` in radians (interpreted mod ``2π``),
    using the **base-first convention**:

    - ``b`` = base angle
    - ``f`` = fiber angle

    The Klein bottle arises as the quotient of :math:`\mathbb{T}^2` by the action

    .. math::
        g(b,f) = (b+\pi,\,-f).

    The induced quotient distance is

    .. math::
        d_{\mathrm{KB}}([x],[y]) = \min\{ d_{\mathbb{T}^2}(x,y),\ d_{\mathbb{T}^2}(x,g(y)) \},

    where :math:`d_{\mathbb{T}^2}` is the flat torus metric (coordinatewise circular
    distance, then Euclidean norm).

    Attributes
    ----------
    name :
        Metric identifier (default ``"KB_flat"``).

    Notes
    -----
    - This is an *intrinsic quotient metric* computed via minimizing over the two
      representatives ``y`` and ``g(y)``.
    - Inputs may be any real angles; wrapping into ``[0,2π)`` is handled internally.
    - This is the natural metric to use when your total-space coordinates already
      live in angle form (base,fiber) and you want the Klein identification.
    """
    name: str = "KB_flat"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute pairwise Klein bottle distances on angle-coordinate data.

        Parameters
        ----------
        X :
            Array of shape ``(n, 2)`` containing angles ``(b, f)`` in radians.
        Y :
            Optional array of shape ``(m, 2)``. If omitted, uses ``X``.

        Returns
        -------
        D :
            Distance matrix of shape ``(n, m)`` where ``D[i,j]`` is the quotient
            distance between ``X[i]`` and ``Y[j]``.

        Raises
        ------
        ValueError
            If ``X`` or ``Y`` is not a 2D array of shape ``(*, 2)``.
        """
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X must be (n,2) angles (base,fiber). Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 2:
            raise ValueError(f"Y must be (m,2) angles (base,fiber). Got {Y0.shape}.")

        # g(b,f) = (b + pi, -f) modulo 2pi
        Y1 = Y0.copy()
        Y1[:, 0] = np.mod(Y1[:, 0] + np.pi, 2.0 * np.pi)
        Y1[:, 1] = np.mod(-Y1[:, 1], 2.0 * np.pi)

        D0 = _t2_flat_pairwise_angles(X, Y0)
        D1 = _t2_flat_pairwise_angles(X, Y1)
        return np.minimum(D0, D1)


@dataclass(frozen=True)
class TorusDiagFlatMetric:
    r"""
    Flat metric on the :math:`\mathbb{Z}_2` quotient of the flat torus by a diagonal
    π-shift.

    Points are angle pairs ``(b, f)`` in radians (mod ``2π``), again using the
    **base-first convention**.

    The :math:`\mathbb{Z}_2` action is

    .. math::
        g(b,f) = (b+\pi,\ f+\pi).

    The induced quotient distance is

    .. math::
        d([x],[y]) = \min\{ d_{\mathbb{T}^2}(x,y),\ d_{\mathbb{T}^2}(x,g(y)) \}.

    Topologically, this quotient is homeomorphic to :math:`\mathbb{RP}^1 \times \mathbb{S}^1`
    (i.e. a trivial circle bundle over :math:`\mathbb{RP}^1`), but this class only
    encodes the metric structure via the quotient construction.

    Attributes
    ----------
    name :
        Metric identifier (default ``"T2_diag_flat"``).

    Notes
    -----
    - Symmetric in the two coordinates as a space, but we keep the (base,fiber)
      naming for consistency with the rest of the library.
    - Inputs may be any real angles; wrapping is handled internally.
    """
    name: str = "T2_diag_flat"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute pairwise distances for the diagonal π-shift quotient.

        Parameters
        ----------
        X :
            Array of shape ``(n, 2)`` containing angles ``(b, f)`` in radians.
        Y :
            Optional array of shape ``(m, 2)``. If omitted, uses ``X``.

        Returns
        -------
        D :
            Distance matrix of shape ``(n, m)`` giving quotient distances.

        Raises
        ------
        ValueError
            If ``X`` or ``Y`` is not a 2D array of shape ``(*, 2)``.
        """
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X must be (n,2) angles (base,fiber). Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 2:
            raise ValueError(f"Y must be (m,2) angles (base,fiber). Got {Y0.shape}.")

        # g(b,f) = (b + pi, f + pi) modulo 2pi
        Y1 = Y0.copy()
        Y1[:, 0] = np.mod(Y1[:, 0] + np.pi, 2.0 * np.pi)
        Y1[:, 1] = np.mod(Y1[:, 1] + np.pi, 2.0 * np.pi)

        D0 = _t2_flat_pairwise_angles(X, Y0)
        D1 = _t2_flat_pairwise_angles(X, Y1)
        return np.minimum(D0, D1)


def T2_Z2QuotientFlatMetric(kind: str = "klein") -> Metric:
    r"""
    Factory for :math:`\mathbb{Z}_2` quotient metrics on angle-coordinate data
    ``(base, fiber)`` in ``[0,2π)²``.

    This returns a metric object implementing the quotient distance induced by the
    flat torus metric upstairs.

    Parameters
    ----------
    kind :
        Which :math:`\mathbb{Z}_2` action to use:

        - ``"klein"`` (aliases: ``"kb"``, ``"klein_bottle"``):
          :math:`(b,f) \sim (b+\pi,\,-f)` giving the Klein bottle.
        - ``"diag"`` (aliases: ``"diagonal"``, ``"diag_pi"``, ``"diagonal_pi"``):
          :math:`(b,f) \sim (b+\pi,\ f+\pi)` giving the diagonal π-shift quotient.

    Returns
    -------
    metric :
        A metric object with a vectorized ``pairwise(X, Y=None)`` method, suitable for
        passing anywhere the library expects a :class:`Metric`.

    Raises
    ------
    ValueError
        If ``kind`` is not recognized.

    Examples
    --------
    >>> M = T2_Z2QuotientFlatMetric("klein")
    >>> D = M.pairwise(X)   # X has shape (n,2) of (base,fiber) angles
    """
    kind = str(kind).lower().strip()
    if kind in {"klein", "kb", "klein_bottle"}:
        return KleinBottleFlatMetric()
    if kind in {"diag", "diagonal", "diag_pi", "diagonal_pi"}:
        return TorusDiagFlatMetric()
    raise ValueError(f"Unknown kind={kind!r}. Expected 'klein' or 'diag'.")


# ============================================================
# Converting scalar metrics -> vectorized metrics
# ============================================================

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
    if hasattr(metric, "pairwise"):
        return metric  # type: ignore[return-value]
    return SciPyCdistMetric(metric=metric, name=getattr(metric, "__name__", "custom_metric"))


# ============================================================
# Euclidean Z2 quotient metrics on embedded data
# ============================================================

ActionFn = Callable[[np.ndarray], np.ndarray]


def _pairwise_euclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


@dataclass(frozen=True)
class Z2QuotientMetricEuclidean:
    """
    Z2 quotient metric induced from ambient Euclidean distance in R^d.

    d([x],[y]) = min(||x - y||, ||x - g(y)||)
    """
    action: ActionFn
    dim: int
    name: str = "Z2QuotientMetricEuclidean"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"X must be (n,{self.dim}). Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != self.dim:
            raise ValueError(f"Y must be (m,{self.dim}). Got {Y0.shape}.")

        Y1 = np.asarray(self.action(Y0), dtype=float)
        if Y1.shape != Y0.shape:
            raise ValueError(f"action returned shape {Y1.shape}, expected {Y0.shape}.")

        D0 = _pairwise_euclidean(X, Y0)
        D1 = _pairwise_euclidean(X, Y1)
        return np.minimum(D0, D1)

    def dist(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=float).reshape(self.dim)
        q = np.asarray(q, dtype=float).reshape(self.dim)
        q2 = np.asarray(self.action(q), dtype=float).reshape(self.dim)
        return float(min(np.linalg.norm(p - q), np.linalg.norm(p - q2)))


# ============================================================
# C^2 torus (R^4) Z2 quotient metrics with base-first convention
# ============================================================
# We assume the torus is embedded as (z1,z2) in C^2 with real coords:
#   z1 = x1 + i x2, z2 = x3 + i x4.
#
# To make "first coordinate is base projection" explicit, we allow:
#   base_in="z1"  or  base_in="z2"
# ============================================================

def act_klein_C2_torus_base_in_z1(Y: np.ndarray) -> np.ndarray:
    """
    base_in="z1": base angle lives in z1, fiber angle lives in z2.

    Klein action in angles (b,f): (b,f) ~ (b+pi, -f)
    becomes (z1,z2) -> (-z1, conj(z2)).

    Real coords: (x1,x2,x3,x4) -> (-x1,-x2, x3,-x4)
    """
    Y = np.asarray(Y, dtype=float)
    if Y.shape[-1] != 4:
        raise ValueError(f"Expected last dim 4 for C^2 data. Got {Y.shape}.")
    out = Y.copy()
    out[..., 0:2] *= -1.0
    out[..., 3] *= -1.0
    return out


def act_klein_C2_torus_base_in_z2(Y: np.ndarray) -> np.ndarray:
    """
    base_in="z2": base angle lives in z2, fiber angle lives in z1.

    Klein action in (b,f): (b,f) ~ (b+pi, -f)
    in (z1,z2) ordering means: (z_fiber, z_base) -> (conj(z_fiber), -z_base)
    i.e. (z1,z2) -> (conj(z1), -z2).

    Real coords: (x1,x2,x3,x4) -> (x1,-x2, -x3,-x4)
    """
    Y = np.asarray(Y, dtype=float)
    if Y.shape[-1] != 4:
        raise ValueError(f"Expected last dim 4 for C^2 data. Got {Y.shape}.")
    out = Y.copy()
    out[..., 1] *= -1.0       # conj(z1)
    out[..., 2:4] *= -1.0     # -z2
    return out


def act_diag_C2_torus_base_in_z1(Y: np.ndarray) -> np.ndarray:
    """
    Diagonal pi-shift: (b,f) ~ (b+pi, f+pi) corresponds to (z1,z2)->(-z1,-z2),
    independent of which factor is called base.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.shape[-1] != 4:
        raise ValueError(f"Expected last dim 4 for C^2 data. Got {Y.shape}.")
    return -Y


def act_diag_C2_torus_base_in_z2(Y: np.ndarray) -> np.ndarray:
    # Same map; kept for clarity/symmetry.
    return act_diag_C2_torus_base_in_z1(Y)


# Back-compat aliases (older code may import these names)
def act_klein_C2_torus(Y: np.ndarray) -> np.ndarray:
    return act_klein_C2_torus_base_in_z1(Y)


def act_diag_C2_torus(Y: np.ndarray) -> np.ndarray:
    return act_diag_C2_torus_base_in_z1(Y)


def Torus_KleinQuotientMetric_R4(*, base_in: str = "z2") -> "Metric":
    """
    Z2 quotient metric on R^4 C^2-torus embedding that implements the Klein identification
    (base,fiber) ~ (base+pi, -fiber) with an explicit base-factor choice.

    base_in:
      - "z1" (default): base angle is encoded in z1
      - "z2":            base angle is encoded in z2
    """
    base_in = str(base_in).lower().strip()
    if base_in == "z1":
        act = act_klein_C2_torus_base_in_z1
        nm = "T2_to_Klein_R4(base=z1)"
    elif base_in == "z2":
        act = act_klein_C2_torus_base_in_z2
        nm = "T2_to_Klein_R4(base=z2)"
    else:
        raise ValueError("base_in must be 'z1' or 'z2'.")
    return Z2QuotientMetricEuclidean(action=act, dim=4, name=nm)


def Torus_DiagQuotientMetric_R4(*, base_in: str = "z2") -> "Metric":
    """
    Z2 quotient metric on R^4 C^2-torus embedding for the diagonal pi-shift:
      (base,fiber) ~ (base+pi, fiber+pi)

    This is symmetric under swapping base/fiber, but we keep base_in for API consistency.
    """
    base_in = str(base_in).lower().strip()
    if base_in not in {"z1", "z2"}:
        raise ValueError("base_in must be 'z1' or 'z2'.")
    act = act_diag_C2_torus_base_in_z1
    nm = f"T2_to_Diag_R4(base={base_in})"
    return Z2QuotientMetricEuclidean(action=act, dim=4, name=nm)


def Torus_Z2QuotientMetric_R4(kind: str = "klein", *, base_in: str = "z2") -> "Metric":
    """
    Factory for Z2 quotient metrics on a C^2 torus embedded in R^4.

    kind:
      - "klein": (b,f)~(b+pi,-f)
      - "diag":  (b,f)~(b+pi,f+pi)

    base_in:
      - "z1" (default) or "z2" (for klein; diag is symmetric but accepted)
    """
    kind = str(kind).lower().strip()
    if kind in {"klein", "kb", "klein_bottle"}:
        return Torus_KleinQuotientMetric_R4(base_in=base_in)
    if kind in {"diag", "diagonal", "diag_pi", "diagonal_pi"}:
        return Torus_DiagQuotientMetric_R4(base_in=base_in)
    raise ValueError(f"Unknown kind={kind!r}. Expected 'klein' or 'diag'.")


# ============================================================
# Existing R^5 Z2 quotient (kept for compatibility)
# ============================================================

@dataclass(frozen=True)
class Z2QuotientMetricR5:
    """
    Z2 quotient metric induced from ambient Euclidean distance in R^5.

    Points are in R^5, intended as (v, Re z, Im z) with v in R^3, z on S^1.
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
    """(v, a, b) -> (-v, a, b)  i.e. (v,z) -> (-v, z)"""
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    out[..., :3] *= -1.0
    return out


def act_pi_twist(Y: np.ndarray) -> np.ndarray:
    """(v, a, b) -> (-v, -a, -b)  i.e. (v,z) -> (-v, -z)"""
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    out[..., :3] *= -1.0
    out[..., 3:5] *= -1.0
    return out


def act_reflection_twist(Y: np.ndarray) -> np.ndarray:
    """(v, a, b) -> (-v, a, -b)  i.e. (v,z) -> (-v, conj z)"""
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
    """Hamilton product, vectorized. q=(a,b,c,d) with a scalar."""
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
    X = _safe_normalize_rows(X, eps=eps)
    Y = _safe_normalize_rows(Y, eps=eps)
    dots = np.clip(X @ Y.T, -1.0, 1.0)
    return np.arccos(dots)


def _s3_chordal_pairwise(X: np.ndarray, Y: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    X = _safe_normalize_rows(X, eps=eps)
    Y = _safe_normalize_rows(Y, eps=eps)
    return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


def _unit_quat_from_axis_angle(v: np.ndarray, theta: float, *, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3,)
    nv = max(np.linalg.norm(v), eps)
    vhat = v / nv
    return np.array([np.cos(theta), *(np.sin(theta) * vhat)], dtype=float)


def _default_v_axis() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0], dtype=float)


@dataclass(frozen=True)
class ZpHopfQuotientMetricS3:
    """
    Quotient metric on S^3 / <g>, where g = exp(v * 2pi/p) lies in the Hopf fiber circle S^1_v.

    d([x],[y]) = min_{m=0..p-1} d_S3(x, y * g^m)
    """
    p: int
    v_axis: np.ndarray = field(default_factory=_default_v_axis)
    base: str = "geodesic"  # "geodesic" or "chordal"
    name: str = "ZpHopfQuotientMetricS3"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        p = int(self.p)
        if p <= 0:
            raise ValueError(f"p must be positive. Got {self.p}.")
        if self.base not in ("geodesic", "chordal"):
            raise ValueError(f"base must be 'geodesic' or 'chordal'. Got {self.base}.")

        v = np.asarray(self.v_axis, dtype=float).reshape(3,)
        nv = max(np.linalg.norm(v), self.eps)
        v = v / nv
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "v_axis", v)

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError(f"X must be (n,4) quaternions. Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 4:
            raise ValueError(f"Y must be (m,4) quaternions. Got {Y0.shape}.")

        Xn = _safe_normalize_rows(X, eps=self.eps)
        Yn = _safe_normalize_rows(Y0, eps=self.eps)

        dist_fn = _s3_geodesic_pairwise if self.base == "geodesic" else _s3_chordal_pairwise

        theta0 = 2.0 * np.pi / float(self.p)
        gs = np.stack(
            [_unit_quat_from_axis_angle(self.v_axis, m * theta0, eps=self.eps) for m in range(self.p)],
            axis=0,
        )

        Dmin = None
        for m in range(self.p):
            Ym = _quat_mul(Yn, gs[m][None, :])
            Dm = dist_fn(Xn, Ym, eps=self.eps)
            Dmin = Dm if Dmin is None else np.minimum(Dmin, Dm)
        return Dmin


def _pick_u_perp_v(v_axis: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v_axis, dtype=float).reshape(3,)
    nv = max(np.linalg.norm(v), eps)
    v = v / nv

    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, v)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])

    u = np.cross(v, a)
    nu = max(np.linalg.norm(u), eps)
    return u / nu


def _right_mul_by_u(Q: np.ndarray, u_axis: np.ndarray) -> np.ndarray:
    u_axis = np.asarray(u_axis, dtype=float).reshape(3,)
    u_quat = np.array([0.0, u_axis[0], u_axis[1], u_axis[2]], dtype=float)
    return _quat_mul(Q, u_quat[None, :])


@dataclass(frozen=True)
class Z2LensAntipodalQuotientMetricS3:
    """
    Metric on the quotient of the lens space L(p,1)=S^3/<g> by the Z2-action tau(q)=q*u
    that covers antipodal on S^2 under pi_v(q)=q v q^{-1}.

    Requires p even so that tau^2 is trivial on L(p,1).
    """
    p: int
    v_axis: np.ndarray = field(default_factory=_default_v_axis)
    base: str = "geodesic"  # "geodesic" or "chordal"
    name: str = "Z2LensAntipodalQuotientMetricS3"
    eps: float = 1e-12
    u_axis: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        p = int(self.p)
        if p <= 0:
            raise ValueError(f"p must be positive. Got {self.p}.")
        if p % 2 != 0:
            raise ValueError(f"antipodal quotient requires even p. Got p={p}.")
        if self.base not in ("geodesic", "chordal"):
            raise ValueError(f"base must be 'geodesic' or 'chordal'. Got {self.base}.")

        v = np.asarray(self.v_axis, dtype=float).reshape(3,)
        v = v / max(np.linalg.norm(v), self.eps)

        ua = None
        if self.u_axis is not None:
            ua = np.asarray(self.u_axis, dtype=float).reshape(3,)
            ua = ua / max(np.linalg.norm(ua), self.eps)

        object.__setattr__(self, "p", p)
        object.__setattr__(self, "v_axis", v)
        object.__setattr__(self, "u_axis", ua)

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError(f"X must be (n,4) quaternions. Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 4:
            raise ValueError(f"Y must be (m,4) quaternions. Got {Y0.shape}.")

        Xn = _safe_normalize_rows(X, eps=self.eps)
        Yn = _safe_normalize_rows(Y0, eps=self.eps)

        dist_fn = _s3_geodesic_pairwise if self.base == "geodesic" else _s3_chordal_pairwise

        theta0 = 2.0 * np.pi / float(self.p)
        gs = np.stack(
            [_unit_quat_from_axis_angle(self.v_axis, m * theta0, eps=self.eps) for m in range(self.p)],
            axis=0,
        )

        u_axis = _pick_u_perp_v(self.v_axis, eps=self.eps) if self.u_axis is None else self.u_axis
        assert u_axis is not None
        u_axis = u_axis / max(np.linalg.norm(u_axis), self.eps)

        Ytau = _right_mul_by_u(Yn, u_axis)

        def min_over_zp(Yrep: np.ndarray) -> np.ndarray:
            Dmin = None
            for m in range(self.p):
                Ym = _quat_mul(Yrep, gs[m][None, :])
                Dm = dist_fn(Xn, Ym, eps=self.eps)
                Dmin = Dm if Dmin is None else np.minimum(Dmin, Dm)
            return Dmin

        return np.minimum(min_over_zp(Yn), min_over_zp(Ytau))


def S3QuotientMetric(
    p: int,
    *,
    v_axis: np.ndarray = np.array([1.0, 0.0, 0.0]),
    antipodal: bool = False,
    base: str = "geodesic",
    u_axis: Optional[np.ndarray] = None,
    name: Optional[str] = None,
) -> "Metric":
    p = int(p)
    if p <= 0:
        raise ValueError(f"p must be positive. Got {p}.")
    if base not in ("geodesic", "chordal"):
        raise ValueError(f"base must be 'geodesic' or 'chordal'. Got {base}.")

    v_axis = np.asarray(v_axis, dtype=float).reshape(3,)

    if antipodal:
        if p % 2 != 0:
            raise ValueError(f"antipodal=True requires even p. Got p={p}.")
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
# Old scalar metric functions (kept for compatibility)
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


def KB_flat_dist(p, q):
    """
    Scalar Klein bottle flat distance on angle coords p=(base,fiber), q=(base,fiber),
    using (b,f)~(b+pi,-f).
    """
    p = np.asarray(p, dtype=float).reshape(2,)
    q = np.asarray(q, dtype=float).reshape(2,)

    def _t2(pv, qv):
        diff = np.abs(pv - qv)
        diff = np.minimum(diff, 2.0 * np.pi - diff)
        return float(np.linalg.norm(diff))

    qg = np.array([(q[0] + np.pi) % (2.0 * np.pi), (-q[1]) % (2.0 * np.pi)], dtype=float)
    return min(_t2(p, q), _t2(p, qg))


def T2_diag_flat_dist(p, q):
    """
    Scalar flat distance for the Z2 quotient on (base,fiber):
        (b,f) ~ (b+pi, f+pi)
    """
    p = np.asarray(p, dtype=float).reshape(2,)
    q = np.asarray(q, dtype=float).reshape(2,)

    def _t2(pv, qv):
        diff = np.abs(pv - qv)
        diff = np.minimum(diff, 2.0 * np.pi - diff)
        return float(np.linalg.norm(diff))

    qg = np.array([(q[0] + np.pi) % (2.0 * np.pi), (q[1] + np.pi) % (2.0 * np.pi)], dtype=float)
    return min(_t2(p, q), _t2(p, qg))


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

    if metric is KB_flat_dist:
        M = KleinBottleFlatMetric()
        return M.pairwise(X, None if data2 is None else Y)

    if metric is T2_diag_flat_dist:
        M = TorusDiagFlatMetric()
        return M.pairwise(X, None if data2 is None else Y)

    M = as_metric(metric)
    return M.pairwise(X, None if data2 is None else Y)
