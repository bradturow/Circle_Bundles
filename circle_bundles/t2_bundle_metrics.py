# circle_bundles/t2_bundle_metrics.py
from __future__ import annotations

"""
T^2 circle-bundle / O(2)-bundle quotient metrics (angle-coordinate models).

We model total-space points as angles:
    (theta1, theta2, phi) in R^3 representing  T^2 x S^1,
with the understanding that angles are taken mod 2pi when evaluating distances.

We define a natural product metric upstairs and then construct quotient metrics
by taking a min over group actions (finite groups like Z2, or lattice actions like Z^2).

IMPORTANT IMPLEMENTATION NOTE
-----------------------------
- Coordinates are (theta1, theta2, phi) = (base1, base2, fiber) in radians.
- Inputs may be any real angles.

- For quotient metrics, actions are intended to act on a *lift* in R^3 (no wrapping).
  Wrapping into [0,2pi) happens ONLY inside the upstairs metric
  (T2xS1ProductAngleMetric) when it computes distances.

- For Z^2 quotients we approximate the true quotient metric by searching over a bounded
  window [-W, W]^2. Increase W if you see artifacts.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class R2xS1ProductLiftMetric:
    """
    Product metric upstairs on R^2 x S^1 in lifted angle coords (theta1, theta2, phi):
        d^2 = ||(theta1,theta2)-(theta1',theta2')||_2^2 + (fiber_weight * d_S1(phi,phi'))^2

    CRITICAL: base coords are NOT wrapped here. The quotient over Z^2 induces the torus.
    """
    fiber_weight: float = 1.0
    name: str = "R2xS1_lift_product"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must be (n,3). Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 3:
            raise ValueError(f"Y must be (m,3). Got {Y0.shape}.")

        # Euclidean on the lifted base coordinates
        Db = np.linalg.norm(X[:, None, :2] - Y0[None, :, :2], axis=-1)

        # Circular distance on the fiber
        Df = s1_pairwise_angles(X[:, 2], Y0[:, 2])

        w = float(self.fiber_weight)
        return np.sqrt(Db**2 + (w * Df) ** 2)


# ---------------------------------------------------------------------
# Metric protocol (matches circle_bundles.metrics.Metric)
# ---------------------------------------------------------------------

class Metric(Protocol):
    name: str
    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray: ...


# ---------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------

def wrap_angles(X: np.ndarray) -> np.ndarray:
    """Wrap any real angles into [0, 2pi)."""
    return np.mod(X, TWOPI)


def tN_flat_pairwise_angles(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Flat N-torus distance on [0,2pi)^N using coordinatewise circular distance + L2 norm.

    X: (n,N), Y: (m,N) -> (n,m)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    diff = np.abs(X[:, None, :] - Y[None, :, :])
    torus_diff = np.minimum(diff, TWOPI - diff)
    return np.linalg.norm(torus_diff, axis=-1)


def s1_pairwise_angles(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Circular distance on S^1 for 1D angle arrays: phi (n,), psi (m,) -> (n,m)."""
    phi = np.asarray(phi, dtype=float).reshape(-1, 1)
    psi = np.asarray(psi, dtype=float).reshape(1, -1)
    d = np.abs(phi - psi)
    return np.minimum(d, TWOPI - d)


# ---------------------------------------------------------------------
# Upstairs product metric on T^2 x S^1 in angle coords
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class T2xS1ProductAngleMetric:
    """
    Product metric on (theta1, theta2, phi) (angles mod 2pi):
        d^2 = d_T2^2 + (fiber_weight * d_S1)^2
    where d_T2 is L2 of wrapped diffs in coords (0,1) and d_S1 is wrapped diff in coord (2).

    IMPORTANT: This class is where wrapping occurs. Actions should NOT wrap.
    """
    fiber_weight: float = 1.0
    name: str = "T2xS1_product"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = wrap_angles(np.asarray(X, dtype=float))
        Y0 = X if Y is None else wrap_angles(np.asarray(Y, dtype=float))

        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must be (n,3) angles (theta1,theta2,phi). Got {X.shape}.")
        if Y0.ndim != 2 or Y0.shape[1] != 3:
            raise ValueError(f"Y must be (m,3) angles (theta1,theta2,phi). Got {Y0.shape}.")

        Db = tN_flat_pairwise_angles(X[:, :2], Y0[:, :2])  # base torus part
        Df = s1_pairwise_angles(X[:, 2], Y0[:, 2])         # fiber circle part
        w = float(self.fiber_weight)
        return np.sqrt(Db**2 + (w * Df) ** 2)


# ---------------------------------------------------------------------
# Quotient wrappers
# ---------------------------------------------------------------------

ActionFn = Callable[[np.ndarray], np.ndarray]
Z2ActionFn = Callable[[np.ndarray, int, int], np.ndarray]  # action by (m,n) in Z^2


@dataclass(frozen=True)
class FiniteQuotientMetric:
    """
    Quotient metric by a finite group action given explicitly by a list of actions.

        d([x],[y]) = min_{g in actions} d0(x, g(y))

    actions should include the identity action.
    """
    base_metric: Metric
    actions: Sequence[ActionFn]
    name: str = "FiniteQuotientMetric"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        if len(self.actions) == 0:
            raise ValueError("actions must be nonempty (include identity).")

        Dmin: Optional[np.ndarray] = None
        for act in self.actions:
            Yg = np.asarray(act(Y0), dtype=float)
            Dg = self.base_metric.pairwise(X, Yg)
            Dmin = Dg if Dmin is None else np.minimum(Dmin, Dg)

        assert Dmin is not None
        return Dmin


@dataclass(frozen=True)
class Z2QuotientMetric:
    """
    Quotient metric by a Z^2-action specified as a *single* action (m,n) ↦ act(Y,m,n).

    This avoids subtle order-dependence issues that can arise when you try to implement
    a cocycle-defined Z^2-action via sequential application of separate generators.

    We approximate the true quotient metric by searching over:
        (m,n) in [-window,window]^2.
    """
    base_metric: Metric
    action: Z2ActionFn
    window: int = 1
    name: str = "Z2QuotientMetric"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        W = int(self.window)
        if W < 0:
            raise ValueError("window must be >= 0.")

        ms = np.arange(-W, W + 1, dtype=int)
        ns = np.arange(-W, W + 1, dtype=int)

        Dmin: Optional[np.ndarray] = None
        for m in ms:
            for n in ns:
                Yg = np.asarray(self.action(Y0, int(m), int(n)), dtype=float)
                Dg = self.base_metric.pairwise(X, Yg)
                Dmin = Dg if Dmin is None else np.minimum(Dmin, Dg)

        assert Dmin is not None
        return Dmin


# ---------------------------------------------------------------------
# T^2 x S^1 action primitives (LIFTED angles: NO WRAPPING HERE)
# ---------------------------------------------------------------------

def act_fiber_reflection(Y: np.ndarray) -> np.ndarray:
    """
    Fiber reflection: (theta1, theta2, phi) -> (theta1, theta2, -phi).

    NOTE: No wrapping. This acts on a lift in R^3.
    """
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    out[..., 2] = -out[..., 2]
    return out


def act_Z2_trivial(Y: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Trivial Z^2 action on base lifts:
      theta1 += 2pi*m
      theta2 += 2pi*n
      phi unchanged

    NOTE: No wrapping here; wrapping happens in the upstairs metric.
    """
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    out[..., 0] = out[..., 0] + TWOPI * float(m)
    out[..., 1] = out[..., 1] + TWOPI * float(n)
    return out


def act_Z2_oriented_euler(
    Y: np.ndarray,
    m: int,
    n: int,
    *,
    k: int,
    convention: str = "twist_on_theta1_shift",
) -> np.ndarray:
    """
    A standard cocycle model for an oriented S^1-bundle over T^2 with Euler class k.

    This defines an action of Z^2 on R^2 x S^1 (implemented on lifted angles in R^3).
    Distances are computed after wrapping inside T2xS1ProductAngleMetric, so fiber shifts
    differing by 2pi*integer are metrically invisible (as they should be for S^1).

    Parameters
    ----------
    convention:
        - "twist_on_theta1_shift": phi += k*m*theta2
        - "twist_on_theta2_shift": phi += k*n*theta1
    """
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()

    mm = float(m)
    nn = float(n)
    kk = float(int(k))

    # IMPORTANT: use the *input* base coordinates (out currently equals Y)
    theta1 = out[..., 0]
    theta2 = out[..., 1]

    out[..., 0] = theta1 + TWOPI * mm
    out[..., 1] = theta2 + TWOPI * nn

    convention = str(convention).strip()
    if convention == "twist_on_theta1_shift":
        out[..., 2] = out[..., 2] + kk * mm * theta2
    elif convention == "twist_on_theta2_shift":
        out[..., 2] = out[..., 2] + kk * nn * theta1
    else:
        raise ValueError("Unknown convention. Use 'twist_on_theta1_shift' or 'twist_on_theta2_shift'.")

    return out


def act_Z2_nonorientable(
    Y: np.ndarray,
    m: int,
    n: int,
    *,
    reflect_along: int,
    k: int = 0,
) -> np.ndarray:
    """
    Z^2 action modeling an O(2)-bundle over T^2 with reflection monodromy along one base generator.

    reflect_along:
        0 means the theta1 loop reflects the fiber when traversed oddly (m odd)
        1 means the theta2 loop reflects the fiber when traversed oddly (n odd)

    k:
        optional additional coupling term (useful for experiments with twisted Euler behavior).
        For k=0 this is the "pure reflection monodromy" model.
    """
    reflect_along = int(reflect_along)
    if reflect_along not in (0, 1):
        raise ValueError("reflect_along must be 0 or 1.")
    other = 1 - reflect_along

    Y = np.asarray(Y, dtype=float)
    out = Y.copy()

    mm = float(m)
    nn = float(n)
    kk = float(int(k))

    theta1 = out[..., 0]
    theta2 = out[..., 1]

    out[..., 0] = theta1 + TWOPI * mm
    out[..., 1] = theta2 + TWOPI * nn

    # Reflection parity determined by the chosen base generator
    if reflect_along == 0:
        if (int(m) % 2) != 0:
            out[..., 2] = -out[..., 2]
    else:
        if (int(n) % 2) != 0:
            out[..., 2] = -out[..., 2]

    # Optional coupling (kept simple for experiments; metrically mod-2pi anyway)
    if int(k) != 0:
        if reflect_along == 0:
            out[..., 2] = out[..., 2] + kk * mm * theta2  # couple to theta2
        else:
            out[..., 2] = out[..., 2] + kk * nn * theta1  # couple to theta1

    return out


# ---------------------------------------------------------------------
# Nice factories for experiments
# ---------------------------------------------------------------------

def T2_circle_bundle_metric_oriented(
    k: int,
    *,
    fiber_weight: float = 1.0,
    window: int = 1,
    convention: str = "twist_on_theta1_shift",
):
    base_metric = R2xS1ProductLiftMetric(fiber_weight=float(fiber_weight))

    def _act(Y: np.ndarray, m: int, n: int) -> np.ndarray:
        return act_Z2_oriented_euler(Y, m, n, k=int(k), convention=str(convention))

    return Z2QuotientMetric(
        base_metric=base_metric,
        action=_act,
        window=int(window),
        name=f"T2_S1_oriented_euler_{int(k)}",
    )


def T2_circle_bundle_metric_nonorientable(
    *,
    reflect_along: int = 0,
    k: int = 0,
    fiber_weight: float = 1.0,
    window: int = 1,
):
    base_metric = R2xS1ProductLiftMetric(fiber_weight=float(fiber_weight))

    def _act(Y: np.ndarray, m: int, n: int) -> np.ndarray:
        return act_Z2_nonorientable(Y, m, n, reflect_along=int(reflect_along), k=int(k))

    return Z2QuotientMetric(
        base_metric=base_metric,
        action=_act,
        window=int(window),
        name=f"T2_O2_nonorientable(reflect={int(reflect_along)},k={int(k)})",
    )


__all__ = [
    # upstairs metric
    "T2xS1ProductAngleMetric",
    # wrappers
    "FiniteQuotientMetric",
    "Z2QuotientMetric",
    # helpers / primitives
    "wrap_angles",
    "tN_flat_pairwise_angles",
    "s1_pairwise_angles",
    "act_fiber_reflection",
    "act_Z2_trivial",
    "act_Z2_oriented_euler",
    "act_Z2_nonorientable",
    # factories
    "T2_circle_bundle_metric_oriented",
    "T2_circle_bundle_metric_nonorientable",
]
