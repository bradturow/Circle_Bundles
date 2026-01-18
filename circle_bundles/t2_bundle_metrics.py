# circle_bundles/t2_bundle_metrics.py
from __future__ import annotations

"""
T^2 circle-bundle / O(2)-bundle quotient metrics (angle-coordinate models).

We model total-space points as angles:
    (theta1, theta2, phi) in R^3 representing  T^2 x S^1,
with the understanding that angles are taken mod 2pi when evaluating distances.

We define a natural product metric upstairs and then construct quotient metrics
by taking a min over group actions (finite groups like Z2, or lattice actions like Z^2).

Design goals
------------
- Keep *all* T^2 bundle-specific quotient logic out of metrics.py.
- Support:
    * oriented S^1-bundles over T^2 (Euler class k in Z)
    * O(2)-bundles with reflection monodromy along one base generator
- Use your existing "new-style" Metric interface: objects with .pairwise(X, Y).

Conventions / IMPORTANT IMPLEMENTATION NOTE
------------------------------------------
- Coordinates are (theta1, theta2, phi) = (base1, base2, fiber) in radians.
- Inputs may be any real angles.

- CRITICAL: For quotient metrics, the group action should act on a *lift* to R^3.
  Therefore, our generators/actions DO NOT wrap mod 2pi.

- Wrapping into [0,2pi) happens ONLY inside the base "upstairs" metric
  (T2xS1ProductAngleMetric) when it computes distances.

- For Z^2 quotients we approximate the true quotient metric by searching over a bounded
  exponent window [-W, W]^2 (good for experiments; increase W if needed).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Protocol

import numpy as np

TWOPI = 2.0 * np.pi


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

    IMPORTANT: This class is where wrapping occurs. Actions/generators should NOT wrap.
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
GenFn = Callable[[np.ndarray, int], np.ndarray]  # generator with integer exponent


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
class ZkQuotientMetric:
    """
    Quotient metric by a Z^k-action with k commuting generators.

    We approximate the true quotient metric by searching over a bounded window:
        m_i in [-window, window].

    Each generator is a function gen(Y, m) that applies exponent m directly (m can be negative).

    Practical note:
      - window=1 is usually enough for local bundle-map experiments,
        but if you see artifacts, bump to 2 or 3.
    """
    base_metric: Metric
    generators: Sequence[GenFn]  # length k
    window: int = 1
    name: str = "ZkQuotientMetric"

    def pairwise(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y0 = X if Y is None else np.asarray(Y, dtype=float)

        k = len(self.generators)
        if k == 0:
            raise ValueError("Need at least one generator for ZkQuotientMetric.")
        W = int(self.window)
        if W < 0:
            raise ValueError("window must be >= 0.")

        grids = np.meshgrid(*([np.arange(-W, W + 1)] * k), indexing="ij")
        exponents = np.stack([g.ravel() for g in grids], axis=1)  # (K, k)

        Dmin: Optional[np.ndarray] = None
        for mvec in exponents:
            Yg = Y0
            for gen, m in zip(self.generators, mvec):
                Yg = gen(Yg, int(m))
            Dg = self.base_metric.pairwise(X, Yg)
            Dmin = Dg if Dmin is None else np.minimum(Dmin, Dg)

        assert Dmin is not None
        return Dmin


# ---------------------------------------------------------------------
# T^2 x S^1 action primitives (LIFTED angles: NO WRAPPING HERE)
# ---------------------------------------------------------------------

def gen_base_shift(axis: int) -> GenFn:
    """
    Z-generator shifting theta_axis by 2pi*m (axis 0 or 1).

    NOTE: No wrapping. This acts on a lift in R^3.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1 for base coordinates.")

    def _gen(Y: np.ndarray, m: int) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        out = Y.copy()
        out[..., axis] = out[..., axis] + TWOPI * float(m)
        return out

    return _gen


def gen_fiber_shift() -> GenFn:
    """
    Z-generator shifting fiber angle phi by 2pi*m.

    NOTE: No wrapping. This acts on a lift in R^3.
    """
    def _gen(Y: np.ndarray, m: int) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        out = Y.copy()
        out[..., 2] = out[..., 2] + TWOPI * float(m)
        return out

    return _gen


def act_fiber_reflection(Y: np.ndarray) -> np.ndarray:
    """
    Fiber reflection: (theta1, theta2, phi) -> (theta1, theta2, -phi).

    NOTE: No wrapping. This acts on a lift in R^3.
    """
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    out[..., 2] = -out[..., 2]
    return out


def gen_twist_by_other_base(*, shift_axis: int, other_axis: int, k: int) -> GenFn:
    """
    Z-generator that:
      theta_shift_axis += 2pi*m
      phi += k*m*theta_other_axis

    This bilinear coupling is a standard way to realize Euler class k
    in a Z^2 quotient model for oriented circle bundles over T^2.

    IMPORTANT: No wrapping here; wrapping happens in the base metric when measuring distance.
    """
    if shift_axis not in (0, 1) or other_axis not in (0, 1) or shift_axis == other_axis:
        raise ValueError("shift_axis and other_axis must be distinct elements of {0,1}.")
    k = int(k)

    def _gen(Y: np.ndarray, m: int) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        out = Y.copy()
        mm = float(m)
        out[..., shift_axis] = out[..., shift_axis] + TWOPI * mm
        out[..., 2] = out[..., 2] + float(k) * mm * out[..., other_axis]
        return out

    return _gen


def gen_base_shift_with_fiber_reflection(axis: int) -> GenFn:
    """
    Z-generator:
      theta_axis += 2pi*m
      if m is odd, reflect fiber (phi -> -phi)

    Encodes O(2) monodromy (-1 on the fiber) along one base generator.

    NOTE: No wrapping here; wrapping happens in the base metric when measuring distance.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1 for base coordinates.")

    def _gen(Y: np.ndarray, m: int) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        out = Y.copy()
        out[..., axis] = out[..., axis] + TWOPI * float(m)
        if (int(m) % 2) != 0:
            out[..., 2] = -out[..., 2]
        return out

    return _gen


# ---------------------------------------------------------------------
# Nice factories for experiments
# ---------------------------------------------------------------------

def T2_circle_bundle_metric_oriented(
    k: int,
    *,
    fiber_weight: float = 1.0,
    window: int = 1,
    convention: str = "twist_on_theta1_shift",
) -> Metric:
    """
    Metric on the total space of an oriented S^1-bundle over T^2 with Euler class k,
    modeled as a Z^2-quotient of the upstairs product metric.

    Parameters
    ----------
    k:
        Intended Euler class (sign depends on your cocycle orientation conventions).
    fiber_weight:
        Scales fiber distance relative to base.
    window:
        Search range for (m,n) in [-window,window]^2 in the quotient metric.
    convention:
        - "twist_on_theta1_shift": generator along theta1 includes twist by theta2
        - "twist_on_theta2_shift": generator along theta2 includes twist by theta1
    """
    base_metric = T2xS1ProductAngleMetric(fiber_weight=float(fiber_weight))

    convention = str(convention).strip()
    if convention == "twist_on_theta1_shift":
        g_tw = gen_twist_by_other_base(shift_axis=0, other_axis=1, k=int(k))
        g_sh = gen_base_shift(axis=1)
        # Often nicer seam behavior if we apply the "other-axis" shift first:
        generators = [g_sh, g_tw]
    elif convention == "twist_on_theta2_shift":
        g_sh = gen_base_shift(axis=0)
        g_tw = gen_twist_by_other_base(shift_axis=1, other_axis=0, k=int(k))
        generators = [g_sh, g_tw]
    else:
        raise ValueError("Unknown convention. Use 'twist_on_theta1_shift' or 'twist_on_theta2_shift'.")

    return ZkQuotientMetric(
        base_metric=base_metric,
        generators=generators,
        window=int(window),
        name=f"T2_S1_oriented_euler_{int(k)}",
    )


def T2_circle_bundle_metric_nonorientable(
    *,
    reflect_along: int = 0,
    k: int = 0,
    fiber_weight: float = 1.0,
    window: int = 1,
) -> Metric:
    """
    Metric on an O(2)-bundle over T^2 with reflection monodromy along one base generator.

    reflect_along:
        Which base generator reflects the fiber when traversed oddly (0 for theta1 loop, 1 for theta2 loop).
    k:
        Optional additional twist coupling (useful for twisted Euler experiments).
        Set k=0 for the pure reflection-monodromy model.
    """
    reflect_along = int(reflect_along)
    if reflect_along not in (0, 1):
        raise ValueError("reflect_along must be 0 or 1.")
    other = 1 - reflect_along

    base_metric = T2xS1ProductAngleMetric(fiber_weight=float(fiber_weight))

    g_reflect = gen_base_shift_with_fiber_reflection(axis=reflect_along)
    g_other = gen_base_shift(axis=other)

    if int(k) != 0:
        twist = gen_twist_by_other_base(shift_axis=reflect_along, other_axis=other, k=int(k))

        def g1(Y: np.ndarray, m: int) -> np.ndarray:
            # both depend on m; act on the lifted coordinates (no wrap)
            return g_reflect(twist(Y, m), m)

    else:
        g1 = g_reflect

    gens = [None, None]  # type: ignore[assignment]
    gens[reflect_along] = g1
    gens[other] = g_other

    return ZkQuotientMetric(
        base_metric=base_metric,
        generators=[gens[0], gens[1]],  # type: ignore[arg-type]
        window=int(window),
        name=f"T2_O2_nonorientable(reflect={reflect_along},k={int(k)})",
    )


__all__ = [
    # upstairs metric
    "T2xS1ProductAngleMetric",
    # wrappers
    "FiniteQuotientMetric",
    "ZkQuotientMetric",
    # helpers / primitives
    "wrap_angles",
    "tN_flat_pairwise_angles",
    "act_fiber_reflection",
    "gen_base_shift",
    "gen_fiber_shift",
    "gen_twist_by_other_base",
    "gen_base_shift_with_fiber_reflection",
    # factories
    "T2_circle_bundle_metric_oriented",
    "T2_circle_bundle_metric_nonorientable",
]
