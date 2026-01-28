# circle_bundles/bundle.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Literal,
    TypedDict,
    Protocol,
    runtime_checkable,
)

import numpy as np

from .nerve.combinatorics import Edge, Tri, canon_edge, canon_tri
from .trivializations.coordinatization import (
    GlobalTrivializationResult,
    MaxTrivialSubcomplex,
    compute_max_trivial_subcomplex as compute_max_trivial_subcomplex_core,
)
from .trivializations.bundle_map import FramePacking  # Literal["none","coloring","coloring2"] in your impl
from .utils.status_utils import _status, _status_clear

Simp = Tuple[int, ...]
SubcomplexMode = Literal["full", "cocycle", "max_trivial"]
Tet = Tuple[int, int, int, int]


def canon_simplex(sig: Iterable[int]) -> Simp:
    """Canonicalize a simplex key (sorted tuple of ints)."""
    return tuple(sorted(int(x) for x in sig))


# ============================================================
# Typed dicts / protocols (typing only; runtime is still normal dicts/objects)
# ============================================================

class BundleMeta(TypedDict, total=False):
    # core build settings (what you already set)
    ref_angle: float
    min_points_edge: int
    delta_min_points: int
    prime: int
    landmarks_per_patch: int
    min_patch_size: int
    cc_method: str
    pca_anchor: str
    prefer_edge_weight: str
    trivialization_method: str
    theta_units: str


class BundleMapMeta(TypedDict, total=False):
    strict_semicircle: bool
    semicircle_tol: float
    reducer: Optional[Dict[str, Any]]
    compute_chart_disagreement: bool
    packing: FramePacking   # or Literal["none","coloring","coloring2"]


class PullbackMeta(TypedDict, total=False):
    subcomplex: str
    strict_semicircle: bool
    semicircle_tol: float
    compute_chart_disagreement: bool
    packing: FramePacking


@runtime_checkable
class HasEdgeWeights(Protocol):
    """Optional protocol for objects that may carry per-edge errors/weights."""
    rms_angle_err: Dict[Edge, float]


@runtime_checkable
class HasWitnessErr(Protocol):
    witness_err: Dict[Edge, float]


# ----------------------------
# Result containers
# ----------------------------

@dataclass
class BundleMapResult:
    """
    Result of the bundle-map solver (global fiber coordinates from local trivializations).

    The bundle-map solver takes:
    - cover data (membership matrix ``U`` and partition of unity ``pou``),
    - local circle coordinates ``f`` on each chart, and
    - estimated transition functions ``Omega`` on overlaps,

    and produces a globally consistent set of fiber coordinates (and supporting diagnostics).

    Attributes
    ----------
    F:
        Final global bundle coordinates of shape ``(n_samples, D_used)`` where
        ``D_used`` is the ambient frame dimension used by the solver at the moment
        it outputs coordinates (after optional packing / projection / reduction).

        In the current implementation, ``F`` lives in the same ambient space as the
        projected frames used for gluing (i.e. columns correspond to the ambient
        coordinates of the Stiefel-projected frames).

    pre_F:
        Per-chart (pre-gluing) coordinates of shape ``(n_charts, n_samples, D_used)``.
        For each chart j and sample s with ``U[j,s]=True``, ``pre_F[j,s]`` stores the
        chartwise value ``F_j(s)``. Else it is zero.

        This is the main object used to compute chart disagreement diagnostics.

    Omega_used:
        The O(2)-valued cocycle actually used by the solver for diagnostics/gluing,
        stored as a dict ``(i,j) -> (n_samples,2,2)`` with canonical undirected edges.
        (This may differ from the raw simplicial cocycle on a restricted edge set,
        or after optional post-projection processing.)

    Phi_used:
        The Stiefel-projected frames actually used for gluing, of shape
        ``(n_charts, n_samples, D_used, 2)``.

        (Despite the name, this is *not* a ±1 orientation gauge vector; it is the
        frame field in the ambient space used by the solver.)

    report:
        A solver report object (implementation-defined) with diagnostic statistics and warnings.

    meta:
        Lightweight provenance (solver options such as semicircle constraints, packing mode, etc.).
    """
    F: np.ndarray
    pre_F: np.ndarray
    Omega_used: Dict[Edge, np.ndarray]
    Phi_used: np.ndarray
    report: Any
    meta: BundleMapMeta = field(default_factory=dict)


@dataclass
class PullbackTotalSpaceResult:
    """
    Pullback total space data ``Z = (base | fiber)`` equipped with a product metric.

    This object is returned by :meth:`BundleResult.get_pullback_data`. It packages:

    - ``total_data``: concatenation of base coordinates and computed fiber coordinates
    - ``metric``: a product metric compatible with that concatenation
    - ``bundle_map``: the underlying :class:`BundleMapResult` used to build the fiber block

    Attributes
    ----------
    total_data:
        Array of shape ``(n_samples, base_dim + fiber_dim)`` equal to
        ``np.concatenate([base, fiber], axis=1)``.
    metric:
        Product metric on concatenated vectors. Typically an instance of
        ``ProductMetricConcat`` (or anything satisfying your Metric protocol).
    bundle_map:
        The :class:`BundleMapResult` used to obtain the fiber coordinates.
    base_dim:
        Number of base columns in ``total_data``.
    fiber_dim:
        Number of fiber columns in ``total_data``.
    base_weight, fiber_weight:
        Weights used by the product metric.
    meta:
        Provenance and options used to construct this pullback object.
    """
    total_data: np.ndarray
    metric: Any
    bundle_map: BundleMapResult

    base_dim: int
    fiber_dim: int
    base_weight: float
    fiber_weight: float

    meta: PullbackMeta = field(default_factory=dict)

    def base_proj_map(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Project concatenated vectors onto their base block.

        Parameters
        ----------
        X:
            Optional array of concatenated vectors of shape ``(n, base_dim + fiber_dim)``.
            If omitted, projects ``self.total_data``.

        Returns
        -------
        base:
            The base block of shape ``(n, base_dim)``.
        """        
        A = self.total_data if X is None else np.asarray(X, dtype=float)
        return A[:, : self.base_dim]

    def to_text(self) -> str:
        """
        Return a short human-readable summary.

        Returns
        -------
        summary:
            A multi-line string describing array shapes and the metric name.
        """
        return (
            "=== Pullback Total Space ===\n"
            f"total_data shape: {tuple(self.total_data.shape)}\n"
            f"metric: {getattr(self.metric, 'name', type(self.metric).__name__)}"
        )


# ----------------------------
# BundleResult (main user object)
# ----------------------------

@dataclass
class BundleResult:
    r"""
    Output of :func:`circle_bundles.bundle.build_bundle`.

    A :class:`~circle_bundles.bundle.BundleResult` collects the user-facing artifacts of a
    discrete circle-bundle reconstruction:

    - a **cover** of the base space (charts, nerve, partition of unity),
    - a **local trivialization** (local circle coordinates on each chart),
    - an estimated **O(2) cocycle** (transition functions on overlaps),
    - **quality diagnostics** and **characteristic class representatives**,
    - plus cached convenience methods for downstream analysis and visualization.

    Typical workflow
    ----------------
    Build a bundle:

    >>> import circle_bundles as cb
    >>> bundle = cb.build_bundle(data, cover, show=True)

    Inspect already-computed outputs:

    >>> bundle.classes
    >>> bundle.quality

    Compute and reuse cached downstream results:

    >>> pers = bundle.get_persistence()
    >>> mt = bundle.get_max_trivial_subcomplex()
    >>> gt = bundle.get_global_trivialization()
    >>> bm = bundle.get_bundle_map()
    >>> pb = bundle.get_pullback_data()

    Notes on caching
    ---------------
    Most user-facing computations are exposed as ``get_*`` methods which cache their result
    in ``bundle._cache``. Passing ``recompute=True`` forces a fresh computation.
    Cached results may also be stored on a corresponding attribute (e.g. ``bundle.persistence``)
    when the call uses default settings.

    Attributes
    ----------
    cover:
        Cover object over the base space. Expected to provide:
        ``U`` (membership matrix), ``pou`` (partition of unity),
        landmark/base data (e.g. ``landmarks`` and ``base_points`` when applicable),
        and nerve accessors (``nerve_edges()``, ``nerve_triangles()``, ``nerve_tetrahedra()``).
    data:
        Original data array of shape ``(n_samples, ambient_dim)``.
    local_triv:
        Local trivialization result. Typically contains ``f`` (local angles per chart)
        and a ``valid`` mask over charts.
    cocycle:
        Estimated cocycle / transition data. Typically provides ``Omega`` and (for oriented
        variants) ``theta``.
    transitions:
        Transition-fit report and per-overlap diagnostics (often includes per-edge RMS errors).
    quality:
        Bundle quality diagnostics (consistency checks; optional witness errors when enabled).
    classes:
        Characteristic class representatives (e.g. SW1 and Euler-type representatives when available).
    meta:
        Lightweight provenance and default preferences used in construction.
    """
    cover: Any
    data: np.ndarray
    local_triv: Any
    cocycle: Any
    transitions: Any
    quality: Any
    classes: Any
    meta: BundleMeta
    total_metric: Any = None

    persistence: Any = None
    max_trivial: Any = None
    global_trivialization: Any = None
    bundle_map: Any = None

    _cache: Dict[Any, Any] = field(default_factory=dict, repr=False)

    # ============================================================
    # Persistence (edge-driven)
    # ============================================================

    def compute_persistence(
        self,
        *,
        prefer_edge_weight: Optional[str] = None,
        edge_weights: Optional[Dict[Edge, float]] = None,
        show: bool = False,
    ):
        """
        Compute an edge-driven persistence-style filtration for class representatives.

        This constructs a filtration of the nerve by assigning a scalar weight to each edge,
        then tracking when characteristic class representatives become coboundaries as edges
        are added in increasing-weight order.

        Parameters
        ----------
        prefer_edge_weight:
            Which built-in edge weight source to use when ``edge_weights`` is not provided.
            Common options are ``"rms"`` (transition RMS angle error) or ``"witness"`` (witness error),
            depending on what is available in this bundle.
            If omitted, defaults to ``bundle.meta["prefer_edge_weight"]`` (fallback: ``"rms"``).
        edge_weights:
            Explicit per-edge weights keyed by nerve edges ``(i, j)``.
            If provided, this overrides ``prefer_edge_weight``.
        show:
            If True, print a short summary of the resulting filtration / birth-death information.

        Returns
        -------
        persistence:
            A persistence result object (implementation-defined) containing at least:
            - an ordered edge list / filtration
            - class-specific birth/death information used downstream.
        """

        from .analysis.class_persistence import compute_bundle_persistence, summarize_edge_driven_persistence

        if prefer_edge_weight is None:
            prefer_edge_weight = self.meta.get("prefer_edge_weight", "rms")

        rms_angle_err = getattr(self.transitions, "rms_angle_err", None)
        witness_err = getattr(self.quality, "witness_err", None)

        pers = compute_bundle_persistence(
            cover=self.cover,
            classes=self.classes,
            edge_weights=edge_weights,
            rms_angle_err=rms_angle_err,
            witness_err=witness_err,
            prefer_edge_weight=prefer_edge_weight,
        )
        if show:
            summarize_edge_driven_persistence(pers, show=True)
        return pers

    def get_persistence(
        self,
        *,
        prefer_edge_weight: Optional[str] = None,
        edge_weights: Optional[Dict[Edge, float]] = None,
        recompute: bool = False,
        show: bool = False,
    ):
        """
        Cached wrapper for :meth:`compute_persistence`.

        Parameters
        ----------
        prefer_edge_weight, edge_weights:
            See :meth:`compute_persistence`.
        recompute:
            If True, ignore cached results and recompute.
        show:
            If True, print a summary (passed through to :meth:`compute_persistence`).

        Returns
        -------
        persistence:
            The cached or newly computed persistence object.
        """

        if prefer_edge_weight is None:
            prefer_edge_weight = self.meta.get("prefer_edge_weight", "rms")

        ew_key = None if edge_weights is None else id(edge_weights)
        key = ("persistence", str(prefer_edge_weight), ew_key)

        if recompute or key not in self._cache:
            self._cache[key] = self.compute_persistence(
                prefer_edge_weight=prefer_edge_weight,
                edge_weights=edge_weights,
                show=show,
            )

        pers = self._cache[key]

        if self.persistence is None and edge_weights is None:
            self.persistence = pers

        return pers

    # ============================================================
    # Max-trivial subcomplex
    # ============================================================

    def compute_max_trivial_subcomplex(
        self,
        *,
        persistence=None,
        prefer_edge_weight: Optional[str] = None,
        edge_weights: Optional[Dict[Edge, float]] = None,
    ) -> MaxTrivialSubcomplex:
        """
        Compute a maximal subcomplex on which selected class representatives become coboundaries.

        Informally: this finds a large subcomplex (in the edge-driven filtration sense)
        where the obstruction information has “died,” so that trivializations / lifts can
        be constructed reliably on that subcomplex.

        Parameters
        ----------
        persistence:
            A persistence object as returned by :meth:`get_persistence`.
            If omitted, it will be computed using ``prefer_edge_weight`` / ``edge_weights``.
        prefer_edge_weight, edge_weights:
            Used only when ``persistence`` is not provided. See :meth:`get_persistence`.

        Returns
        -------
        max_trivial:
            A :class:`~circle_bundles.trivializations.coordinatization.MaxTrivialSubcomplex`
            containing kept/removed simplices and filtration index information.

        Raises
        ------
        ValueError:
            If no such subcomplex is achieved (e.g. required codeath conditions never occur).
        """
        if persistence is None:
            persistence = self.get_persistence(
                prefer_edge_weight=prefer_edge_weight,
                edge_weights=edge_weights,
            )

        out = compute_max_trivial_subcomplex_core(
            persistence=persistence,
            edges=list(persistence.edges),
            triangles=list(getattr(persistence, "triangles", [])),
            tets=list(getattr(persistence, "tets", [])),
        )
        if out is None:
            raise ValueError("No max-trivial subcomplex: SW1/Euler codeath (or Euler cobirth) not achieved.")
        return out

    def get_max_trivial_subcomplex(
        self,
        *,
        prefer_edge_weight: Optional[str] = None,
        edge_weights: Optional[Dict[Edge, float]] = None,
        recompute: bool = False,
    ) -> MaxTrivialSubcomplex:
        """
        Cached wrapper for :meth:`compute_max_trivial_subcomplex`.

        Parameters
        ----------
        prefer_edge_weight, edge_weights:
            See :meth:`get_persistence`.
        recompute:
            If True, ignore cached results and recompute.

        Returns
        -------
        max_trivial:
            The cached or newly computed max-trivial subcomplex.
        """
        if prefer_edge_weight is None:
            prefer_edge_weight = self.meta.get("prefer_edge_weight", "rms")

        ew_key = None if edge_weights is None else id(edge_weights)
        key = ("max_trivial", str(prefer_edge_weight), ew_key)

        if recompute or key not in self._cache:
            self._cache[key] = self.compute_max_trivial_subcomplex(
                prefer_edge_weight=prefer_edge_weight,
                edge_weights=edge_weights,
            )

        mt = self._cache[key]

        if self.max_trivial is None and edge_weights is None:
            self.max_trivial = mt

        return mt

    # ============================================================
    # Global trivialization
    # ============================================================

    def compute_global_trivialization(
        self,
        *,
        method: Optional[str] = None,
        prefer_edge_weight: Optional[str] = None,
        edge_weights: Optional[Dict[Edge, float]] = None,
        orient: bool = True,
        require_orientable: bool = True,
    ) -> GlobalTrivializationResult:
        """
        Compute a global fiber coordinate on a max-trivial subcomplex.

        This uses a subset of edges (typically from :meth:`get_max_trivial_subcomplex`) and
        combines local trivializations into a single globally consistent fiber coordinate ``F``.

        Parameters
        ----------
        method:
            Global trivialization method identifier. If omitted, uses
            ``bundle.meta["trivialization_method"]`` (fallback: ``"singer"``).
        prefer_edge_weight, edge_weights:
            Used to determine the max-trivial subcomplex when needed.
            See :meth:`get_max_trivial_subcomplex`.
        orient:
            If True, attempt to orient the cocycle on the chosen edge set and apply the
            induced gauge to local coordinates before trivializing.
        require_orientable:
            If True and ``orient=True``, raise an error when the cocycle cannot be oriented
            on the chosen edge set.

        Returns
        -------
        global_triv:
            A :class:`~circle_bundles.trivializations.coordinatization.GlobalTrivializationResult`
            containing a global coordinate array ``F`` and metadata.

        Raises
        ------
        ValueError:
            If orientation is required but not possible on the chosen edge set.
        AttributeError:
            If required cocycle fields (e.g. ``cocycle.theta``) are missing.
        """
        from .trivializations.coordinatization import (
            build_global_trivialization,
            theta_dict_to_edge_vector_radians,
            apply_orientation_gauge_to_f,
        )

        if method is None:
            method = self.meta.get("trivialization_method", "singer")

        max_triv = self.get_max_trivial_subcomplex(
            prefer_edge_weight=prefer_edge_weight,
            edge_weights=edge_weights,
        )

        cocycle = getattr(self, "cocycle", None)
        if cocycle is None:
            raise AttributeError("self.cocycle is missing; cannot build global trivialization.")

        theta_dict = getattr(cocycle, "theta", None)
        if theta_dict is None:
            raise AttributeError("cocycle.theta is missing; cannot build global trivialization.")

        edges_used = list(max_triv.kept_edges)
        f_use = np.asarray(self.local_triv.f, dtype=float)

        phi_pm1 = None
        coc_use = cocycle

        if orient:
            ok, coc_oriented, phi = cocycle.orient_if_possible(
                edges_used,
                n_vertices=int(self.cover.U.shape[0]),
                require_all_edges_present=True,
            )
            if (not ok) and require_orientable:
                raise ValueError("Cocycle is not orientable on the chosen edge set; cannot trivialize.")
            if ok:
                coc_use = coc_oriented
                phi_pm1 = phi
                f_use = apply_orientation_gauge_to_f(
                    f=f_use,
                    phi_pm1=phi_pm1,
                    ref_angle=float(getattr(cocycle, "ref_angle", 0.0)),
                    U=self.cover.U,
                )

        theta_edge = theta_dict_to_edge_vector_radians(edges=edges_used, theta=coc_use.theta)

        F = build_global_trivialization(
            edges=edges_used,
            U=self.cover.U,
            pou=self.cover.pou,
            f=f_use,
            theta_edge=theta_edge,
            method=str(method),
            beta_edge=None,
        )

        n_flipped = int(np.sum(np.asarray(phi_pm1) == -1)) if phi_pm1 is not None else 0

        return GlobalTrivializationResult(
            method=str(method),
            edges_used=list(edges_used),
            F=np.asarray(F),
            meta={
                "k_removed": int(max_triv.k_removed),
                "n_edges_used": int(len(max_triv.kept_edges)),
                "n_triangles_used": int(len(getattr(max_triv, "kept_triangles", []))),
                "n_tetrahedra_used": int(len(getattr(max_triv, "kept_tetrahedra", []))),
                "theta_units": "radians",
                "oriented": bool(phi_pm1 is not None),
                "n_charts_reflected": n_flipped,
                "ref_angle": float(getattr(cocycle, "ref_angle", 0.0)),
                "convention": "A: Omega(j,k) maps k→j; after orienting, f_j ≈ f_k + theta_{jk} on overlaps",
            },
        )

    def get_global_trivialization(
        self,
        *,
        method: Optional[str] = None,
        prefer_edge_weight: Optional[str] = None,
        edge_weights: Optional[Dict[Edge, float]] = None,
        recompute: bool = False,
        orient: bool = True,
        require_orientable: bool = True,
    ) -> GlobalTrivializationResult:
        """
        Cached wrapper for :meth:`compute_global_trivialization`.

        Parameters
        ----------
        method:
            See :meth:`compute_global_trivialization`.
        prefer_edge_weight, edge_weights:
            See :meth:`get_max_trivial_subcomplex`.
        recompute:
            If True, ignore cached results and recompute.
        orient, require_orientable:
            See :meth:`compute_global_trivialization`.

        Returns
        -------
        global_triv:
            The cached or newly computed global trivialization result.
        """
        if method is None:
            method = self.meta.get("trivialization_method", "singer")
        if prefer_edge_weight is None:
            prefer_edge_weight = self.meta.get("prefer_edge_weight", "rms")

        ew_key = None if edge_weights is None else id(edge_weights)
        key = ("global_triv", str(method), str(prefer_edge_weight), ew_key, bool(orient), bool(require_orientable))

        if recompute or key not in self._cache:
            self._cache[key] = self.compute_global_trivialization(
                method=method,
                prefer_edge_weight=prefer_edge_weight,
                edge_weights=edge_weights,
                orient=orient,
                require_orientable=require_orientable,
            )

        gt = self._cache[key]

        if (
            self.global_trivialization is None
            and edge_weights is None
            and str(method) == str(self.meta.get("trivialization_method", "singer"))
            and bool(orient) is True
            and bool(require_orientable) is True
        ):
            self.global_trivialization = gt

        return gt

    # ============================================================
    # Frame dataset + bundle map + pullback total space
    # ============================================================

    def get_frame_dataset(
        self,
        *,
        stage: str = "post_projection",
        reducer=None,
        max_frames: Optional[int] = None,
        rng_seed: Optional[int] = None,
        edges: Optional[Iterable[Edge]] = None,
        subcomplex: SubcomplexMode = "full",
        persistence=None,
        packing: FramePacking = "coloring",
    ):
        """
        Build a frame dataset used by the bundle-map solver.

        This is a lower-level helper primarily used to debug or experiment with the solver.
        Most users should call :meth:`get_bundle_map` directly.

        Parameters
        ----------
        stage:
            Which internal stage of the frame pipeline to return (implementation-defined).
        reducer:
            Optional dimensionality reducer / frame selection configuration.
        max_frames:
            Optional cap on the number of frames generated.
        rng_seed:
            Random seed used when frame sampling is randomized.
        edges:
            Optional explicit edge list on which to build frames. If omitted, edges may be
            chosen based on ``subcomplex`` and ``persistence``.
        subcomplex:
            Which edge set to use when ``edges`` is not provided:
            ``"full"`` uses all nerve edges; other modes use edges derived from persistence.
        persistence:
            Persistence object required when ``subcomplex != "full"``.
        packing:
            Frame packing strategy (implementation-defined; affects which local frames are selected).

        Returns
        -------
        frames:
            A frame dataset object (implementation-defined) used as solver input.
        """
        from .trivializations.bundle_map import get_frame_dataset as _get_frame_dataset

        Omega = getattr(self.cocycle, "Omega", None)
        if Omega is None:
            raise AttributeError("bundle.cocycle.Omega is missing; cannot build frame dataset.")

        edges_used = edges

        if edges_used is None and subcomplex != "full":
            p = persistence if persistence is not None else getattr(self, "persistence", None)
            if p is None:
                raise ValueError(
                    "subcomplex != 'full' requires persistence. "
                    "Compute bundle.persistence first or pass persistence=..."
                )
            from .analysis.class_persistence import _edges_for_subcomplex_from_persistence
            edges_used = _edges_for_subcomplex_from_persistence(p, subcomplex)

        return _get_frame_dataset(
            U=self.cover.U,
            pou=self.cover.pou,
            Omega=Omega,
            edges=edges_used,
            reducer=reducer,
            stage=stage,
            max_frames=max_frames,
            rng_seed=rng_seed,
            packing=packing,
        )

    def compute_bundle_map(
        self,
        *,
        edges: Optional[Iterable[Edge]] = None,
        strict_semicircle: bool = True,
        semicircle_tol: float = 1e-8,
        reducer=None,
        show_summary: bool = True,
        compute_chart_disagreement: bool = True,
        packing: FramePacking = "coloring",
    ) -> BundleMapResult:
        from .trivializations.bundle_map import get_bundle_map

        Omega = getattr(self.cocycle, "Omega", None)
        if Omega is None:
            raise AttributeError("bundle.cocycle.Omega is missing; cannot run bundle_map.get_bundle_map().")

        F, pre_F, Omega_used, Phi_used, report = get_bundle_map(
            U=self.cover.U,
            pou=self.cover.pou,
            f=self.local_triv.f,
            Omega=Omega,
            edges=edges,
            strict_semicircle=bool(strict_semicircle),
            semicircle_tol=float(semicircle_tol),
            reducer=reducer,
            show_summary=bool(show_summary),
            compute_chart_disagreement=bool(compute_chart_disagreement),
            packing=packing,
        )

        # --- stable, curated reducer metadata (avoid dumping whole __dict__) ---
        reducer_meta = None
        if reducer is not None:
            reducer_meta = {
                "method": getattr(reducer, "method", None),
                "stage": getattr(reducer, "stage", None),
                "d": getattr(reducer, "d", None),
                "max_frames": getattr(reducer, "max_frames", None),
                "rng_seed": getattr(reducer, "rng_seed", None),
                "psc_verbosity": getattr(reducer, "psc_verbosity", None),
            }

        return BundleMapResult(
            F=np.asarray(F),
            pre_F=np.asarray(pre_F),
            Omega_used=Omega_used,
            Phi_used=np.asarray(Phi_used),
            report=report,
            meta={
                "strict_semicircle": bool(strict_semicircle),
                "semicircle_tol": float(semicircle_tol),
                "reducer": reducer_meta,
                "compute_chart_disagreement": bool(compute_chart_disagreement),
                "packing": packing,  # keep as FramePacking for consistency
            },
        )
        
  

    def get_bundle_map(
        self,
        *,
        edges: Optional[Iterable[Edge]] = None,
        subcomplex: SubcomplexMode = "full",
        persistence=None,
        strict_semicircle: bool = True,
        semicircle_tol: float = 1e-8,
        reducer=None,
        packing: FramePacking = "coloring",
        recompute: bool = False,
        show_summary: bool = False,
        compute_chart_disagreement: bool = True,
    ) -> BundleMapResult:
        """
        Cached wrapper for :meth:`compute_bundle_map`.

        Parameters
        ----------
        edges:
            Explicit edge set to use. If provided, ``subcomplex`` is ignored.
        subcomplex:
            If ``edges`` is not provided, choose edges from:
            ``"full"`` (all nerve edges), ``"cocycle"``, or ``"max_trivial"`` (derived from persistence).
        persistence:
            Required when ``subcomplex != "full"``.
        strict_semicircle, semicircle_tol, reducer, packing, compute_chart_disagreement:
            See :meth:`compute_bundle_map`.
        recompute:
            If True, ignore cached results and recompute.
        show_summary:
            If True, print a solver summary (passed through to :meth:`compute_bundle_map`).

        Returns
        -------
        bundle_map:
            The cached or newly computed :class:`BundleMapResult`.
        """
        edges_used = edges
        subcomplex_key = None

        if edges_used is None:
            subcomplex = str(subcomplex)
            if subcomplex not in {"full", "cocycle", "max_trivial"}:
                raise ValueError(f"subcomplex must be one of 'full','cocycle','max_trivial'. Got {subcomplex!r}")
            subcomplex_key = subcomplex

            if subcomplex != "full":
                p = persistence if persistence is not None else getattr(self, "persistence", None)
                if p is None:
                    raise ValueError(
                        "subcomplex != 'full' requires persistence. "
                        "Compute bundle.persistence first or pass persistence=..."
                    )
                from .analysis.class_persistence import _edges_for_subcomplex_from_persistence
                edges_used = _edges_for_subcomplex_from_persistence(p, subcomplex)


        red_key = None
        if reducer is not None:
            red_key = (
                getattr(reducer, "method", None),
                getattr(reducer, "stage", None),      # <-- IMPORTANT: include stage
                getattr(reducer, "d", None),
                getattr(reducer, "max_frames", None),
                getattr(reducer, "rng_seed", None),
                getattr(reducer, "psc_verbosity", None),
            )

        edges_key = None
        if edges_used is not None:
            edges_key = tuple(sorted({canon_edge(*e) for e in edges_used if e[0] != e[1]}))

        key = (
            "bundle_map",
            edges_key,
            subcomplex_key,
            str(packing),
            bool(strict_semicircle),
            float(semicircle_tol),
            red_key,
            bool(compute_chart_disagreement),
        )

        if recompute or key not in self._cache:
            self._cache[key] = self.compute_bundle_map(
                edges=edges_used,
                strict_semicircle=strict_semicircle,
                semicircle_tol=semicircle_tol,
                reducer=reducer,
                packing=packing,
                show_summary=show_summary,
                compute_chart_disagreement=compute_chart_disagreement,
            )

        bm = self._cache[key]

        if (
            self.bundle_map is None
            and edges is None
            and subcomplex == "full"
            and reducer is None
            and str(packing) == "coloring"
            and bool(strict_semicircle) is True
            and float(semicircle_tol) == 1e-8
            and bool(compute_chart_disagreement) is True
        ):
            self.bundle_map = bm

        return bm

    def get_pullback_data(
        self,
        *,
        bundle_map: Optional["BundleMapResult"] = None,
        edges: Optional[Iterable[Edge]] = None,
        subcomplex: SubcomplexMode = "full",
        persistence=None,
        strict_semicircle: bool = True,
        semicircle_tol: float = 1e-8,
        reducer=None,
        recompute_bundle_map: bool = False,
        compute_chart_disagreement: bool = True,
        packing: FramePacking = "coloring2",
        base_weight: float = 1.0,
        fiber_weight: float = 1.0,
        show_summary: bool = True,
    ) -> "PullbackTotalSpaceResult":
        """
        Construct the pullback total space ``Z = (base | fiber)`` and a compatible product metric.

        This is a convenience method for downstream tasks that want an explicit
        Euclidean-like representation of the reconstructed bundle, e.g. clustering
        or nearest-neighbor search in a product space.

        Parameters
        ----------
        bundle_map:
            Optional precomputed :class:`BundleMapResult`. If provided, bundle-map options are ignored.
        edges, subcomplex, persistence, strict_semicircle, semicircle_tol, reducer, compute_chart_disagreement, packing:
            Options used to compute the bundle map when ``bundle_map`` is not provided.
            See :meth:`get_bundle_map`.
        recompute_bundle_map:
            If True and ``bundle_map`` is not provided, forces recomputation of the bundle map.
        base_weight:
            Weight applied to base distances in the product metric.
        fiber_weight:
            Weight applied to fiber distances in the product metric.
        show_summary:
            If True, display a short summary of the bundle-map report and the ambient product space.

        Returns
        -------
        pullback:
            A :class:`PullbackTotalSpaceResult` containing concatenated data and a product metric.

        Raises
        ------
        ValueError:
            If base/fiber shapes are incompatible (e.g. different number of samples).
        """
        from .metrics import ProductMetricConcat, EuclideanMetric, as_metric
        from .trivializations.bundle_map import show_bundle_map_summary

        bm = bundle_map
        if bm is None:
            bm = self.get_bundle_map(
                edges=edges,
                subcomplex=subcomplex,
                persistence=persistence,
                strict_semicircle=strict_semicircle,
                semicircle_tol=semicircle_tol,
                reducer=reducer,
                recompute=recompute_bundle_map,
                show_summary=False,
                compute_chart_disagreement=compute_chart_disagreement,
                packing=packing,
            )

        fiber = np.asarray(bm.F, dtype=float)
        base = np.asarray(self.cover.base_points, dtype=float)

        if base.ndim == 1:
            base = base.reshape(-1, 1)
        elif base.ndim != 2:
            raise ValueError(f"cover.base_points must be 1D or 2D. Got {base.shape}.")

        if fiber.ndim != 2:
            raise ValueError(f"bundle_map.F must be 2D. Got {fiber.shape}.")
        if base.shape[0] != fiber.shape[0]:
            raise ValueError(f"base and fiber must have same n_samples: {base.shape[0]} vs {fiber.shape[0]}")

        total_data = np.concatenate([base, fiber], axis=1)

        base_metric = getattr(self.cover, "metric", None)
        if base_metric is None:
            base_metric = EuclideanMetric()
        base_metric = as_metric(base_metric)

        pm = ProductMetricConcat(
            base_metric=base_metric,
            base_dim=int(base.shape[1]),
            base_weight=float(base_weight),
            fiber_weight=float(fiber_weight),
            name=f"product_concat(base={getattr(base_metric,'name','metric')},bw={base_weight},fw={fiber_weight})",
        )

        out = PullbackTotalSpaceResult(
            total_data=total_data,
            metric=pm,
            bundle_map=bm,
            base_dim=int(base.shape[1]),
            fiber_dim=int(fiber.shape[1]),
            base_weight=float(base_weight),
            fiber_weight=float(fiber_weight),
            meta={
                "subcomplex": str(subcomplex),
                "strict_semicircle": bool(strict_semicircle),
                "semicircle_tol": float(semicircle_tol),
                "compute_chart_disagreement": bool(compute_chart_disagreement),
                "packing": packing,
            },
        )

        if show_summary:
            rep = getattr(bm, "report", None)
            if rep is not None:
                ambient_dim = int(total_data.shape[1])
                ambient_fiber_dim = int(fiber.shape[1])

                def _cover_base_symbol_latex(cover) -> str:
                    m = getattr(cover, "metric", None)
                    mname = getattr(m, "name", "")
                    if str(mname).strip().lower() == "euclidean":
                        return ""
                    s = getattr(cover, "base_name_latex", None)
                    if isinstance(s, str) and s.strip():
                        return s.strip()
                    s = getattr(cover, "base_name", None)
                    if isinstance(s, str) and s.strip():
                        return s.strip()
                    return "B"

                B = _cover_base_symbol_latex(self.cover)
                if B == "":
                    rhs = rf"\widetilde{{E}}\subset\mathbb{{R}}^{{{ambient_dim}}}"
                else:
                    rhs = rf"\widetilde{{E}}\subset {B}\times\mathbb{{R}}^{{{ambient_fiber_dim}}}"

                extra_rows = [(r"\text{Ambient space}", rhs)]

                show_bundle_map_summary(
                    bm.report,
                    show=True,
                    mode="auto",
                    rounding=3,
                    extra_rows=extra_rows,
                )

        return out

    # ============================================================
    # Visualization (optional deps, lazily imported)
    # ============================================================

    def bundle_app(
        self,
        *,
        get_dist_mat,
        initial_r: float = 0.1,
        r_max: float = 2.0,
        colors=None,
        densities=None,
        data_landmark_inds=None,
        landmarks=None,
        max_samples: int = 10_000,
        base_metric=None,
        rng=None,
    ):
        """
        Create a Dash application for interactive bundle exploration (does not run it).

        This constructs and returns a Dash app object. To actually run the app, use
        :meth:`show_bundle`.

        Parameters
        ----------
        get_dist_mat:
            Callable used to compute pairwise distances among sampled points for interactive filtering.
            Signature is implementation-defined (typically ``get_dist_mat(X) -> (n,n)``).
        initial_r:
            Initial distance threshold for the interactive neighborhood slider.
        r_max:
            Maximum value of the neighborhood slider.
        colors:
            Optional per-sample colors (passed through to the viewer).
        densities:
            Optional per-sample density visualizations or payloads (viewer-dependent).
        data_landmark_inds:
            Optional indices mapping samples to landmarks (viewer-dependent).
        landmarks:
            Optional base landmark coordinates. Defaults to ``cover.landmarks`` when available.
        max_samples:
            Maximum number of samples shown in the viewer (subsamples if needed).
        base_metric:
            Optional base metric object used for landmark embedding / distances.
        rng:
            Optional RNG used for subsampling.

        Returns
        -------
        app:
            A Dash app instance.

        Raises
        ------
        ImportError:
            If optional visualization dependencies (dash/plotly/scikit-learn) are not installed.
        """
        # Default base landmarks from cover unless explicitly overridden
        if landmarks is None:
            landmarks = getattr(self.cover, "landmarks", None)

            # Normalize 1D landmark point -> (1, d_base)
            if isinstance(landmarks, np.ndarray):
                landmarks = np.asarray(landmarks)
                base_points = getattr(self.cover, "base_points", None)
                if base_points is not None:
                    bp = np.asarray(base_points)
                    d_base = int(bp.shape[1]) if bp.ndim == 2 else 1
                    if landmarks.ndim == 1 and landmarks.shape[0] == d_base:
                        landmarks = landmarks.reshape(1, d_base)

        try:
            from .viz.bundle_dash import prepare_bundle_viz_inputs_from_bundle, make_bundle_app
        except ImportError as e:
            raise ImportError(
                "BundleResult.bundle_app() requires optional dependencies (dash, plotly, scikit-learn). "
                "Install them to use the interactive bundle viewer."
            ) from e

        viz = prepare_bundle_viz_inputs_from_bundle(
            self,
            get_dist_mat=get_dist_mat,
            max_samples=max_samples,
            base_metric=base_metric,
            colors=colors,
            densities=densities,
            data_landmark_inds=data_landmark_inds,
            landmarks=landmarks,
            rng=rng,
        )
        return make_bundle_app(viz, initial_r=float(initial_r), r_max=float(r_max))

    def show_bundle(
        self,
        *,
        get_dist_mat: Optional[Callable[..., np.ndarray]] = None,
        initial_r: float = 0.1,
        r_max: float = 2.0,
        colors=None,
        densities=None,
        data_landmark_inds=None,
        landmarks=None,
        max_samples: int = 10_000,
        base_metric=None,
        rng=None,
        debug: bool = False,
        port: Optional[int] = None,
    ):
        """
        Run the interactive Dash bundle viewer.

        This is a convenience wrapper that constructs the app (see :meth:`bundle_app`)
        and runs it in a local server.

        Parameters
        ----------
        get_dist_mat, initial_r, r_max, colors, densities, data_landmark_inds, landmarks, max_samples, base_metric, rng:
            Passed through to :meth:`bundle_app`.
        debug:
            If True, run the Dash server in debug mode.
        port:
            Optional port to bind the server to. If omitted, Dash chooses a default (or internal logic may pick a free port).

        Returns
        -------
        app:
            The Dash app instance that was run.

        Raises
        ------
        ImportError:
            If optional visualization dependencies (dash/plotly/scikit-learn) are not installed.
        """
        if base_metric is None:
            base_metric = getattr(self.cover, "metric", None)
        if base_metric is None:
            from .metrics import EuclideanMetric
            base_metric = EuclideanMetric()

        if landmarks is None:
            landmarks = getattr(self.cover, "landmarks", None)
            if isinstance(landmarks, np.ndarray):
                landmarks = np.asarray(landmarks)
                base_points = getattr(self.cover, "base_points", None)
                if base_points is not None:
                    bp = np.asarray(base_points)
                    d_base = int(bp.shape[1]) if bp.ndim == 2 else 1
                    if landmarks.ndim == 1 and landmarks.shape[0] == d_base:
                        landmarks = landmarks.reshape(1, d_base)

        try:
            from .viz.bundle_dash import (
                prepare_bundle_viz_inputs_from_bundle,
                make_bundle_app,
                run_bundle_app,
            )
        except ImportError as e:
            raise ImportError(
                "BundleResult.show_bundle() requires optional dependencies (dash, plotly, scikit-learn). "
                "Install them to use the interactive bundle viewer."
            ) from e

        viz = prepare_bundle_viz_inputs_from_bundle(
            self,
            get_dist_mat=get_dist_mat,
            max_samples=max_samples,
            base_metric=base_metric,
            colors=colors,
            densities=densities,
            data_landmark_inds=data_landmark_inds,
            landmarks=landmarks,
            rng=rng,
        )
        app = make_bundle_app(viz, initial_r=float(initial_r), r_max=float(r_max))
        run_bundle_app(app, port=port, debug=bool(debug))
        return app

    def show_nerve(
        self,
        *,
        title: Optional[str] = None,
        show_labels: bool = True,
        show_axes: bool = False,
        tri_opacity: float = 0.25,
        tri_color: str = "pink",
        cochains: Optional[List[Dict[Tuple[int, ...], object]]] = None,
        edge_weight_source: Optional[str] = None,
        edge_weights: Optional[Dict[Edge, float]] = None,
        edge_cutoff: Optional[float] = None,
        highlight_edges: Optional[Set[Edge]] = None,
        highlight_color: str = "red",
    ):
        """
        Visualize the cover nerve (Plotly).

        Parameters
        ----------
        title:
            Optional plot title.
        show_labels:
            If True, display vertex labels.
        show_axes:
            If True, show coordinate axes.
        tri_opacity, tri_color:
            Appearance options for triangle faces (if present).
        cochains:
            Optional list of cochains to visualize (passed through to the cover nerve plotter).
        edge_weight_source:
            If ``edge_weights`` is not provided, choose a built-in source:
            ``"rms"`` (transition RMS error), ``"witness"`` (witness error), or ``"none"``.
            Defaults to ``bundle.meta["prefer_edge_weight"]`` when available.
        edge_weights:
            Explicit per-edge weights keyed by nerve edges ``(i, j)``.
        edge_cutoff:
            Optional cutoff threshold for displaying edges based on weight.
        highlight_edges:
            Optional set of edges to highlight.
        highlight_color:
            Color used for highlighted edges.

        Returns
        -------
        fig:
            A Plotly figure object (or whatever the cover’s ``show_nerve`` returns).
        """
        cover = self.cover

        if edge_weight_source is None:
            edge_weight_source = self.meta.get("prefer_edge_weight", "rms")

        if edge_weights is None:
            if edge_weight_source == "rms":
                edge_weights = getattr(self.transitions, "rms_angle_err", None)
            elif edge_weight_source == "witness":
                edge_weights = getattr(self.quality, "witness_err", None)
            elif edge_weight_source == "none":
                edge_weights = None
            else:
                raise ValueError("edge_weight_source must be 'rms', 'witness', 'none', or None.")

        return cover.show_nerve(
            title=title,
            show_labels=show_labels,
            show_axes=show_axes,
            tri_opacity=tri_opacity,
            tri_color=tri_color,
            cochains=cochains,
            edge_weights=edge_weights,
            edge_cutoff=edge_cutoff,
            highlight_edges=highlight_edges,
            highlight_color=highlight_color,
        )

    def show_max_trivial(
        self,
        *,
        prefer_edge_weight: Optional[str] = None,
        edge_weights: Optional[Dict[Edge, float]] = None,
        recompute: bool = False,
        title: Optional[str] = None,
        show_labels: bool = True,
        show_axes: bool = False,
        tri_opacity: float = 0.5,
        tri_color: str = "pink",
        hide_removed_edges: bool = True,
        highlight_removed: bool = True,
        removed_color: str = "red",
        highlight_kept: bool = False,
        kept_color: str = "green",
    ):
        """
        Visualize the max-trivial subcomplex (Plotly).

        The max-trivial subcomplex is the largest subcomplex (in the edge-filtration sense)
        on which selected characteristic class representatives become coboundaries.

        Parameters
        ----------
        prefer_edge_weight, edge_weights, recompute:
            Used to compute the max-trivial subcomplex. See :meth:`get_max_trivial_subcomplex`.
        title:
            Optional plot title.
        show_labels, show_axes:
            Display options for landmark labels / axes.
        tri_opacity, tri_color:
            Appearance options for triangle faces.
        hide_removed_edges:
            If True, draw only the kept edges (and triangles supported by kept edges).
        highlight_removed:
            If True, highlight removed edges.
        removed_color:
            Color for removed edges when highlighted.
        highlight_kept:
            If True, additionally highlight kept edges (thicker overlay).
        kept_color:
            Color for kept-edge highlighting overlay.

        Returns
        -------
        fig:
            A Plotly figure.
        """
        from .viz.nerve_plotly import make_nerve_figure, embed_landmarks

        if prefer_edge_weight is None:
            prefer_edge_weight = self.meta.get("prefer_edge_weight", "rms")

        max_triv = self.get_max_trivial_subcomplex(
            prefer_edge_weight=prefer_edge_weight,
            edge_weights=edge_weights,
            recompute=recompute,
        )

        cover = self.cover
        landmarks = np.asarray(cover.landmarks)

        all_edges = list(cover.nerve_edges())
        all_tris = list(cover.nerve_triangles())

        removed_set = {canon_edge(*e) for e in max_triv.removed_edges}
        kept_set = {canon_edge(*e) for e in max_triv.kept_edges}

        if hide_removed_edges:
            edges_to_draw = sorted(kept_set)
            kept_pairs = set(kept_set)
            tris_to_draw = []
            for t in all_tris:
                i, j, k = canon_tri(*t)
                if (canon_edge(i, j) in kept_pairs) and (canon_edge(i, k) in kept_pairs) and (canon_edge(j, k) in kept_pairs):
                    tris_to_draw.append((i, j, k))
        else:
            edges_to_draw = all_edges
            tris_to_draw = all_tris

        if title is None:
            title = "Max Subcomplex On Which Class Representatives Are Coboundaries"

        fig = make_nerve_figure(
            landmarks=landmarks,
            edges=edges_to_draw,
            triangles=tris_to_draw,
            show_labels=show_labels,
            show_axes=show_axes,
            tri_opacity=tri_opacity,
            tri_color=tri_color,
            highlight_edges=(removed_set if highlight_removed else None),
            highlight_color=removed_color,
            title=title,
        )

        if highlight_kept and kept_set:
            import plotly.graph_objects as go
            emb = embed_landmarks(landmarks)
            n = emb.shape[0]
            hx, hy, hz = [], [], []
            for (i, j) in sorted(kept_set):
                if not (0 <= i < n and 0 <= j < n):
                    continue
                hx += [emb[i, 0], emb[j, 0], None]
                hy += [emb[i, 1], emb[j, 1], None]
                hz += [emb[i, 2], emb[j, 2], None]
            fig.add_trace(
                go.Scatter3d(
                    x=hx,
                    y=hy,
                    z=hz,
                    mode="lines",
                    line=dict(width=6.0, color=kept_color),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        fig.show()
        return fig

    def show_circle_nerve(
        self,
        *,
        use_max_trivial: bool = True,
        prefer_edge_weight: Optional[str] = None,
        edge_weights_for_max_trivial: Optional[Dict[Edge, float]] = None,
        recompute: bool = False,
        omega: Optional[Dict[Edge, int]] = None,
        weights: Optional[Dict[Edge, float]] = None,
        phi: Optional[Dict[int, int]] = None,
        weights_source: str = "rms",
        reorder_cycle: bool = True,
        fail_if_not_cycle: bool = True,
        auto_omega: bool = True,
        compute_phi: bool = True,
        phi_source: str = "orient_if_possible",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ):
        """
        Circle-layout nerve visualization for cycle graphs.

        This helper is intended for covers whose nerve graph is a single cycle. It can display:
        - optional per-edge weights (e.g. RMS transition errors),
        - an O(1) cocycle ``omega`` when available, and
        - an optional vertex potential ``phi`` (orientation gauge) derived from orientability.

        Parameters
        ----------
        use_max_trivial:
            If True, compute and display the kept edge set from the max-trivial subcomplex.
        prefer_edge_weight, edge_weights_for_max_trivial, recompute:
            Options for computing the max-trivial subcomplex (used only when ``use_max_trivial=True``).
        omega:
            Optional O(1) data keyed by edges. If omitted and ``auto_omega=True``, uses
            ``bundle.classes.omega_O1_used`` when available.
        weights:
            Optional per-edge weights. If omitted, ``weights_source`` chooses a built-in source.
        phi:
            Optional vertex potential (e.g. ±1 per chart). If omitted and ``compute_phi=True``,
            it will be computed from orientability logic (see ``phi_source``).
        weights_source:
            Which built-in weight source to use when ``weights`` is omitted:
            ``"rms"``, ``"witness"``, or ``"none"``.
        reorder_cycle:
            If True, reorder vertices along the cycle before plotting for a cleaner layout.
        fail_if_not_cycle:
            If True, raise an error when the nerve is not a single cycle.
        title, save_path, show:
            Plot labeling/output options.
        kwargs:
            Passed through to the underlying circle-nerve visualizer.

        Returns
        -------
        out:
            The underlying visualizer return value (often a Plotly/Matplotlib object depending on implementation).
        """
        from .viz.nerve_circle import (
            show_circle_nerve,
            is_single_cycle_graph,
            cycle_order_from_edges,
            reindex_edges,
            reindex_edge_dict,
            reindex_vertex_dict,
        )

        cover = self.cover
        if getattr(cover, "U", None) is None:
            raise AttributeError("bundle.cover.U is missing; build the cover first.")
        n = int(cover.U.shape[0])

        edges = list(cover.nerve_edges())
        ok, msg = is_single_cycle_graph(n, edges)
        if (not ok) and fail_if_not_cycle:
            raise ValueError(f"cover nerve is not a single cycle graph: {msg}")

        kept_edges = None
        if use_max_trivial:
            if prefer_edge_weight is None:
                prefer_edge_weight = self.meta.get("prefer_edge_weight", "rms")
            max_triv = self.get_max_trivial_subcomplex(
                prefer_edge_weight=prefer_edge_weight,
                edge_weights=edge_weights_for_max_trivial,
                recompute=recompute,
            )
            kept_edges = list(max_triv.kept_edges)

        if omega is None and auto_omega:
            omega = getattr(getattr(self, "classes", None), "omega_O1_used", None)

        if weights is None and weights_source != "none":
            if weights_source == "rms":
                weights = getattr(self.transitions, "rms_angle_err", None)
            elif weights_source == "witness":
                weights = getattr(self.quality, "witness_err", None)
            else:
                raise ValueError("weights_source must be 'rms', 'witness', or 'none'.")

        if phi is None and compute_phi:
            if phi_source != "orient_if_possible":
                raise ValueError("phi_source currently only supports 'orient_if_possible'.")
            edge_set_for_phi = kept_edges if kept_edges is not None else edges
            cocycle = getattr(getattr(self, "classes", None), "cocycle_used", None)
            if cocycle is None:
                raise AttributeError("bundle.classes.cocycle_used is missing; cannot compute phi.")
            Omega = cocycle.restrict(edge_set_for_phi)
            phi_vec = Omega.orient_if_possible(edge_set_for_phi)[2]
            phi = {i: int(phi_vec[i]) for i in range(n)}

        if reorder_cycle and ok:
            order = cycle_order_from_edges(n, edges, start=0)
            old_to_new = {old: new for new, old in enumerate(order)}

            edges2 = reindex_edges(edges, old_to_new)
            kept2 = reindex_edges(kept_edges, old_to_new) if kept_edges is not None else None
            omega2 = reindex_edge_dict(omega, old_to_new)
            w2 = reindex_edge_dict(weights, old_to_new)
            phi2 = reindex_vertex_dict(phi, old_to_new)
        else:
            edges2, kept2, omega2, w2, phi2 = edges, kept_edges, omega, weights, phi

        if title is None:
            title = "Nerve Visualization"

        return show_circle_nerve(
            n_vertices=n,
            edges=edges2,
            kept_edges=kept2,
            omega=omega2,
            weights=w2,
            phi=phi2,
            title=title,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def compare_trivs(
        self,
        *,
        ncols: Union[int, str] = "auto",
        title_size: int = 14,
        align: bool = False,
        s: float = 1.0,
        save_path: Optional[str] = None,
        max_pairs: int = 25,
        metric: str = "mean",
        show: bool = True,
        return_selected: bool = False,
    ):
        """
        Compare local trivializations on overlaps (static diagnostic plot).

        This renders a grid of overlap comparisons to help identify problematic chart pairs.

        Parameters
        ----------
        ncols:
            Number of columns in the comparison grid (or ``"auto"`` to choose automatically).
        title_size:
            Font size for subplot titles.
        align:
            If True, align trivializations before comparison (implementation-defined).
        s:
            Marker size scaling (visualization parameter).
        save_path:
            If provided, save the figure to this path.
        max_pairs:
            Maximum number of overlap pairs to display.
        metric:
            Scoring method used to rank/choose overlap pairs (implementation-defined; e.g. ``"mean"``).
        show:
            If True, display the plot.
        return_selected:
            If True, also return which overlap pairs were selected.

        Returns
        -------
        fig_or_output:
            Whatever the underlying ``compare_trivs`` returns (often a Matplotlib figure).
        """
        from .viz.angles import compare_trivs
        return compare_trivs(
            cover=self.cover,
            f=self.local_triv.f,
            edges=list(self.cover.nerve_edges()),
            ncols=ncols,
            title_size=title_size,
            align=align,
            s=s,
            save_path=save_path,
            show=show,
            max_pairs=max_pairs,
            metric=metric,
            return_selected=return_selected,
        )

    
    
    

# ============================================================
# build_bundle pipeline 
# ============================================================

def build_bundle(
    data: np.ndarray,
    cover,
    *,
    total_metric=None,
    cc: object = "pca2",
    min_patch_size: int = 10,
    weights=None,
    min_points_edge: int = 5,
    ref_angle: float = 0.0,
    delta_min_points: int = 5,
    compute_witness: bool = False,
    show: bool = False,
    verbose: Optional[bool] = None,
) -> "BundleResult":
    """
    Build a discrete circle bundle from data over a specified base-space cover.

    This is the core end-to-end pipeline:

    1. ensure/build the cover (membership matrix ``U`` and partition of unity ``pou``),
    2. compute local circle coordinates on each chart (local trivialization),
    3. estimate O(2) transition functions on overlaps,
    4. compute quality diagnostics, and
    5. compute characteristic class representatives (e.g. SW1/Euler-type data when available).

    Parameters
    ----------
    data:
        Data array of shape ``(n_samples, ambient_dim)`` representing points in the total space.
    cover:
        Cover object over the base space. Must provide (after ``cover.build()``):
        ``U`` (membership matrix), ``pou`` (partition of unity), and nerve accessors.
        If ``U``/``pou`` are missing, this function calls ``cover.build()``.
    total_metric:
        Optional metric used by local trivialization routines (when they support it).
        If omitted, routines typically default to Euclidean behavior.
    cc:
        Local circle-coordinate method/config. Common choices include:
        - ``"pca2"`` for a fast PCA-based circle coordinate method, or
        - a Dreimac configuration object, or
        - a custom callable implementing your CC interface.
    min_patch_size:
        Minimum number of points required in a chart to attempt local trivialization.
        Charts with fewer points are treated as failures.
    weights:
        Optional per-sample or per-overlap weights used in transition estimation
        (implementation-defined; pass only if you know the expected format).
    min_points_edge:
        Minimum number of overlap points required to fit a transition on an edge.
        Overlaps with fewer points are treated as errors.
    ref_angle:
        Reference angle (in radians) used to fix reflection conventions in O(2) fitting.
    delta_min_points:
        Minimum number of points used by certain consistency/quality checks on simplices.
    compute_witness:
        If True, compute witness-based diagnostics (may be slower).
    show:
        If True, display a summary visualization after construction.
    verbose:
        If True, print status messages during construction. Defaults to ``show``.

    Returns
    -------
    bundle:
        A :class:`BundleResult` packaging the cover, local trivialization, transitions,
        diagnostics, characteristic classes, and convenience methods.

    Raises
    ------
    ValueError:
        If local trivialization fails on any chart or if required overlaps are missing.
    """
    from .trivializations.local_triv import compute_local_triv, DreimacCCConfig
    from .o2_cocycle import estimate_transitions
    from .analysis.quality import compute_bundle_quality
    from .characteristic_class import compute_classes, show_summary
    from .utils.status_utils import _status, _status_clear

    if verbose is None:
        verbose = bool(show)

    data = np.asarray(data)

    # Build cover if needed
    if getattr(cover, "U", None) is None or getattr(cover, "pou", None) is None:
        cover.build()

    edges = list(cover.nerve_edges())
    triangles = list(cover.nerve_triangles())
    tets = list(cover.nerve_tetrahedra())

    # ----------------------------
    # Local trivializations
    # ----------------------------
    triv = compute_local_triv(
        data,
        cover.U,
        cc=cc,
        total_metric=total_metric,
        min_patch_size=min_patch_size,
        verbose=bool(verbose),
        fail_fast=True,  # canonical behavior
    )

    if not np.all(triv.valid):
        raise ValueError(f"Local triv failed on sets: {sorted(triv.errors.keys())}")

    # ----------------------------
    # Transition estimation
    # ----------------------------
    if verbose:
        _status_clear()
        _status("Estimating transition functions...")

    cocycle, trep = estimate_transitions(
        cover.U,
        triv.f,
        edges=edges,
        weights=weights,
        min_points=min_points_edge,
        ref_angle=ref_angle,
        fail_fast_missing=True,  # canonical behavior
    )

    # ----------------------------
    # Quality summary
    # ----------------------------
    if verbose:
        _status("Gathering summary data...")

    quality = compute_bundle_quality(
        cover,
        triv,
        cocycle,
        trep,
        edges=edges,
        triangles=triangles,
        delta_min_points=delta_min_points,
        delta_fail_fast=True,  # canonical behavior
        compute_witness=compute_witness,
    )

    # ----------------------------
    # Characteristic classes (always)
    # ----------------------------
    if verbose:
        _status("Computing characteristic class representatives...")

    classes = compute_classes(
        cover,
        cocycle,
        edges=edges,
        triangles=triangles,
        tets=tets,
        try_orient=True,
        compute_euler_num=True,
    )

    # ----------------------------
    # Optional summary visualization
    # ----------------------------
    if show:
        if verbose:
            _status_clear()
        show_summary(classes, quality=quality, show=True)
    else:
        if verbose:
            _status_clear()

    # ----------------------------
    # Metadata (keep only what we might actually use downstream)
    # ----------------------------
    if isinstance(cc, str) and cc.lower() == "pca2":
        cc_method = "pca2"
    elif isinstance(cc, DreimacCCConfig):
        cc_method = "dreimac"
    elif callable(cc):
        cc_method = "custom_cc"
    else:
        cc_method = type(cc).__name__

    return BundleResult(
        cover=cover,
        data=data,
        total_metric=total_metric,
        local_triv=triv,
        cocycle=cocycle,
        transitions=trep,
        quality=quality,
        classes=classes,
        meta={
            "cc_method": str(cc_method),
            "min_patch_size": int(min_patch_size),
            "ref_angle": float(ref_angle),
            "min_points_edge": int(min_points_edge),
            "delta_min_points": int(delta_min_points),
            "compute_witness": bool(compute_witness),
        },
    )
