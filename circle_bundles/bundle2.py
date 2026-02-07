# circle_bundles/bundle2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

from .nerve.combinatorics import Edge, canon_edge

from .o2_cocycle import O2Cocycle, TransitionReport, estimate_transitions
from .trivializations.local_triv import LocalTrivResult, compute_local_triv
from .analysis.quality import BundleQualityReport, compute_bundle_quality_from_U

# Summaries (polished + uniform)
from .summaries.nerve_summary import summarize_nerve_from_U, NerveSummary
from .summaries.local_triv_summary import summarize_local_trivs
from .summaries.class_summary import summarize_classes_and_persistence

from .summaries.bundle_map_summary import summarize_bundle_map, BundleMapSummary


# classes + persistence
from .analysis.class_persistence import (
    compute_bundle_persistence,
    _edges_for_subcomplex_from_persistence,
    build_edge_weights_from_transition_report,
)

# class reps + restricted-class report
from .characteristic_class import (
    compute_class_representatives_from_nerve,
    compute_class_data_on_complex,
)

# global trivialization (Singer only) + orientation gauge helper
from .trivializations.global_trivialization import (
    build_global_trivialization_singer,
    apply_orientation_gauge_to_f,
)

from .trivializations.bundle_map import (
    FramePacking,
    get_frame_dataset as _get_frame_dataset,
    show_bundle_map_summary,
    get_bundle_map_v2,   
)


# ----------------------------
# Return type for get_local_trivs
# ----------------------------

@dataclass
class LocalTrivAndCocycle:
    local_triv: LocalTrivResult
    cocycle: O2Cocycle
    report: TransitionReport
    quality: BundleQualityReport
    nerve: NerveSummary


# ----------------------------
# Return type for get_classes
# ----------------------------

@dataclass
class ClassesAndPersistence:
    reps: Any                 # class representatives object (lightweight)
    persistence: Any          # PersistenceResult
    restricted: Any           # derived class report on chosen subcomplex
    summary_text: str         # plain-text class summary (always returned)


# ----------------------------
# Return type for get_bundle_map
# ----------------------------

@dataclass
class BundleMapResult:
    """
    Minimal, user-facing bundle-map output.

    F:
      Global fiber coordinates in the solver's ambient frame space.
      Shape is (n_samples, D_used).
    """
    F: np.ndarray
    pre_F: np.ndarray
    Omega_used: Dict[Edge, np.ndarray]
    Phi_used: np.ndarray
    report: Any
    meta: Dict[str, Any]


# ----------------------------
# Bundle
# ----------------------------

class Bundle:
    """
    Cover-free bundle reconstruction driver.

    Philosophy:
      - U is the only "cover structure" input.
      - We ALWAYS compute edges/triangles/tetrahedra from U automatically (min_points=1),
        up to max_simp_dim (requested).
      - Summaries are:
          (1) NerveSummary
          (2) LocalTrivSummary
          (3) ClassSummary (Characteristic Classes + Persistence)
          (4) BundleMapSummary (solver summary; only shows if bundle-map has been computed)
      - Computation is explicit: methods NEVER auto-run prerequisites.
      - Bundle-map is exposed as:
          - get_frame_dataset(...)
          - get_bundle_map(...)
        and returns ONLY the fiber/total-space coordinates (no concatenation, no product metric).
    """

    _REF_ANGLE: float = 0.0

    def __init__(
        self,
        U: np.ndarray,
        X: np.ndarray,
        *,
        distance_matrix: bool = False,
        pou: Optional[np.ndarray] = None,
        total_metric: Optional[object] = None,
        max_simp_dim: int = 3,
    ):
        self.U = np.asarray(U, dtype=bool)
        self.X = np.asarray(X)
        self.distance_matrix = bool(distance_matrix)
        self.pou = pou
        self.total_metric = total_metric
        self.max_simp_dim = int(max_simp_dim)

        self._validate()

        # --- ALWAYS compute simplices from U (min_points=1) ---
        self._edges_U: List[Tuple[int, int]] = self._edges_from_U(min_points=1)
        self._tris_U: List[Tuple[int, int, int]] = (
            self._tris_from_U(min_points=1) if self.max_simp_dim >= 2 else []
        )
        self._tets_U: List[Tuple[int, int, int, int]] = (
            self._tets_from_U(min_points=1) if self.max_simp_dim >= 3 else []
        )

        # caches (trivs)
        self._local_triv: Optional[LocalTrivResult] = None
        self._cocycle: Optional[O2Cocycle] = None
        self._transition_report: Optional[TransitionReport] = None
        self._quality: Optional[BundleQualityReport] = None

        # caches (summaries)
        self._nerve_summary: Optional[NerveSummary] = None

        # caches (classes/persistence)
        self._class_reps: Optional[Any] = None
        self._class_persistence: Optional[Any] = None
        self._class_restricted: Optional[Any] = None

        # caches (global trivialization)
        self._global_F: Optional[np.ndarray] = None
        self._global_meta: Optional[Dict[str, Any]] = None

        # caches (bundle map)
        self._bundle_map_cache: Dict[Any, BundleMapResult] = {}
        self._bundle_map_last: Optional[BundleMapResult] = None

        # caches (bundle-map summary)
        self._bundle_map_summary: Optional[BundleMapSummary] = None

    # ----------------------------
    # requirement helpers (NO auto-running)
    # ----------------------------

    def _require_local_trivs(self) -> None:
        if (
            self._local_triv is None
            or self._cocycle is None
            or self._transition_report is None
            or self._quality is None
        ):
            raise RuntimeError(
                "Local trivializations/cocycle/quality not computed.\n"
                "Run: bundle.get_local_trivs(...) first."
            )

    def _require_classes(self) -> None:
        if self._class_reps is None or self._class_persistence is None or self._class_restricted is None:
            raise RuntimeError(
                "Classes/persistence not computed.\n"
                "Run: bundle.get_classes(...) first."
            )

    def _require_pou(self, pou_override: Optional[np.ndarray]) -> np.ndarray:
        """
        Return a validated partition of unity, preferring the override if provided.
        """
        P = np.asarray(pou_override, dtype=float) if pou_override is not None else None
        if P is None:
            if self.pou is None:
                raise RuntimeError(
                    "This method requires a partition of unity `pou`.\n"
                    "Provide `pou=...` when constructing Bundle(...), or pass pou=... to this method."
                )
            P = np.asarray(self.pou, dtype=float)

        if P.shape != (self.n_sets, self.n_samples):
            raise ValueError(f"pou must have shape {(self.n_sets, self.n_samples)}; got {P.shape}.")
        return P

    # ----------------------------
    # validation / properties
    # ----------------------------

    def _validate(self) -> None:
        if self.U.ndim != 2:
            raise ValueError(f"U must be 2D (n_sets, n_samples). Got {self.U.shape}.")
        n_sets, n_samples = self.U.shape

        if self.distance_matrix:
            if self.X.ndim != 2 or self.X.shape[0] != self.X.shape[1]:
                raise ValueError(
                    f"distance_matrix=True requires X to be square (n_samples,n_samples). Got {self.X.shape}."
                )
            if self.X.shape[0] != n_samples:
                raise ValueError(f"X has n={self.X.shape[0]} but U has n_samples={n_samples}.")
        else:
            if self.X.ndim != 2:
                raise ValueError(
                    f"distance_matrix=False requires X to be a point cloud (n_samples,D). Got {self.X.shape}."
                )
            if self.X.shape[0] != n_samples:
                raise ValueError(f"X has n={self.X.shape[0]} but U has n_samples={n_samples}.")

        if self.max_simp_dim < 1 or self.max_simp_dim > 5:
            raise ValueError("max_simp_dim must be between 1 and 5.")

        if self.pou is not None:
            P = np.asarray(self.pou, dtype=float)
            if P.shape != (n_sets, n_samples):
                raise ValueError(f"pou must have shape {(n_sets, n_samples)}; got {P.shape}.")

    @property
    def n_sets(self) -> int:
        return int(self.U.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.U.shape[1])

    # ----------------------------
    # simplices from U
    # ----------------------------

    def _edges_from_U(self, *, min_points: int = 1) -> List[Tuple[int, int]]:
        U = self.U
        n_sets = U.shape[0]
        out: List[Tuple[int, int]] = []
        mp = int(min_points)
        for j in range(n_sets):
            Uj = U[j]
            for k in range(j + 1, n_sets):
                if int(np.sum(Uj & U[k])) >= mp:
                    out.append((j, k))
        return out

    def _tris_from_U(self, *, min_points: int = 1) -> List[Tuple[int, int, int]]:
        U = self.U
        n_sets = U.shape[0]
        out: List[Tuple[int, int, int]] = []
        mp = int(min_points)
        for i in range(n_sets):
            Ui = U[i]
            for j in range(i + 1, n_sets):
                ij = Ui & U[j]
                if int(np.sum(ij)) < mp:
                    continue
                for k in range(j + 1, n_sets):
                    if int(np.sum(ij & U[k])) >= mp:
                        out.append((i, j, k))
        return out

    def _tets_from_U(self, *, min_points: int = 1) -> List[Tuple[int, int, int, int]]:
        U = self.U
        n_sets = U.shape[0]
        out: List[Tuple[int, int, int, int]] = []
        if n_sets < 4:
            return out
        mp = int(min_points)
        for i in range(n_sets):
            Ui = U[i]
            for j in range(i + 1, n_sets):
                ij = Ui & U[j]
                if int(np.sum(ij)) < mp:
                    continue
                for k in range(j + 1, n_sets):
                    ijk = ij & U[k]
                    if int(np.sum(ijk)) < mp:
                        continue
                    for l in range(k + 1, n_sets):
                        if int(np.sum(ijk & U[l])) >= mp:
                            out.append((i, j, k, l))
        return out

    # ----------------------------
    # summaries
    # ----------------------------

    def summarize_nerve(
        self,
        *,
        show: bool = True,
        mode: str = "auto",
        verbose: bool = True,
        latex: str | bool = "auto",
        plot: bool = True,
        show_tets_plot: bool = True,
        dpi: int = 200,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> NerveSummary:
        """
        Always works (no need to compute local trivs first).
        """
        summ = summarize_nerve_from_U(
            self.U,
            max_simplex_dim=int(self.max_simp_dim),
            min_points_simplex=1,
            force_compute_from_U=True,
            compute_cardinalities=True,
            plot=False,  # show_summary handles plotting
            show_tets_plot=show_tets_plot,
            dpi=dpi,
            figsize=figsize,
            save_path=save_path,
            verbose=False,
            latex=latex,
        )

        self._nerve_summary = summ

        if verbose:
            if mode == "auto":
                show_mode = "auto" if latex == "auto" else ("latex" if latex is True else "text")
            else:
                show_mode = mode

            summ.show_summary(
                show=bool(show),
                mode=show_mode,
                plot=("auto" if plot else False),
                show_tets_plot=bool(show_tets_plot),
                dpi=int(dpi),
                figsize=figsize,
                save_path=save_path,
            )

        return summ

    def summarize_local_trivs(
        self,
        *,
        show: bool = True,
        mode: str = "auto",
    ):
        """
        Show local-triv summary only if already computed.
        Never computes anything.
        """
        if self._local_triv is None or self._quality is None:
            return None

        summ = summarize_local_trivs(
            self._local_triv,
            n_sets=self.n_sets,
            n_samples=self.n_samples,
            quality=self._quality,
        )
        if show:
            summ.show_summary(show=True, mode=mode)
        return summ

    def summarize_classes(
        self,
        *,
        show: bool = True,
        mode: str = "auto",
        top_k: int = 10,
        show_weight_hist: bool = True,
        hist_bins: int = 40,
    ):
        """
        Show class+persistence summary only if already computed.
        Never computes anything.
        """
        if self._class_reps is None or self._class_persistence is None or self._class_restricted is None:
            return None

        summ = summarize_classes_and_persistence(
            reps=self._class_reps,
            restricted=self._class_restricted,
            persistence=self._class_persistence,
        )

        if show:
            summ.show_summary(
                show=True,
                mode=str(mode),
                top_k=int(top_k),
                show_weight_hist=bool(show_weight_hist),
                hist_bins=int(hist_bins),
            )
        return summ

    def summarize_bundle_map(
        self,
        *,
        show: bool = True,
        mode: str = "auto",
        rounding: int = 3,
    ):
        """
        Show bundle-map summary only if a bundle-map has already been computed.
        Never computes anything.
        """
        if self._bundle_map_summary is None:
            return None

        if show:
            self._bundle_map_summary.show_summary(
                show=True,
                mode=str(mode),
                rounding=int(rounding),
            )
        return self._bundle_map_summary

    def summary(
        self,
        modes: Optional[Iterable[Literal["nerve", "local_triv", "classes", "bundle_map"]]] = None,
        *,
        show: bool = True,
        mode: str = "auto",
        latex: str | bool = "auto",
        top_k: int = 10,
        show_weight_hist: bool = False,
        hist_bins: int = 40,
        bm_rounding: int = 3,
    ) -> Dict[str, object]:
        """
        Show only summaries that are already available.

        Default (modes=None):
          - always show nerve
          - show local_triv if computed
          - show classes if computed
          - show bundle_map if computed
        """
        if modes is None:
            modes_list: List[str] = ["nerve"]
            if self._local_triv is not None and self._quality is not None:
                modes_list.append("local_triv")
            if (
                self._class_reps is not None
                and self._class_persistence is not None
                and self._class_restricted is not None
            ):
                modes_list.append("classes")
            if self._bundle_map_summary is not None:
                modes_list.append("bundle_map")
        else:
            modes_list = list(modes)

        out: Dict[str, object] = {}

        # Add vertical space *between* summaries when displaying multiple blocks.
        # (We keep it simple: print a blank line before every summary after the first.)
        first_shown = True

        for m in modes_list:
            # Determine whether this summary will actually display something.
            # We only print spacing when show=True and this mode is available.
            will_show = False
            if show:
                if m == "nerve":
                    will_show = True  # always available
                elif m == "local_triv":
                    will_show = self._local_triv is not None and self._quality is not None
                elif m == "classes":
                    will_show = (
                        self._class_reps is not None
                        and self._class_persistence is not None
                        and self._class_restricted is not None
                    )
                elif m == "bundle_map":
                    will_show = self._bundle_map_summary is not None

            if will_show and not first_shown:
                print("")  # vertical space between summary blocks
            if will_show:
                first_shown = False

            if m == "nerve":
                out["nerve"] = self.summarize_nerve(
                    show=show,
                    mode=mode,
                    verbose=True,
                    latex=latex,
                    plot=True,
                )

            elif m == "local_triv":
                out["local_triv"] = self.summarize_local_trivs(show=show, mode=mode)

            elif m == "classes":
                out["classes"] = self.summarize_classes(
                    show=show,
                    mode=mode,
                    top_k=int(top_k),
                    show_weight_hist=bool(show_weight_hist),
                    hist_bins=int(hist_bins),
                )

            elif m == "bundle_map":
                out["bundle_map"] = self.summarize_bundle_map(
                    show=show,
                    mode=mode,
                    rounding=int(bm_rounding),
                )

            else:
                raise ValueError(f"Unknown summary mode {m!r}.")

        return out


    # ----------------------------
    # core: local triv + transitions + quality
    # ----------------------------

    def get_local_trivs(
        self,
        *,
        cc: object = "pca2",
        min_patch_size: int = 10,
        min_points_edge: int = 5,
        pou: Optional[np.ndarray] = None,
        show_summary: bool = True,
        mode: str = "auto",
        latex: str | bool = "auto",
        verbose: bool = True,
    ) -> LocalTrivAndCocycle:
        """
        Compute local trivializations + cocycle + quality.

        Notes
        -----
        - `pou` is OPTIONAL here: it is only used for quality diagnostics.
          If omitted, we use `self.pou` (if present). If neither exists, we still compute
          local_triv + cocycle, and quality is computed with pou=None (if your quality
          routine supports it).
        - Passing `pou=...` here does NOT overwrite self.pou.
        """
        # 1) local triv
        if self.distance_matrix:
            lt = compute_local_triv(
                data=self.X,
                U=self.U,
                cc=cc,
                total_metric=None,
                min_patch_size=int(min_patch_size),
                verbose=bool(verbose),
                fail_fast=True,
            )
        else:
            lt = compute_local_triv(
                data=self.X,
                U=self.U,
                cc=cc,
                total_metric=self.total_metric,
                min_patch_size=int(min_patch_size),
                verbose=bool(verbose),
                fail_fast=True,
            )

        # 2) transitions (edges with overlap >= min_points_edge)
        edges_est = self._edges_from_U(min_points=int(min_points_edge))
        cocycle, report = estimate_transitions(
            U=self.U,
            f=lt.f,
            edges=edges_est,
            weights=None,
            min_points=int(min_points_edge),
            ref_angle=float(self._REF_ANGLE),
            fail_fast_missing=True,
        )

        # 3) quality (allow pou override, but do not require it)
        P = None
        if pou is not None:
            P = np.asarray(pou, dtype=float)
            if P.shape != (self.n_sets, self.n_samples):
                raise ValueError(f"pou must have shape {(self.n_sets, self.n_samples)}; got {P.shape}.")
        elif self.pou is not None:
            P = np.asarray(self.pou, dtype=float)

        qual = compute_bundle_quality_from_U(
            U=self.U,
            pou=P,
            local_triv=lt,
            cocycle=cocycle,
            transitions_report=report,
            edges=edges_est,
            triangles=self._tris_U,
            delta_min_points=5,
            delta_use_euclidean=True,
            delta_fail_fast=True,
            eps_min_points=1,
            compute_witness=False,
        )

        # cache
        self._local_triv = lt
        self._cocycle = cocycle
        self._transition_report = report
        self._quality = qual

        # invalidate downstream caches
        self._class_reps = None
        self._class_persistence = None
        self._class_restricted = None
        self._global_F = None
        self._global_meta = None
        self._bundle_map_cache.clear()
        self._bundle_map_last = None
        self._bundle_map_summary = None

        # cache nerve summary (computed from U); do not show during compute
        nerve = self.summarize_nerve(show=False, verbose=False, latex=latex)

        if show_summary:
            self.summary(modes=["local_triv"], show=True, mode=mode, latex=latex)

        return LocalTrivAndCocycle(
            local_triv=lt,
            cocycle=cocycle,
            report=report,
            quality=qual,
            nerve=nerve,
        )

    # ----------------------------
    # classes + persistence
    # ----------------------------

    def get_classes(
        self,
        *,
        edge_weights: Optional[Dict[Tuple[int, int], float]] = None,
        prefer_edge_weight: str = "rms",
        show_summary: bool = True,
        mode: str = "auto",
        top_k: int = 10,
        show_weight_hist: bool = False,
        hist_bins: int = 40,
        restriction_mode: Literal["cocycle", "max_trivial"] = "cocycle",
    ) -> ClassesAndPersistence:
        """
        Compute characteristic class reps + persistence.

        IMPORTANT:
          - This does NOT auto-run get_local_trivs().
            You must run get_local_trivs(...) first.
        """
        self._require_local_trivs()
        assert self._cocycle is not None and self._transition_report is not None

        # ---- 1) reps only ----
        reps = compute_class_representatives_from_nerve(
            cocycle=self._cocycle,
            edges=self._edges_U,
            triangles=self._tris_U,
            tets=self._tets_U,
            n_vertices=self.n_sets,
            try_orient=True,
        )
        self._class_reps = reps

        # ---- 2) persistence ----
        if edge_weights is None:
            rms = getattr(self._transition_report, "rms_angle_err", None)
            wit = getattr(self._transition_report, "witness_err", None)
            ew = build_edge_weights_from_transition_report(
                self._edges_U,
                rms_angle_err=rms,
                witness_err=wit,
                prefer=str(prefer_edge_weight),
            )
        else:
            ew = {tuple(sorted((int(a), int(b)))): float(w) for (a, b), w in edge_weights.items()}

        p = compute_bundle_persistence(
            cover=self,
            classes=reps,
            edges=self._edges_U,
            triangles=self._tris_U,
            tets=self._tets_U,
            edge_weights=ew,
            prefer_edge_weight=str(prefer_edge_weight),
        )
        try:
            setattr(p, "edge_weights", dict(ew))
        except Exception:
            pass

        self._class_persistence = p

        # ---- 3) restrict + derived class data ----
        kept_edges = _edges_for_subcomplex_from_persistence(p, restriction_mode)
        kept_edges_set = {tuple(e) for e in kept_edges}

        def _induced_tris(tris):
            out = []
            for (i, j, k) in tris:
                eij = tuple(sorted((i, j)))
                eik = tuple(sorted((i, k)))
                ejk = tuple(sorted((j, k)))
                if eij in kept_edges_set and eik in kept_edges_set and ejk in kept_edges_set:
                    out.append((i, j, k))
            return out

        def _induced_tets(tets):
            out = []
            for (a, b, c, d) in tets:
                edges6 = [
                    tuple(sorted((a, b))),
                    tuple(sorted((a, c))),
                    tuple(sorted((a, d))),
                    tuple(sorted((b, c))),
                    tuple(sorted((b, d))),
                    tuple(sorted((c, d))),
                ]
                if all(e in kept_edges_set for e in edges6):
                    out.append((a, b, c, d))
            return out

        tris_sub = _induced_tris(self._tris_U)
        tets_sub = _induced_tets(self._tets_U)

        restricted = compute_class_data_on_complex(
            reps=reps,
            edges=kept_edges,
            triangles=tris_sub,
            tets=tets_sub,
            n_vertices=self.n_sets,
            compute_euler_num=True,
        )
        self._class_restricted = restricted

        summ = summarize_classes_and_persistence(
            reps=reps,
            restricted=restricted,
            persistence=p,
        )

        summary_text = str(summ.summary_text)

        if show_summary:
            summ.show_summary(
                show=True,
                mode=str(mode),
                top_k=int(top_k),
                show_weight_hist=bool(show_weight_hist),
                hist_bins=int(hist_bins),
            )

        # invalidate downstream caches
        self._global_F = None
        self._global_meta = None
        self._bundle_map_cache.clear()
        self._bundle_map_last = None
        self._bundle_map_summary = None

        return ClassesAndPersistence(
            reps=reps,
            persistence=p,
            restricted=restricted,
            summary_text=summary_text,
        )

    # ----------------------------
    # global trivialization (Singer only)
    # ----------------------------

    def get_global_trivialization(
        self,
        weight: Optional[float] = None,
        *,
        pou: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Singer-only global trivialization (degree-normalized ALWAYS).

        Requires (must already be computed)
        -----------------------------------
        - get_local_trivs()
        - get_classes()

        pou override
        ------------
        If `pou` is provided, it overrides self.pou for this call only (no overwrite).
        """
        # prerequisites (NO auto-running)
        self._require_local_trivs()
        if self._class_persistence is None:
            raise RuntimeError("Run bundle.get_classes(...) before get_global_trivialization().")
        assert self._cocycle is not None and self._transition_report is not None and self._local_triv is not None

        # partition of unity (required here)
        P = self._require_pou(pou)

        # build edge weights
        rms = getattr(self._transition_report, "rms_angle_err", None)
        wit = getattr(self._transition_report, "witness_err", None)
        ew = build_edge_weights_from_transition_report(
            self._edges_U,
            rms_angle_err=rms,
            witness_err=wit,
            prefer="rms",
        )

        # max-trivial edges from persistence
        kept_edges_max = _edges_for_subcomplex_from_persistence(self._class_persistence, "max_trivial")
        kept_set = {tuple(sorted((int(a), int(b)))) for (a, b) in kept_edges_max}
        if not kept_set:
            raise ValueError("Max-trivial subcomplex has no edges; cannot build global trivialization.")

        max_allowed_weight = max(float(ew[tuple(sorted(e))]) for e in kept_set if tuple(sorted(e)) in ew)

        if weight is None:
            w = float(max_allowed_weight)
        else:
            w = float(weight)
            if w > max_allowed_weight + 1e-12:
                raise ValueError(
                    f"weight={w:g} exceeds the maximal-trivial threshold {max_allowed_weight:g}. "
                    "Choose weight <= max-trivial threshold."
                )

        edges_stage: List[Tuple[int, int]] = []
        for (a, b) in self._edges_U:
            e = tuple(sorted((int(a), int(b))))
            if e not in kept_set:
                continue
            if float(ew.get(e, np.inf)) <= w:
                edges_stage.append(e)
        edges_stage = sorted(set(edges_stage))
        if not edges_stage:
            raise ValueError("No edges remain at this weight within the max-trivial subcomplex.")

        # orient the cocycle on the chosen 1-skeleton (fixes alignment)
        ok, coc_oriented, phi_pm1 = self._cocycle.orient_if_possible(
            edges_stage,
            n_vertices=self.n_sets,
            require_all_edges_present=False,
        )
        if not ok:
            raise ValueError(
                "Cannot build a global S¹ coordinate: the cocycle is non-orientable "
                "on the selected subcomplex (w₁ does not trivialize there)."
            )

        # apply the SAME vertex gauge to local angles
        f = np.asarray(self._local_triv.f, dtype=float)
        U = np.asarray(self.U, dtype=bool)
        f = apply_orientation_gauge_to_f(
            f=f,
            phi_pm1=np.asarray(phi_pm1, dtype=int),
            ref_angle=float(self._REF_ANGLE),
            U=U,
        )

        # Singer using oriented theta (compatible with gauged f)
        theta_use = coc_oriented.theta
        F = build_global_trivialization_singer(
            edges=edges_stage,
            U=U,
            pou=P,
            f=f,
            theta=theta_use,
            n_vertices=self.n_sets,
        )
        return np.asarray(F, dtype=float)

    # ----------------------------
    # bundle-map: frames + coordinates (NO pullback object)
    # ----------------------------

    def get_frame_dataset(
        self,
        *,
        pou: Optional[np.ndarray] = None,
        weight: Optional[float] = None,
    ):
        """
        Build the *pre-projection* frame dataset used by the bundle-map solver.

        Requires (must already be computed)
        -----------------------------------
        - get_local_trivs()   (for Omega)
        - get_classes()       (for persistence + certified "cocycle" subcomplex)
        - partition of unity  (self.pou or pou= override)

        Parameters
        ----------
        pou:
            Optional partition-of-unity override for this call (does not overwrite self.pou).
        weight:
            Optional edge-weight threshold. If provided, it must be <= the cocycle-certification
            threshold (largest weight at which class reps are cocycles). Otherwise an error is raised.

            If weight is None, we use the full cocycle-certified subcomplex.

        Notes
        -----
        - Unlike get_global_trivialization(), we do NOT require coboundaries / max-trivial.
          We only require that the class representatives are cocycles.
        - Always returns frames *before projection* (stage="pre_projection").
        - This is intended for inspection/debugging; most users call get_bundle_map().
        """
        # prerequisites (NO auto-running)
        self._require_local_trivs()
        self._require_classes()
        assert self._cocycle is not None and self._transition_report is not None

        # partition of unity (required)
        P = self._require_pou(pou)

        # persistence must exist by _require_classes()
        p = self._class_persistence
        assert p is not None

        # edge weights: prefer what's stored on persistence, else rebuild
        ew = getattr(p, "edge_weights", None)
        if ew is None:
            rms = getattr(self._transition_report, "rms_angle_err", None)
            wit = getattr(self._transition_report, "witness_err", None)
            ew = build_edge_weights_from_transition_report(
                self._edges_U,
                rms_angle_err=rms,
                witness_err=wit,
                prefer="rms",
            )

        # cocycle-certified subcomplex
        cocycle_edges = _edges_for_subcomplex_from_persistence(p, "cocycle")
        cocycle_set = {tuple(sorted((int(a), int(b)))) for (a, b) in cocycle_edges}
        if not cocycle_set:
            raise ValueError("Cocycle subcomplex has no edges; cannot build frame dataset.")

        # cocycle threshold: maximum weight among cocycle edges
        # (use only edges that appear in ew; missing weights are treated as +inf below)
        w_cocycle = max(float(ew[e]) for e in cocycle_set if e in ew)

        # optional thresholding, but weight cannot exceed cocycle threshold
        if weight is None:
            edges_used = sorted(cocycle_set)
        else:
            w = float(weight)
            if w > w_cocycle + 1e-12:
                raise ValueError(
                    f"weight={w:g} exceeds the cocycle-certification threshold {w_cocycle:g}. "
                    "Choose weight <= cocycle threshold."
                )
            edges_used = sorted(e for e in cocycle_set if float(ew.get(e, np.inf)) <= w)
            if not edges_used:
                raise ValueError("No edges remain at this weight within the cocycle subcomplex.")

        # Always pre-projection frames; remove other knobs from the public API
        return _get_frame_dataset(
            U=self.U,
            pou=P,
            Omega=self._cocycle.Omega,
            edges=edges_used,
            reducer=None,
            stage="pre_projection",
            max_frames=None,
            rng_seed=None,
            packing="coloring",  
        )

    def get_bundle_map(
        self,
        *,
        pou: Optional[np.ndarray] = None,
        weight: Optional[float] = None,
        reducer: Optional[object] = None,
        packing: FramePacking = "coloring2",
        strict_semicircle: bool = True,
        show_summary: bool = True,
        recompute: bool = False,
    ) -> BundleMapResult:
        """
        Compute (or fetch cached) bundle-map coordinatization using the v2 pipeline.

        Fixed policy (by design)
        ------------------------
        - Summary mode is always "auto".
        - Semicircle tolerance is fixed at 1e-8.
        - Summary rounding is fixed at 3.
        - Chart disagreement diagnostic is always computed.

        Requires (must already be computed)
        -----------------------------------
        - get_local_trivs()
        - get_classes()
        - partition of unity (self.pou or pou= override)
        """
        # fixed knobs
        SEMICIRCLE_TOL = 1e-8
        ROUNDING = 3
        COMPUTE_CD = True

        # prerequisites (NO auto-running)
        self._require_local_trivs()
        self._require_classes()
        assert self._cocycle is not None and self._local_triv is not None

        # partition of unity (required)
        P = self._require_pou(pou)

        # persistence exists by _require_classes()
        p = self._class_persistence
        assert p is not None

        # edge weights: prefer what's stored on persistence, else rebuild
        ew = getattr(p, "edge_weights", None)
        if ew is None:
            assert self._transition_report is not None
            rms = getattr(self._transition_report, "rms_angle_err", None)
            wit = getattr(self._transition_report, "witness_err", None)
            ew = build_edge_weights_from_transition_report(
                self._edges_U,
                rms_angle_err=rms,
                witness_err=wit,
                prefer="rms",
            )

        # cocycle-certified subcomplex
        cocycle_edges = _edges_for_subcomplex_from_persistence(p, "cocycle")
        cocycle_set = {tuple(sorted((int(a), int(b)))) for (a, b) in cocycle_edges}
        if not cocycle_set:
            raise ValueError("Cocycle subcomplex has no edges; cannot compute bundle map.")

        # cocycle threshold: maximum weight among cocycle edges (ignoring missing weights)
        w_cocycle = max(float(ew[e]) for e in cocycle_set if e in ew)

        # optional thresholding, but weight cannot exceed cocycle threshold
        if weight is None:
            edges_used = sorted(cocycle_set)
            weight_used = None
        else:
            w = float(weight)
            if w > w_cocycle + 1e-12:
                raise ValueError(
                    f"weight={w:g} exceeds the cocycle-certification threshold {w_cocycle:g}. "
                    "Choose weight <= cocycle threshold."
                )
            edges_used = sorted(e for e in cocycle_set if float(ew.get(e, np.inf)) <= w)
            if not edges_used:
                raise ValueError("No edges remain at this weight within the cocycle subcomplex.")
            weight_used = w

        # canonicalize edges for caching / downstream
        edges_key = tuple(sorted({canon_edge(*e) for e in edges_used if e[0] != e[1]}))

        # reducer participates in caching via a lightweight key
        red_key = None
        if reducer is not None:
            red_key = (
                getattr(reducer, "method", None),
                getattr(reducer, "stage", None),
                getattr(reducer, "d", None),
                getattr(reducer, "max_frames", None),
                getattr(reducer, "rng_seed", None),
                getattr(reducer, "psc_verbosity", None),
            )

        # pou override participates in caching without hashing the full matrix
        pou_key = id(pou) if pou is not None else None

        key = (
            "bundle_map_v2",
            edges_key,
            weight_used,
            str(packing),
            bool(strict_semicircle),
            float(SEMICIRCLE_TOL),
            red_key,
            bool(COMPUTE_CD),
            pou_key,
        )

        if recompute or key not in self._bundle_map_cache:
            
            # cocycle-certified subcomplex edges (list of Edge tuples)
            cocycle_edges_list = list(_edges_for_subcomplex_from_persistence(p, "cocycle"))

            F, pre_F, Omega_used, Phi_used, report = get_bundle_map_v2(
                U=self.U,
                pou=P,
                f=self._local_triv.f,
                Omega=self._cocycle.Omega,

                # IMPORTANT: let v2 own the restriction logic
                edges=None,
                weight=weight_used,

                reducer=reducer,
                packing=packing,
                strict_semicircle=bool(strict_semicircle),

                show_summary=False,  # wrapper controls summary
                compute_chart_disagreement=bool(COMPUTE_CD),

                # REQUIRED when weight is None (and still useful when weight is not None)
                persistence=p,
                cocycle_subcomplex_edges=cocycle_edges_list,
            )
            
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

            ambient_dim = int(np.asarray(F).shape[1])  # output dimension
                
            self._bundle_map_cache[key] = BundleMapResult(
                F=np.asarray(F),
                pre_F=np.asarray(pre_F),
                Omega_used=Omega_used,
                Phi_used=np.asarray(Phi_used),
                report=report,
                meta={
                    "weight": weight_used,
                    "strict_semicircle": bool(strict_semicircle),
                    "semicircle_tol": float(SEMICIRCLE_TOL),
                    "reducer": reducer_meta,
                    "compute_chart_disagreement": bool(COMPUTE_CD),
                    "packing": packing,
                    "subcomplex": "cocycle",
                    "ambient_dim": ambient_dim,
                },
            )

        bm = self._bundle_map_cache[key]
        self._bundle_map_last = bm

        # standardized summary object for bundle.summary()
        try:
            self._bundle_map_summary = summarize_bundle_map(
                bm.report,
                meta=dict(bm.meta or {}),
            )
        except Exception:
            self._bundle_map_summary = None

        if show_summary:
            if self._bundle_map_summary is not None:
                self._bundle_map_summary.show_summary(
                    show=True,
                    mode="auto",
                    rounding=int(ROUNDING),
                )
            else:
                show_bundle_map_summary(
                    bm.report,
                    show=True,
                    mode="auto",
                    rounding=int(ROUNDING),
                    extra_rows=None,
                )

        return bm

    # ----------------------------
    # Accessors (NO auto-running)
    # ----------------------------

    def get_quality(self) -> BundleQualityReport:
        if self._quality is None:
            raise RuntimeError("Quality not computed. Run: bundle.get_local_trivs(...) first.")
        return self._quality

    def get_cocycle(self) -> O2Cocycle:
        if self._cocycle is None:
            raise RuntimeError("Cocycle not computed. Run: bundle.get_local_trivs(...) first.")
        return self._cocycle

    def get_local_triv_result(self) -> LocalTrivResult:
        if self._local_triv is None:
            raise RuntimeError("Local trivializations not computed. Run: bundle.get_local_trivs(...) first.")
        return self._local_triv
