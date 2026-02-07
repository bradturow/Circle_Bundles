# circle_bundles/bundle2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple, List, Any

import numpy as np

from .o2_cocycle import O2Cocycle, TransitionReport, estimate_transitions
from .trivializations.local_triv import LocalTrivResult, compute_local_triv
from .analysis.quality import BundleQualityReport, compute_bundle_quality_from_U

# Summaries (polished + uniform)
from .summaries.nerve_summary import summarize_nerve_from_U, NerveSummary
from .summaries.local_triv_summary import summarize_local_trivs

# classes + persistence
from .analysis.class_persistence import (
    compute_bundle_persistence,
    _edges_for_subcomplex_from_persistence,
    build_edge_weights_from_transition_report,
)

# class summary (classes-only + rounding diag) + persistence block together
from .summaries.class_summary import summarize_classes_and_persistence

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
# Bundle
# ----------------------------

class Bundle:
    """
    Cover-free bundle reconstruction driver.

    Philosophy:
      - U is the only "cover structure" input.
      - We ALWAYS compute edges/triangles/tetrahedra from U automatically (min_points=1),
        up to max_simp_dim (requested).
      - We do NOT print O(2)-estimation stats in summaries (per preference).
      - Summaries are:
          (1) NerveSummary (old cover-summary style, now auto-plots when shown)
          (2) LocalTrivSummary (characteristic-class style diagnostics)
          (3) ClassSummary (Characteristic Classes + Persistence; no triv diagnostics)
      - Computation is explicit: methods NEVER auto-run prerequisites.
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

    # ----------------------------
    # requirement helpers (NO auto-running)
    # ----------------------------

    def _require_local_trivs(self) -> None:
        if self._local_triv is None or self._cocycle is None or self._transition_report is None or self._quality is None:
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

    def _require_pou(self) -> None:
        if self.pou is None:
            raise RuntimeError(
                "This method requires a partition of unity `pou`.\n"
                "Provide `pou=...` when constructing Bundle(...)."
            )

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

    def summary(
        self,
        modes: Optional[Iterable[Literal["nerve", "local_triv", "classes"]]] = None,
        *,
        show: bool = True,
        mode: str = "auto",
        latex: str | bool = "auto",
        # pass-through knobs for classes summary (ignored if classes not present)
        top_k: int = 10,
        show_weight_hist: bool = False,
        hist_bins: int = 40,
    ) -> Dict[str, object]:
        """
        Show only summaries that are already available.

        Default (modes=None):
          - always show nerve
          - show local_triv if computed
          - show classes if computed

        If modes is provided, we *attempt* those, but do not compute missing pieces:
          - missing summaries just return None in the output dict.
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
        else:
            modes_list = list(modes)

        out: Dict[str, object] = {}

        for m in modes_list:
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
        show_summary: bool = True,
        mode: str = "auto",
        latex: str | bool = "auto",
        verbose: bool = True,
    ) -> LocalTrivAndCocycle:
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

        # 3) quality
        qual = compute_bundle_quality_from_U(
            U=self.U,
            pou=self.pou,
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

        # cache nerve summary (computed from U); do not show during compute
        nerve = self.summarize_nerve(show=False, verbose=False, latex=latex)

        if show_summary:
            # show only the triv summary (nerve is always available separately)
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

        Pipeline:
          1) reps only (w1 + twisted Euler representative)
          2) edge-driven persistence on weights filtration
          3) compute “derived class data” only after restricting to a
             persistence-determined subcomplex (so reps are cocycles there)

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
            cover=self,  # only used for simplex extraction when edges/tris/tets omitted
            classes=reps,
            edges=self._edges_U,
            triangles=self._tris_U,
            tets=self._tets_U,
            edge_weights=ew,
            prefer_edge_weight=str(prefer_edge_weight),
        )
        # Recommended: store weights on result for downstream gating/debugging
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

        # ---- Combined class + persistence summary ----
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

        # invalidate global trivialization cache (depends on classes threshold)
        self._global_F = None
        self._global_meta = None

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

        Parameters
        ----------
        weight:
            Edge-weight threshold (rms-angle error etc.) for selecting edges *within* the
            max-trivial subcomplex. If None, uses the maximal-trivial threshold automatically.
            In all cases we forbid weights above the max-trivial threshold.
        pou:
            Optional partition of unity override (shape (n_sets, n_samples)).
            If provided, it is used instead of self.pou (but does not overwrite it).

        Requires (must already be computed)
        -----------------------------------
        - get_local_trivs()  -> provides self._local_triv, self._cocycle, self._transition_report
        - get_classes()      -> provides self._class_persistence (for max-trivial stage)

        Returns
        -------
        F : ndarray, shape (n_samples,)
            Global angles in radians in [0, 2π).
        """
        # ---- prerequisites (NO auto-running) ----
        if self._local_triv is None or self._cocycle is None or self._transition_report is None:
            raise RuntimeError("Run bundle.get_local_trivs(...) before get_global_trivialization().")

        if self._class_persistence is None:
            raise RuntimeError("Run bundle.get_classes(...) before get_global_trivialization().")

        # ---- partition of unity (required) ----
        P = np.asarray(pou, dtype=float) if pou is not None else None
        if P is None:
            if self.pou is None:
                raise ValueError(
                    "get_global_trivialization requires a partition of unity `pou` "
                    "(shape (n_sets, n_samples)). Pass pou=... here or provide it in Bundle(..., pou=...)."
                )
            P = np.asarray(self.pou, dtype=float)

        if P.shape != (self.n_sets, self.n_samples):
            raise ValueError(f"pou must have shape {(self.n_sets, self.n_samples)}; got {P.shape}.")

        # ---- build edge weights same as get_classes() ----
        rms = getattr(self._transition_report, "rms_angle_err", None)
        wit = getattr(self._transition_report, "witness_err", None)
        ew = build_edge_weights_from_transition_report(
            self._edges_U,
            rms_angle_err=rms,
            witness_err=wit,
            prefer="rms",
        )

        # ---- max-trivial edges from persistence ----
        kept_edges_max = _edges_for_subcomplex_from_persistence(self._class_persistence, "max_trivial")
        kept_set = {tuple(sorted((int(a), int(b)))) for (a, b) in kept_edges_max}

        if not kept_set:
            raise ValueError("Max-trivial subcomplex has no edges; cannot build global trivialization.")

        # Max allowed weight among edges in max-trivial subcomplex
        max_allowed_weight = max(
            float(ew[tuple(sorted(e))]) for e in kept_set if tuple(sorted(e)) in ew
        )

        # Default weight = maximal-trivial threshold
        if weight is None:
            w = float(max_allowed_weight)
        else:
            w = float(weight)
            if w > max_allowed_weight + 1e-12:
                raise ValueError(
                    f"weight={w:g} exceeds the maximal-trivial threshold {max_allowed_weight:g}. "
                    "Choose weight <= max-trivial threshold."
                )

        # ---- choose edges at this stage, but never beyond max-trivial ----
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

        # ---- orient the cocycle on the chosen 1-skeleton (THIS FIXES ALIGNMENT) ----
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

        # ---- apply the SAME vertex gauge to local angles ----
        f = np.asarray(self._local_triv.f, dtype=float)
        U = np.asarray(self.U, dtype=bool)

        f = apply_orientation_gauge_to_f(
            f=f,
            phi_pm1=np.asarray(phi_pm1, dtype=int),
            ref_angle=float(self._REF_ANGLE),
            U=U,
        )

        # ---- Singer using oriented theta (now compatible with gauged f) ----
        theta_use = coc_oriented.theta  # canonical edges (j<k) included
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
