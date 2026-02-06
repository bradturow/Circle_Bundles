# circle_bundles/bundle2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple, List

import numpy as np

from .o2_cocycle import O2Cocycle, TransitionReport, estimate_transitions
from .trivializations.local_triv import LocalTrivResult, compute_local_triv
from .analysis.quality import BundleQualityReport, compute_bundle_quality_from_U

# Summaries (polished + uniform)
from .summaries.nerve_summary import summarize_nerve_from_U, NerveSummary
from .summaries.local_triv_summary import summarize_local_trivs


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
          (1) NerveSummary (old cover-summary style, auto-plots when shown)
          (2) LocalTrivSummary (characteristic-class style diagnostics only)
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

        # caches
        self._local_triv: Optional[LocalTrivResult] = None
        self._cocycle: Optional[O2Cocycle] = None
        self._transition_report: Optional[TransitionReport] = None
        self._quality: Optional[BundleQualityReport] = None
        self._nerve_summary: Optional[NerveSummary] = None

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
        Nerve summary driven by U.

        Notes:
          - This function computes the nerve summary (cheap, cover-only).
          - Display is controlled by show/verbose.
          - NerveSummary.show_summary auto-plots if cardinalities exist, unless plot=False.
        """
        summ = summarize_nerve_from_U(
            self.U,
            max_simplex_dim=int(self.max_simp_dim),
            min_points_simplex=1,
            force_compute_from_U=True,
            compute_cardinalities=True,
            plot=False,  # let show_summary control display
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
        compute_missing: bool = False,
    ):
        """
        Characteristic-class style diagnostics summary (no O(2) estimation block).

        Default behavior is cache-only:
          - If local triv / quality have not been computed, this prints a small note and returns None.
          - Set compute_missing=True to compute them via get_local_trivs(show_summary=False).
        """
        if self._local_triv is None or self._quality is None:
            if compute_missing:
                self.get_local_trivs(show_summary=False)
            else:
                if show:
                    print("\n(Local trivializations not computed yet — run get_local_trivs() first.)\n")
                return None

        assert self._local_triv is not None and self._quality is not None

        summ = summarize_local_trivs(
            self._local_triv,
            n_sets=self.n_sets,
            n_samples=self.n_samples,
            quality=self._quality,
        )
        if show:
            summ.show_summary(show=True, mode=mode)
        return summ

    def summary(
        self,
        modes: Optional[Iterable[Literal["nerve", "local_triv"]]] = None,
        *,
        show: bool = True,
        mode: str = "auto",
        latex: str | bool = "auto",
        compute_missing: bool = False,
    ) -> Dict[str, object]:
        """
        Unified summary entry point.

        Default behavior:
          - Always show the cover/nerve summary (it is cover-only and cheap).
          - Only show local_triv summary if cached (unless compute_missing=True).
        """
        if modes is None:
            modes_list = ["nerve", "local_triv"]
        else:
            modes_list = list(modes)

        out: Dict[str, object] = {}
        showed_any = False

        for m in modes_list:
            if m == "nerve":
                # NEW behavior: summary() always shows nerve by default, even if not cached yet.
                out["nerve"] = self.summarize_nerve(
                    show=show,
                    mode=mode,
                    verbose=True,
                    latex=latex,
                    plot=True,
                )
                showed_any = True

            elif m == "local_triv":
                summ = self.summarize_local_trivs(show=show, mode=mode, compute_missing=compute_missing)
                if summ is not None:
                    out["local_triv"] = summ
                    showed_any = True

            else:
                raise ValueError(f"Unknown summary mode {m!r}.")

        if show and not showed_any:
            print("\n(No summaries available.)\n")

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
        """
        Compute:
          1) local trivializations (angles)
          2) O(2) cocycle transitions (needed for quality + downstream)
          3) quality report

        Display policy:
          - If show_summary=True, we show ONLY the local trivialization diagnostics summary.
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

        # 2) transitions (for cocycle): edges with overlap >= min_points_edge
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

        # 3) quality (cover-free)
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

        # Ensure nerve cache exists for downstream, but do not display during compute
        nerve = self.summarize_nerve(show=False, verbose=False, latex=latex)

        if show_summary:
            self.summarize_local_trivs(show=True, mode=mode, compute_missing=False)

        return LocalTrivAndCocycle(
            local_triv=lt,
            cocycle=cocycle,
            report=report,
            quality=qual,
            nerve=nerve,
        )

    # ----------------------------
    # Accessors
    # ----------------------------

    def get_quality(self) -> BundleQualityReport:
        if self._quality is None:
            self.get_local_trivs(show_summary=False)
        assert self._quality is not None
        return self._quality

    def get_cocycle(self) -> O2Cocycle:
        if self._cocycle is None:
            self.get_local_trivs(show_summary=False)
        assert self._cocycle is not None
        return self._cocycle

    def get_local_triv_result(self) -> LocalTrivResult:
        if self._local_triv is None:
            self.get_local_trivs(show_summary=False)
        assert self._local_triv is not None
        return self._local_triv
