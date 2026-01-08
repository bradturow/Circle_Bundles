# covers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .metrics import Euc_met, get_dist_mat
from .geometry import get_bary_coords, points_in_triangle_mask
from .combinatorics import Edge, Tri, canon_edge, canon_tri


Tet = Tuple[int, int, int, int]


def canon_tet(a: int, b: int, c: int, d: int) -> Tet:
    return tuple(sorted((int(a), int(b), int(c), int(d))))


# ----------------------------
# Nerve summary data container
# ----------------------------

@dataclass
class NerveSummary:
    n_sets: int
    n_samples: int

    # recorded simplex counts (what the cover reports)
    n0: int
    n1: int
    n2: int
    n3: int

    # intersection cardinalities (#samples in ⋂ U_i) for recorded simplices
    vert_card: Optional[np.ndarray] = None  # (n0,)
    edge_card: Optional[np.ndarray] = None  # (n1,)
    tri_card: Optional[np.ndarray] = None   # (n2,)
    tet_card: Optional[np.ndarray] = None   # (n3,)

    # overlap order per sample: how many cover sets contain each sample
    sample_overlap_counts: Optional[np.ndarray] = None  # (n_samples,)

    # only surfaced when > 4
    max_overlap_order: Optional[int] = None
    n_samples_with_overlap_ge_5: Optional[int] = None

    warnings: Tuple[str, ...] = ()

    # ---- pretty formatting ----
    def to_text(self) -> str:
        """
        Human-readable, terminal-friendly summary.

        Policy:
          - If max_overlap_order > 4, we print ALL recorded counts for dims 0..3
            (even if some are zero), because mismatch is meaningful.
          - Otherwise, we print counts only up to the last nonzero recorded dim,
            then a single line noting higher dims are zero.
          - We only show overlap-order lines if max_overlap_order > 4.
        """
        lines: List[str] = []
        lines.append("Cover And Nerve Summary")
        lines.append(f"  n_sets = {self.n_sets}, n_samples = {self.n_samples}")

        counts = {0: int(self.n0), 1: int(self.n1), 2: int(self.n2), 3: int(self.n3)}

        if (self.max_overlap_order is not None) and (self.max_overlap_order > 4):
            lines.append("")
            lines.append("  recorded simplex counts:")
            for d in (0, 1, 2, 3):
                lines.append(f"    #( {d}-simplices ) = {counts[d]}")

            lines.append("")
            lines.append("  overlap evidence from U:")
            lines.append(f"    max sample overlap order = {self.max_overlap_order}")
            lines.append(f"    samples in ≥5 sets = {self.n_samples_with_overlap_ge_5}")

        else:
            # Truncate at last nonzero recorded dimension
            last_nonzero = 0
            for d in (3, 2, 1, 0):
                if counts[d] > 0:
                    last_nonzero = d
                    break

            lines.append("")
            lines.append("  recorded simplex counts:")
            for d in range(0, last_nonzero + 1):
                lines.append(f"    #( {d}-simplices ) = {counts[d]}")
            if last_nonzero < 3:
                lines.append(f"    no recorded simplices in dimensions ≥ {last_nonzero + 1}")

        for w in self.warnings:
            lines.append("")
            lines.append(f"  WARNING: {w}")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """
        Markdown + LaTeX-ish version for notebooks.
        (Uses inline math; renders nicely in Jupyter.)
        """
        counts = {0: int(self.n0), 1: int(self.n1), 2: int(self.n2), 3: int(self.n3)}
        md: List[str] = []
        md.append("### Cover And Nerve Summary")
        md.append(f"- $n_\\text{{sets}} = {self.n_sets}$, $n_\\text{{samples}} = {self.n_samples}$")

        if (self.max_overlap_order is not None) and (self.max_overlap_order > 4):
            md.append("")
            md.append("**Recorded Simplex Counts:**")
            md.append("")
            md.append(
                "\n".join(
                    [
                        f"- $\\#(\\text{{{d}-simplices}}) = {counts[d]}$"
                        for d in (0, 1, 2, 3)
                    ]
                )
            )

            md.append("")
            md.append("**Overlap evidence from $U$:**")
            md.append("")
            md.append(f"- $\\max_s \\sum_i U_{i,s} = {self.max_overlap_order}$")
            md.append(f"- $\\#\\{{s : \\sum_i U_{i,s} \\ge 5\\}} = {self.n_samples_with_overlap_ge_5}$")

        else:
            last_nonzero = 0
            for d in (3, 2, 1, 0):
                if counts[d] > 0:
                    last_nonzero = d
                    break

            md.append("")
            md.append("**Recorded Simplex Counts:**")
            md.append("")
            md.append(
                "\n".join(
                    [f"- $\\#(\\text{{{d}-simplices}}) = {counts[d]}$" for d in range(0, last_nonzero + 1)]
                )
            )
            if last_nonzero < 3:
                md.append(f"- *No recorded simplices in dimensions* $\\ge {last_nonzero + 1}$")

        if self.warnings:
            md.append("")
            md.append("**Warnings:**")
            md.append("")
            for w in self.warnings:
                md.append(f"- {w}")

        return "\n".join(md)


# ----------------------------
# Internal helpers (summary)
# ----------------------------

def _simplex_intersection_cardinality(U: np.ndarray, sig: Tuple[int, ...]) -> int:
    """Cardinality (#samples) of ⋂_{i in sig} U_i.  U: (n_sets, n_samples) bool."""
    if len(sig) == 0:
        return int(U.shape[1])
    mask = np.logical_and.reduce(U[list(sig)], axis=0)
    return int(mask.sum())


def _cards_for_simplices(U: np.ndarray, simplices: List[Tuple[int, ...]]) -> np.ndarray:
    """Compute intersection cardinalities for a list of simplices."""
    out = np.empty(len(simplices), dtype=int)
    for idx, sig in enumerate(simplices):
        out[idx] = _simplex_intersection_cardinality(U, tuple(sig))
    return out


def _try_display_markdown(md: str) -> bool:
    """
    Try to display markdown in a notebook. Returns True if it displayed, else False.
    """
    try:
        from IPython.display import display, Markdown  # type: ignore
        display(Markdown(md))
        return True
    except Exception:
        return False


# ----------------------------
# Plot helpers (box-and-whisker)
# ----------------------------

def plot_cover_summary_boxplot(
    summary: NerveSummary,
    *,
    show_tets: bool = True,
    dpi: int = 200,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    showfliers: bool = False,
    whis=(0, 100),          # <--- full range by default
    sharey: bool = False,
    label_fmt: str = "{:g}",
    label_dx: float = 0.06, # x-offset in axis units (relative)
):
    """
    Box-and-whisker plots for intersection cardinalities of recorded simplices.

    - Uses whis=(0,100) by default => whiskers are true min/max (no outlier logic).
    - Adds min/max labels placed to the RIGHT of the whiskers.
    - Prevents labels from messing up tight_layout by excluding them from layout.

    Returns
    -------
    (fig, ax_or_axes)
      - If only one panel exists: returns (fig, ax)
      - If multiple panels exist: returns (fig, axes_list)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    panels: List[Tuple[np.ndarray, str]] = []

    if getattr(summary, "vert_card", None) is not None and summary.vert_card.size > 0:
        panels.append((np.asarray(summary.vert_card, dtype=float), r"$0$-Simplices: $|U_i|$"))

    if summary.edge_card is not None and summary.edge_card.size > 0:
        panels.append((np.asarray(summary.edge_card, dtype=float), r"$1$-Simplices: $|U_i \cap U_j|$"))

    if summary.tri_card is not None and summary.tri_card.size > 0:
        panels.append((np.asarray(summary.tri_card, dtype=float), r"$2$-Simplices: $|U_i \cap U_j \cap U_k|$"))

    if show_tets and (summary.tet_card is not None) and summary.tet_card.size > 0:
        panels.append((np.asarray(summary.tet_card, dtype=float), r"$3$-Simplices: $|U_i \cap U_j \cap U_k \cap U_\ell|$"))

    if len(panels) == 0:
        return None, None

    n = len(panels)
    if figsize is None:
        figsize = (6.0 * n, 4.5)

    # Using constrained_layout tends to behave better than tight_layout for annotated plots
    fig, axes = plt.subplots(
        1, n,
        figsize=figsize,
        dpi=int(dpi),
        sharey=bool(sharey),
        constrained_layout=True,
    )

    axes_list = [axes] if n == 1 else list(axes)

    for ax, (arr, title) in zip(axes_list, panels):
        # One box per panel
        bp = ax.boxplot(
            [arr],
            labels=[""],
            showfliers=bool(showfliers),
            whis=whis,
        )
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("")

        # --- Grab whisker endpoints from the artists (robust, respects whis) ---
        # bp["whiskers"] has 2 Line2D objects: lower and upper whiskers
        w0, w1 = bp["whiskers"][0], bp["whiskers"][1]

        y0 = w0.get_ydata()
        y1 = w1.get_ydata()
        low = float(np.min(np.r_[y0, y1]))
        high = float(np.max(np.r_[y0, y1]))

        # x position: whiskers are at x=1; place text slightly to the right.
        # We'll offset in *axes-fraction* to be stable across panels.
        x_whisk = 1.0
        x_text = x_whisk + 0.03  # in data coords; OK since x-range is tiny

        # Put labels BESIDE the whisker endpoints (right side), centered vertically on the endpoint
        tmax = ax.text(
            x_text, high,
            f"max = {label_fmt.format(high)}",
            va="center", ha="left",
            fontsize=9,
            clip_on=True,
        )
        tmin = ax.text(
            x_text, low,
            f"min = {label_fmt.format(low)}",
            va="center", ha="left",
            fontsize=9,
            clip_on=True,
        )

        # Critical: don't let these annotations affect layout engines
        # (prevents giant whitespace + "tight layout not applied" warnings)
        try:
            tmax.set_in_layout(False)
            tmin.set_in_layout(False)
        except Exception:
            pass

        # Ensure enough x-room so labels aren't clipped on the right
        ax.set_xlim(0.7, 1.35)

        # Small y padding so text isn't sitting on the border
        ypad = 0.02 * max(1.0, (high - low))
        ax.set_ylim(low - ypad, high + ypad)

    axes_list[0].set_ylabel("Intersection Cardinality (#Samples)")

    if save_path is not None:
        out = save_path
        if out.lower().endswith(".pdf"):
            out = out[:-4] + "_cover_summary.pdf"
        else:
            out = out + "_cover_summary.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")

    plt.show()
    return fig, (axes_list[0] if n == 1 else axes_list)



# ----------------------------
# Base cover API
# ----------------------------

@dataclass
class CoverBase:
    base_points: np.ndarray                 # (n_samples, dB)
    U: Optional[np.ndarray] = None          # (n_sets, n_samples) bool
    pou: Optional[np.ndarray] = None        # (n_sets, n_samples) float
    landmarks: Optional[np.ndarray] = None  # (n_sets, dB) float

    def build(self) -> "CoverBase":
        raise NotImplementedError

    def ensure_built(self) -> "CoverBase":
        if self.U is None or self.pou is None:
            self.build()
        return self

    def nerve_edges(self) -> List[Edge]:
        raise NotImplementedError

    def nerve_triangles(self) -> List[Tri]:
        raise NotImplementedError

    # Optional 3-simplices
    def nerve_tetrahedra(self) -> List[Tet]:
        return []

    # ----------------------------
    # Nerve visualization
    # ----------------------------

    def show_nerve(
        self,
        *,
        title: Optional[str] = None,
        show_labels: bool = True,
        show_axes: bool = False,
        tri_opacity: float = 0.25,
        tri_color: str = "pink",
        cochains: Optional[List[Dict[Tuple[int, ...], object]]] = None,
        edge_weights: Optional[Dict[Edge, float]] = None,
        edge_cutoff: Optional[float] = None,
        highlight_edges: Optional[Set[Edge]] = None,
        highlight_color: str = "red",
    ):
        """
        Show the nerve of the cover.

        - If edge_weights is None: static plot (no slider).
        - Otherwise: slider plot filtering by edge_weights.
        """
        self.ensure_built()
        if self.landmarks is None:
            raise AttributeError("cover.landmarks is missing; cannot visualize nerve.")
        if self.U is None:
            raise AttributeError("cover.U is missing; build the cover first.")

        if edge_weights is None:
            from .viz.nerve_plotly import make_nerve_figure

            fig = make_nerve_figure(
                landmarks=np.asarray(self.landmarks),
                edges=list(self.nerve_edges()),
                triangles=list(self.nerve_triangles()),
                show_labels=show_labels,
                show_axes=show_axes,
                tri_opacity=tri_opacity,
                tri_color=tri_color,
                edge_weights=None,
                edge_cutoff=edge_cutoff,
                highlight_edges=highlight_edges,
                highlight_color=highlight_color,
                cochains=cochains,
                title=title,
            )
            fig.show()
            return fig

        from .viz.nerve_plotly import nerve_with_slider

        return nerve_with_slider(
            cover=self,
            edge_weights=edge_weights,
            show_labels=show_labels,
            tri_opacity=tri_opacity,
            tri_color=tri_color,
            show_axes=show_axes,
        )

    # ----------------------------
    # Summary
    # ----------------------------

    def summarize(
        self,
        *,
        compute_cardinalities: bool = True,
        plot: bool = False,
        plot_kind: str = "box",        # "box" (default). (Percentile could be re-added later if desired.)
        latex: str | bool = "auto",    # "auto" | True | False
        show_tets_plot: bool = True,
        dpi: int = 200,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> NerveSummary:
        """
        Summarize the cover and its recorded nerve (dims 0..3), plus overlap-order evidence from U.

        Printing rules:
          - If max sample overlap order > 4 (i.e. some sample lies in ≥5 sets),
            then print recorded counts for 0..3 explicitly (even if 2/3 are zero),
            and also print overlap info (max order + #samples in ≥5 sets).
          - Otherwise, print counts only up to last nonzero recorded dimension,
            plus a single 'no recorded simplices in dimensions ≥ ...' line.

        Plotting:
          - plot_kind="box": box-and-whisker plot of intersection cardinalities for
            recorded 1/2/3-simplices, omitting any missing dimensions.
        """
        self.ensure_built()
        if self.U is None:
            raise AttributeError("cover.U is missing; build the cover first.")

        U = np.asarray(self.U, dtype=bool)
        n_sets, n_samples = U.shape

        edges = list(self.nerve_edges())
        tris = list(self.nerve_triangles())
        tets = list(self.nerve_tetrahedra())

        n0 = int(n_sets)
        n1 = int(len(edges))
        n2 = int(len(tris))
        n3 = int(len(tets))

        # overlap order per sample (used only for the ≥5 logic + warning)
        sample_overlap = U.sum(axis=0).astype(int)
        max_order = int(sample_overlap.max()) if sample_overlap.size else 0

        max_order_show = max_order if max_order > 4 else None
        n_ge_5_show = int(np.sum(sample_overlap >= 5)) if max_order > 4 else None

        warnings: List[str] = []
        if max_order > 4:
            # If U has ≥5 overlaps, the true nerve has dims ≥4; if we're only recording up to 3,
            # warn about the mismatch explicitly.
            if max_order - 1 > 3:
                warnings.append(
                    f"U contains overlaps of order {max_order} (so the true nerve has simplices up to dimension "
                    f"{max_order - 1}), but this cover only records up to 3-simplices."
                )
            # Also warn if recorded lower-dim counts are unexpectedly zero given the overlap evidence.
            if n2 == 0:
                warnings.append(
                    "U indicates ≥5-way overlaps (hence some triple overlaps must exist), "
                    "but this cover reports 0 recorded 2-simplices."
                )
            if n3 == 0:
                warnings.append(
                    "U indicates ≥5-way overlaps (hence some 4-way overlaps must exist), "
                    "but this cover reports 0 recorded 3-simplices."
                )

        vert_card = edge_card = tri_card = tet_card = None
        if compute_cardinalities:
            vert_card = U.sum(axis=1).astype(int)  
            edge_card = _cards_for_simplices(U, [tuple(e) for e in edges]) if n1 > 0 else np.array([], dtype=int)
            tri_card = _cards_for_simplices(U, [tuple(t) for t in tris]) if n2 > 0 else np.array([], dtype=int)
            tet_card = _cards_for_simplices(U, [tuple(tt) for tt in tets]) if n3 > 0 else np.array([], dtype=int)

        summ = NerveSummary(
            n_sets=int(n_sets),
            n_samples=int(n_samples),
            n0=n0,
            n1=n1,
            n2=n2,
            n3=n3,
            vert_card = vert_card,
            edge_card=edge_card,
            tri_card=tri_card,
            tet_card=tet_card,
            sample_overlap_counts=sample_overlap,
            max_overlap_order=max_order_show,
            n_samples_with_overlap_ge_5=n_ge_5_show,
            warnings=tuple(warnings),
        )

        if verbose:
            want_latex = (latex is True) or (latex == "auto")
            did_display = False
            if want_latex:
                did_display = _try_display_markdown(summ.to_markdown()) if (latex != False) else False
            if not did_display:
                print(summ.to_text())

        if plot:
            if not compute_cardinalities:
                raise ValueError("plot=True requires compute_cardinalities=True.")
            if plot_kind != "box":
                raise ValueError("Currently only plot_kind='box' is supported.")
            plot_cover_summary_boxplot(
                summ,
                show_tets=bool(show_tets_plot),
                dpi=dpi,
                figsize=figsize,
                save_path=save_path,
            )

        return summ


# ----------------------------
# Metric ball cover
# ----------------------------

class MetricBallCover(CoverBase):
    def __init__(
        self,
        base_points: np.ndarray,
        landmarks: np.ndarray,
        radius: float,
        metric: Any = Euc_met,
    ):
        super().__init__(base_points=np.asarray(base_points))
        self.landmarks = np.asarray(landmarks)
        self.radius = float(radius)
        self.metric = metric

    def build(self) -> "MetricBallCover":
        if self.landmarks is None:
            raise ValueError("MetricBallCover requires landmarks.")

        dist = get_dist_mat(self.landmarks, data2=self.base_points, metric=self.metric)  # (n_sets, n_samples)
        self.U = dist < self.radius

        # Linear hat POU, normalized
        w = np.maximum(0.0, 1.0 - dist / self.radius)
        w *= self.U
        denom = w.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        self.pou = w / denom
        return self

    def nerve_edges(self) -> List[Edge]:
        self.ensure_built()
        assert self.U is not None

        n_sets = self.U.shape[0]
        edges: List[Edge] = []
        for j in range(n_sets):
            Uj = self.U[j]
            for k in range(j + 1, n_sets):
                if np.any(Uj & self.U[k]):
                    edges.append(canon_edge(j, k))
        return edges

    def nerve_triangles(self) -> List[Tri]:
        self.ensure_built()
        assert self.U is not None

        n_sets = self.U.shape[0]
        tris: List[Tri] = []
        for i in range(n_sets):
            Ui = self.U[i]
            for j in range(i + 1, n_sets):
                ij = Ui & self.U[j]
                if not np.any(ij):
                    continue
                for k in range(j + 1, n_sets):
                    if np.any(ij & self.U[k]):
                        tris.append(canon_tri(i, j, k))
        return tris

    def nerve_tetrahedra(self) -> List[Tet]:
        """
        3-simplices in the nerve: 4-way overlaps U_i ∩ U_j ∩ U_k ∩ U_l ≠ ∅.
        """
        self.ensure_built()
        assert self.U is not None

        n_sets = self.U.shape[0]
        tets: List[Tet] = []
        for i in range(n_sets):
            Ui = self.U[i]
            for j in range(i + 1, n_sets):
                ij = Ui & self.U[j]
                if not np.any(ij):
                    continue
                for k in range(j + 1, n_sets):
                    ijk = ij & self.U[k]
                    if not np.any(ijk):
                        continue
                    for l in range(k + 1, n_sets):
                        if np.any(ijk & self.U[l]):
                            tets.append(canon_tet(i, j, k, l))
        return tets


# ----------------------------
# Triangulation star cover
# ----------------------------

class TriangulationStarCover(CoverBase):
    """
    Cover by open stars of vertices in a triangulation K.
    Uses K_preimages: the location of each base point in |K| coordinates.

    Expected:
      - K.get_simplices() yields (simplex_tuple, filtration_value) like Gudhi.
      - vertex_coords_dict maps old_vertex_id -> coords (D,)
    """
    def __init__(
        self,
        base_points: np.ndarray,
        K_preimages: np.ndarray,
        K: Any,
        vertex_coords_dict: dict,
    ):
        super().__init__(base_points=np.asarray(base_points))
        self.K_preimages = np.asarray(K_preimages)
        self.K = K
        self.vertex_coords_dict = dict(vertex_coords_dict)

        self.vid_old_to_new: Optional[dict[int, int]] = None
        self.vid_new_to_old: Optional[dict[int, int]] = None
        self.vertex_coords: Optional[np.ndarray] = None  # (nV, D)

        self.triangles: Optional[List[Tri]] = None       # list of (i,j,k) in new ids
        self.tetrahedra: Optional[List[Tet]] = None      # list of (i,j,k,l) in new ids

        self.sample_tri: Optional[np.ndarray] = None     # (n_samples,) triangle index
        self.sample_bary: Optional[np.ndarray] = None    # (n_samples,3)

    def build(self) -> "TriangulationStarCover":
        self._relabel_vertices()
        self._extract_triangles()
        self._extract_tetrahedra()          # extracted, but note nerve_tetrahedra() below
        self._assign_samples_to_triangles()
        self._build_star_sets_U()
        self._build_pou_from_barycentric()
        return self

    def nerve_edges(self) -> List[Edge]:
        self.ensure_built()
        assert self.vid_old_to_new is not None

        edges: set[Edge] = set()
        for s, _ in self.K.get_simplices():
            if len(s) == 2:
                a, b = (self.vid_old_to_new[int(v)] for v in s)
                edges.add(canon_edge(a, b))
        return sorted(edges)

    def nerve_triangles(self) -> List[Tri]:
        self.ensure_built()
        assert self.triangles is not None
        return list(self.triangles)

    def nerve_tetrahedra(self) -> List[Tet]:
        # Explicitly: “this cover currently models a 2-complex”
        # (We still extract tetrahedra from K and store them in self.tetrahedra,
        #  but we do not expose them here unless/when you decide to.)
        return []

    def _relabel_vertices(self) -> None:
        old_vids = sorted(int(v) for v in self.vertex_coords_dict.keys())
        self.vid_old_to_new = {v: i for i, v in enumerate(old_vids)}
        self.vid_new_to_old = {i: v for v, i in self.vid_old_to_new.items()}
        self.vertex_coords = np.stack([self.vertex_coords_dict[v] for v in old_vids], axis=0).astype(float)
        self.landmarks = self.vertex_coords.copy()

    def _extract_triangles(self) -> None:
        assert self.vid_old_to_new is not None
        tris: set[Tri] = set()
        for s, _ in self.K.get_simplices():
            if len(s) == 3:
                a, b, c = (self.vid_old_to_new[int(v)] for v in s)
                tris.add(canon_tri(a, b, c))
        self.triangles = sorted(tris)

    def _extract_tetrahedra(self) -> None:
        """
        Extract 3-simplices from K, if any exist.
        This does NOT change how U/POU are built (still triangle-based).
        It simply stores them for possible future use.
        """
        assert self.vid_old_to_new is not None
        tets: set[Tet] = set()
        for s, _ in self.K.get_simplices():
            if len(s) == 4:
                a, b, c, d = (self.vid_old_to_new[int(v)] for v in s)
                tets.add(canon_tet(a, b, c, d))
        self.tetrahedra = sorted(tets)

    def _assign_samples_to_triangles(self, tol: float = 1e-8) -> None:
        assert self.vertex_coords is not None
        assert self.triangles is not None

        P = self.K_preimages
        n = P.shape[0]
        self.sample_tri = -np.ones(n, dtype=int)
        self.sample_bary = np.zeros((n, 3), dtype=float)

        for t_idx, (i, j, k) in enumerate(self.triangles):
            V = self.vertex_coords[[i, j, k]]
            bary = get_bary_coords(P, V)
            inside = points_in_triangle_mask(bary, tol=tol)
            newly = inside & (self.sample_tri == -1)
            if np.any(newly):
                self.sample_tri[newly] = t_idx
                self.sample_bary[newly] = bary[newly]

        if np.any(self.sample_tri == -1):
            raise ValueError("Some samples were not assigned to any triangle (bad preimage / tol too strict).")

    def _build_star_sets_U(self) -> None:
        assert self.vertex_coords is not None
        assert self.triangles is not None
        assert self.sample_tri is not None

        nV = self.vertex_coords.shape[0]
        nS = self.base_points.shape[0]
        U = np.zeros((nV, nS), dtype=bool)

        for s in range(nS):
            i, j, k = self.triangles[self.sample_tri[s]]
            U[i, s] = True
            U[j, s] = True
            U[k, s] = True

        self.U = U

    def _build_pou_from_barycentric(self) -> None:
        assert self.U is not None
        assert self.triangles is not None
        assert self.sample_tri is not None
        assert self.sample_bary is not None

        nV, nS = self.U.shape
        pou = np.zeros((nV, nS), dtype=float)

        for s in range(nS):
            i, j, k = self.triangles[self.sample_tri[s]]
            u, v, w = self.sample_bary[s]
            pou[i, s] = u
            pou[j, s] = v
            pou[k, s] = w

        pou *= self.U
        denom = pou.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        self.pou = pou / denom
