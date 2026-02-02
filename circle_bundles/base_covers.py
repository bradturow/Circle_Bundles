# base_covers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .metrics import EuclideanMetric, as_metric

from .geometry.geometry import get_bary_coords, points_in_triangle_mask
from .nerve.combinatorics import Edge, Tri, canon_edge, canon_tri


Tet = Tuple[int, int, int, int]


def canon_tet(a: int, b: int, c: int, d: int) -> Tet:
    return tuple(sorted((int(a), int(b), int(c), int(d))))

def _as_2d_points(X: np.ndarray, *, name: str = "points") -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim == 2:
        return X
    raise ValueError(f"{name} must be 1D or 2D. Got shape {X.shape}.")


# ----------------------------
# Nerve summary data container
# ----------------------------

@dataclass
class NerveSummary:
    """
    Summary of a cover and the recorded nerve up to dimension 3.

    This container is produced by :meth:`CoverBase.summarize` and is intended to be
    lightweight: it stores basic counts, optional intersection cardinalities, and
    optional evidence about higher-order overlaps detected directly from the cover
    membership matrix ``U``.

    Notes
    -----
    - The library may only *record* simplices up to dimension 3 (tetrahedra), but
      the underlying cover matrix ``U`` can still contain higher-order overlaps.
      When detected (sample overlap order ≥ 5), the summary exposes:
        - ``max_overlap_order`` and
        - ``n_samples_with_overlap_ge_5``
      and typically includes warnings about the mismatch.

    Attributes
    ----------
    n_sets :
        Number of cover sets.
    n_samples :
        Number of samples/points covered.

    n0, n1, n2, n3 :
        Recorded simplex counts in dimensions 0..3.

    vert_card, edge_card, tri_card, tet_card :
        Optional intersection cardinalities for recorded simplices, i.e.
        ``|⋂_{i in σ} U_i|`` for each recorded simplex ``σ``.
        Shapes are ``(n0,)``, ``(n1,)``, ``(n2,)``, ``(n3,)`` respectively
        when present.

    sample_overlap_counts :
        For each sample ``s``, the number of cover sets containing it, i.e.
        ``(# of i with U[i,s] = True)``. Shape ``(n_samples,)``.

    max_overlap_order :
        Only populated when the maximum sample overlap order exceeds 4.
        In that case this is ``max(sample_overlap_counts)``.
    n_samples_with_overlap_ge_5 :
        Only populated when ``max_overlap_order`` is set. Counts samples lying
        in 5 or more cover sets.

    warnings :
        Tuple of human-readable warning strings surfaced by :meth:`summarize`.
    """
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
            md.append(f"- $\\max_s \\sum_i U_{{i,s}} = {self.max_overlap_order}$")
            md.append(f"- $\\#\\{{s : \\sum_i U_{{i,s}} \\ge 5\\}} = {self.n_samples_with_overlap_ge_5}$")

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
    whis=(0, 100),
    sharey: bool = False,
    label_fmt: str = "{:g}",
    label_dx: float = 0.06,
):
    """
    Plot intersection cardinalities for recorded nerve simplices.

    Produces a 1×k grid of box-and-whisker plots for the cardinalities:

    - 0-simplices: ``|U_i|`` (when available)
    - 1-simplices: ``|U_i ∩ U_j|``
    - 2-simplices: ``|U_i ∩ U_j ∩ U_k|``
    - 3-simplices: ``|U_i ∩ U_j ∩ U_k ∩ U_ℓ|`` (optional)

    By default, whiskers use the full range ``whis=(0,100)`` so min/max are the
    actual extrema (not outlier-based). The plot annotates min/max next to the
    whiskers.

    Parameters
    ----------
    summary :
        A :class:`NerveSummary` produced by :meth:`CoverBase.summarize` with
        cardinalities computed.
    show_tets :
        If True, include the 3-simplex panel when ``summary.tet_card`` is present.
    dpi :
        Figure DPI.
    figsize :
        Optional figure size. If omitted, a width proportional to the number of
        panels is chosen.
    save_path :
        Optional output prefix. If provided, saves a PDF named
        ``<save_path>_cover_summary.pdf``.
    showfliers :
        Whether to show fliers in the boxplot.
    whis :
        Whisker definition passed to ``matplotlib.axes.Axes.boxplot``.
        Default ``(0,100)`` gives true min/max.
    sharey :
        Share the y-axis across panels.
    label_fmt :
        Format string used for min/max labels.
    label_dx :
        Horizontal label offset (currently expressed as a small shift in x data units).

    Returns
    -------
    fig, ax_or_axes :
        ``(fig, ax)`` if only one panel is drawn, otherwise ``(fig, axes_list)``.
        Returns ``(None, None)`` if no panels are available.

    Notes
    -----
    This function is designed for quick diagnostics. For publication-style plots,
    you may want custom axis limits, log scaling, or annotated quantiles.
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

        # Put labels beside the whisker endpoints, centered vertically on the endpoint
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
    """
    Base class for covers over a set of base points.

    A cover is represented primarily by a boolean membership matrix ``U`` with shape
    ``(n_sets, n_samples)``, where ``U[i,s]`` indicates whether sample ``s`` lies in
    cover set ``U_i``. Most workflows also use a partition of unity ``pou`` of the
    same shape.

    Subclasses implement :meth:`build` to populate ``U`` and ``pou``, and implement
    the nerve accessors (:meth:`nerve_edges`, :meth:`nerve_triangles`, optionally
    :meth:`nerve_tetrahedra`) to report recorded simplices.

    Parameters
    ----------
    base_points :
        Array of shape ``(n_samples, dB)`` giving base coordinates for each sample.
        One-dimensional inputs are reshaped to ``(n_samples, 1)``.
    U :
        Optional boolean membership matrix of shape ``(n_sets, n_samples)``.
        If not provided, subclasses compute it in :meth:`build`.
    pou :
        Optional partition of unity weights of shape ``(n_sets, n_samples)``.
        Typically satisfies ``pou[:,s].sum() == 1`` for each sample ``s`` (when covered).
    landmarks :
        Optional landmark coordinates of shape ``(n_sets, dB)`` used for visualization
        and for some cover constructions (e.g. metric ball covers).
    metric :
        Distance model used by the cover (for constructions that require it).
        May be a Metric object with ``pairwise`` or a scalar callable; it is normalized
        by :meth:`ensure_metric`.
    full_dist_mat :
        Optional cached sample-sample distance matrix for visualization.

    Attributes
    ----------
    base_name, base_name_latex :
        Optional base labels used in plots and summaries. If not set explicitly,
        :meth:`ensure_metric` will inherit these from the metric object when available.

    Notes
    -----
    - Most downstream algorithms expect ``U`` and ``pou`` to be populated.
      Use :meth:`ensure_built` to build lazily.
    - The “recorded nerve” is whatever simplices the cover reports via
      ``nerve_edges/nerve_triangles/...``. It may be truncated even if ``U``
      contains higher-order overlaps.
    """
    base_points: np.ndarray                 # (n_samples, dB)
    U: Optional[np.ndarray] = None          # (n_sets, n_samples) bool
    pou: Optional[np.ndarray] = None        # (n_sets, n_samples) float
    landmarks: Optional[np.ndarray] = None  # (n_sets, dB) float
    metric: Any = None                 # should be a Metric object (has .pairwise)
    full_dist_mat: Optional[np.ndarray] = None  # optional cache for viz

# --- optional metadata for summaries/plots ---
    base_name: Optional[str] = None
    base_name_latex: Optional[str] = None        
        
        
    def __post_init__(self):
        self.base_points = _as_2d_points(self.base_points, name="cover.base_points")
        if self.landmarks is not None:
            self.landmarks = _as_2d_points(self.landmarks, name="cover.landmarks")        
    
    def normalize_shapes(self) -> None:
        self.base_points = _as_2d_points(self.base_points, name="cover.base_points")
        if self.landmarks is not None:
            self.landmarks = _as_2d_points(self.landmarks, name="cover.landmarks")

    def ensure_metric(self):
        """
        Normalize self.metric into a vectorized Metric object with .pairwise.
        Defaults to EuclideanMetric if missing.

        Also: if the metric carries base_name/base_name_latex and the cover does not
        already have explicit base labels, inherit them.
        """
        if self.metric is None:
            self.metric = EuclideanMetric()
        else:
            self.metric = as_metric(self.metric)

        # ---- inherit base labels from metric if user didn't set cover labels ----
        if getattr(self, "base_name", None) in (None, ""):
            bn = getattr(self.metric, "base_name", None)
            if isinstance(bn, str) and bn.strip():
                self.base_name = bn.strip()

        if getattr(self, "base_name_latex", None) in (None, ""):
            bL = getattr(self.metric, "base_name_latex", None)
            if isinstance(bL, str) and bL.strip():
                self.base_name_latex = bL.strip()

        return self.metric
                
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
        Visualize the cover nerve as an interactive Plotly figure.

        This method requires the cover to have landmark coordinates (one per cover set)
        and a built membership matrix ``U``. It plots:

        - vertices at ``self.landmarks`` (one point per cover set),
        - edges from :meth:`nerve_edges`,
        - optional filled triangles from :meth:`nerve_triangles`.

        Two modes are supported:

        **Static mode (no slider)**
            If ``edge_weights`` is None, a single figure is produced. Optionally, you may:
            - show triangles with configurable opacity/color,
            - highlight specific edges,
            - overlay cochains.

        **Slider mode (edge filtering)**
            If ``edge_weights`` is provided, the visualization includes a slider that filters
            edges by weight threshold (and displays the induced subcomplex). This delegates to
            :func:`circle_bundles.viz.nerve_plotly.nerve_with_slider`.

        Parameters
        ----------
        title :
            Optional plot title.
        show_labels :
            Whether to show vertex labels (set indices) in the plot.
        show_axes :
            Whether to show 3D axes in the Plotly scene (useful for debugging).
        tri_opacity :
            Opacity for triangle faces (ignored if no triangles are drawn).
        tri_color :
            Color used for triangle faces (Plotly color string).
        cochains :
            Optional list of cochain dictionaries to overlay on the nerve.
            Each cochain is a mapping from a simplex (tuple of vertex indices) to a value
            used by the plotting code (e.g. for coloring). The exact interpretation depends
            on the nerve plotting utilities.
        edge_weights :
            Optional mapping from edges to weights. If provided, enables slider mode.
            The keys must be canonical edges as returned by :func:`canon_edge` /
            :meth:`nerve_edges`.
        edge_cutoff :
            Optional edge cutoff for static plots (may be ignored in slider mode,
            depending on the plotting utility).
        highlight_edges :
            Optional set of edges to emphasize in a separate style layer.
        highlight_color :
            Color used for highlighted edges.

        Returns
        -------
        fig :
            The Plotly figure object.

        Raises
        ------
        AttributeError
            If ``landmarks`` or ``U`` is missing (i.e., the cover cannot be visualized).
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
        Summarize the cover and its recorded nerve (dimensions 0..3), with optional plots.

        This function builds (if needed) the cover membership matrix ``U`` and computes:

        - number of sets and samples,
        - recorded simplex counts:
          ``#0-simplices = n_sets``, and counts of recorded edges/triangles/tetrahedra
          as returned by :meth:`nerve_edges`, :meth:`nerve_triangles`, :meth:`nerve_tetrahedra`,
        - overlap-order evidence from ``U``:
          for each sample, how many cover sets contain it (the “overlap order”).

        A key consistency check is whether the data show overlaps beyond what the cover
        explicitly records. For example, if some sample lies in ≥5 sets, then the *true*
        nerve necessarily contains simplices of dimension ≥4. Since this library’s covers
        typically record simplices only up to dimension 3, we surface this mismatch as a warning.

        Parameters
        ----------
        compute_cardinalities :
            If True, also compute intersection cardinalities for recorded simplices:
            ``|U_i|``, ``|U_i ∩ U_j|``, ``|U_i ∩ U_j ∩ U_k|``, and optionally 4-way intersections.
            These populate the ``*_card`` fields of the returned :class:`NerveSummary`.
        plot :
            If True, display a plot summarizing the intersection cardinalities.
            Requires ``compute_cardinalities=True``.
        plot_kind :
            Currently only ``"box"`` is supported: box-and-whisker plots of recorded
            simplex intersection cardinalities, omitting missing dimensions.
        latex :
            Controls printing style when ``verbose=True``:
            - ``"auto"`` (default): attempt rich Markdown display in notebooks, else plain text
            - True: force Markdown attempt
            - False: force plain text printing
        show_tets_plot :
            If True and tetrahedra cardinalities exist, include them in the plot panel list.
        dpi, figsize :
            Plot formatting controls.
        save_path :
            If provided and ``plot=True``, save a PDF of the plot using this prefix/path.
            The function appends ``"_cover_summary.pdf"``.
        verbose :
            If True, print/display the summary.

        Returns
        -------
        summary :
            A :class:`NerveSummary` object containing counts, optional cardinalities,
            overlap evidence, and any warnings.

        Raises
        ------
        AttributeError
            If the cover has no membership matrix ``U`` after building (should not happen
            for correct cover implementations).
        ValueError
            If ``plot=True`` but ``compute_cardinalities=False``, or if an unsupported
            ``plot_kind`` is requested.

        Notes
        -----
        **Printing / warning policy**
        - If the maximum sample overlap order is > 4 (i.e. some sample lies in ≥5 sets),
          the summary prints all recorded simplex counts for dimensions 0..3 and includes
          overlap-order lines (max order and number of samples in ≥5 sets).
        - Otherwise, it prints counts only up to the last nonzero recorded dimension and
          adds a single “no recorded simplices in dimensions ≥ …” line.

        **Interpretation**
        - Overlap order is computed directly from ``U`` and reflects *evidence in the data*.
        - Simplex counts are the *recorded* nerve simplices returned by the cover methods.
          If those counts disagree with the overlap evidence, warnings highlight the mismatch.
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
    """
    Cover by metric balls centered at landmark points.

    Given base points ``x_s`` and landmark centers ``ℓ_i``, this cover defines:

    - membership: ``U[i,s] = (d(ℓ_i, x_s) < radius)``
    - partition of unity: a linear “hat” weight
      ``w[i,s] = max(0, 1 - d(ℓ_i, x_s) / radius)``, normalized across sets for each sample.

    This is a simple, widely useful cover construction for point clouds in Euclidean
    space or in any space equipped with a vectorized metric.

    Parameters
    ----------
    base_points :
        Array of shape ``(n_samples, dB)`` giving base coordinates.
    landmarks :
        Array of shape ``(n_sets, dB)`` giving the cover centers.
    radius :
        Ball radius (in the units of the chosen metric).
    metric :
        Distance model. May be:
        - a Metric-like object with ``pairwise(X, Y=None)``, or
        - a scalar callable ``metric(p, q)`` (requires SciPy for vectorization fallback), or
        - None (defaults to :class:`~circle_bundles.metrics.EuclideanMetric`).

    Notes
    -----
    - The cover sets are *open* balls (strict inequality).
    - The recorded nerve is computed directly from ``U`` by checking whether
      intersections are nonempty (for edges, triangles, and tetrahedra).
    - If a sample lies in no sets (possible if radius is too small), the current
      POU normalization leaves that sample with all-zero weights.

    See Also
    --------
    TriangulationStarCover :
        Cover induced by open stars in a triangulation.
    """

    def __init__(
        self,
        base_points: np.ndarray,
        landmarks: np.ndarray,
        radius: float,
        metric: Any = None,
    ):
        super().__init__(base_points=np.asarray(base_points))
        self.landmarks = np.asarray(landmarks)
        self.radius = float(radius)
        self.metric = metric
        self.ensure_metric()
        self.normalize_shapes()

    def build(self) -> "MetricBallCover":
        """
        Build the cover membership matrix ``U`` and partition of unity ``pou``.

        Populates
        ---------
        U :
            Boolean array of shape ``(n_sets, n_samples)`` where ``U[i,s]`` indicates
            whether sample ``s`` lies in the radius-ball around landmark ``i``.
        pou :
            Float array of shape ``(n_sets, n_samples)`` giving normalized linear
            hat weights over the sets containing each sample.

        Returns
        -------
        self :
            The built cover (for chaining).
        """
        if self.landmarks is None:
            raise ValueError("MetricBallCover requires landmarks.")

        M = self.ensure_metric()
        dist = M.pairwise(np.asarray(self.landmarks), np.asarray(self.base_points))  # (n_sets, n_samples)

        self.U = dist < self.radius

        # Linear hat POU, normalized
        w = np.maximum(0.0, 1.0 - dist / self.radius)
        w *= self.U
        denom = w.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        self.pou = w / denom
        return self

    def nerve_edges(self) -> List[Edge]:
        """
        Recorded 1-simplices of the nerve.

        An edge (i,j) is included when the intersection ``U_i ∩ U_j`` contains
        at least one sample.

        Returns
        -------
        edges :
            List of canonical edges ``(i,j)`` with ``i < j``.
        """
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
        """
        Recorded 2-simplices of the nerve.

        A triangle (i,j,k) is included when the triple intersection
        ``U_i ∩ U_j ∩ U_k`` is nonempty.

        Returns
        -------
        triangles :
            List of canonical triangles ``(i,j,k)`` with ``i < j < k``.
        """
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
        Recorded 3-simplices of the nerve.

        A tetrahedron (i,j,k,ℓ) is included when the 4-way intersection
        ``U_i ∩ U_j ∩ U_k ∩ U_ℓ`` is nonempty.

        Returns
        -------
        tetrahedra :
            List of canonical 4-tuples ``(i,j,k,ℓ)`` with ``i < j < k < ℓ``.
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
    Cover by open stars of vertices in a triangulation.

    This cover is defined from a simplicial complex ``K`` that triangulates a base
    space (typically 2D for this implementation), together with “preimages”
    ``K_preimages`` giving each sample’s location in the geometric realization ``|K|``.

    Each cover set corresponds to a vertex ``v`` of ``K`` and contains samples whose
    assigned triangle includes that vertex (i.e. samples lying in triangles incident
    to ``v``). The partition of unity is defined using barycentric coordinates within
    the assigned triangle: the three vertices of the triangle receive the barycentric
    weights, and all other vertices receive weight 0.

    Parameters
    ----------
    base_points :
        Array of shape ``(n_samples, dB)``. Kept for consistency with other covers.
        (This cover uses ``K_preimages`` for assignment/POU.)
    K_preimages :
        Array of shape ``(n_samples, D)`` giving each sample’s position in the
        coordinate system used by ``vertex_coords_dict`` (i.e. coordinates for
        the geometric realization of ``K``).
    K :
        Simplicial complex object with a Gudhi-like interface:
        ``K.get_simplices()`` yields ``(simplex_tuple, filtration_value)``.
        This implementation expects at least 1- and 2-simplices.
    vertex_coords_dict :
        Mapping ``old_vertex_id -> coords`` where ``coords`` has shape ``(D,)``.
        Vertex ids need not be contiguous; this cover relabels them to ``0..nV-1``.
    metric :
        Optional metric (not used for construction here), but accepted for API
        uniformity and for downstream plotting/labels.

    Attributes
    ----------
    landmarks :
        Set to the relabeled vertex coordinates of shape ``(nV, D)`` (after build).
        These are used for nerve visualization.

    Notes
    -----
    - This implementation builds ``U`` and ``pou`` from *triangle membership*.
      It is effectively a 2-complex cover: :meth:`nerve_tetrahedra` is intentionally
      not exposed (returns ``[]``), even if ``K`` contains 3-simplices.
    - Every sample must be assigned to some triangle. If not, :meth:`build` raises
      an error, typically indicating inconsistent preimages or a too-strict tolerance.

    See Also
    --------
    MetricBallCover :
        A cover built from metric balls around landmark points.
    """

    def __init__(
        self,
        base_points: np.ndarray,
        K_preimages: np.ndarray,
        K: Any,
        vertex_coords_dict: dict,
        metric: Any = None,
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
        """
        Build the star cover membership matrix ``U`` and barycentric POU ``pou``.

        This method:
        1) relabels vertices to contiguous ids and sets ``landmarks``,
        2) extracts triangles from ``K``,
        3) assigns each sample to a triangle using barycentric coordinates,
        4) builds ``U`` by marking membership in the three incident vertex stars,
        5) builds ``pou`` from barycentric weights.

        Returns
        -------
        self :
            The built cover (for chaining).

        Raises
        ------
        ValueError
            If any sample cannot be assigned to a triangle (e.g. inconsistent
            coordinates or tolerance too strict).
        """
        self._relabel_vertices()
        self._extract_triangles()
        self._extract_tetrahedra()          # extracted but not exposed via the nerve API
        self._assign_samples_to_triangles()
        self._build_star_sets_U()
        self._build_pou_from_barycentric()
        return self

    def nerve_edges(self) -> List[Edge]:
        """
        Recorded 1-simplices of the nerve (edges of the triangulation).

        Returns
        -------
        edges :
            Sorted list of canonical edges in the relabeled vertex ids.
        """
        self.ensure_built()
        assert self.vid_old_to_new is not None

        edges: set[Edge] = set()
        for s, _ in self.K.get_simplices():
            if len(s) == 2:
                a, b = (self.vid_old_to_new[int(v)] for v in s)
                edges.add(canon_edge(a, b))
        return sorted(edges)

    def nerve_triangles(self) -> List[Tri]:
        """
        Recorded 2-simplices of the nerve (triangles of the triangulation).

        Returns
        -------
        triangles :
            List of canonical triangles in relabeled vertex ids.
        """
        self.ensure_built()
        assert self.triangles is not None
        return list(self.triangles)

    def nerve_tetrahedra(self) -> List[Tet]:
        """
        Recorded 3-simplices of the nerve.

        This cover currently models a 2-complex (triangle-based stars), so we do not
        expose tetrahedra even if the underlying complex ``K`` contains them.

        Returns
        -------
        tetrahedra :
            Always returns ``[]``.
        """
        return []

    def _relabel_vertices(self) -> None:
        """
        Relabel vertex ids to contiguous integers and populate landmark coordinates.

        Populates
        ---------
        vid_old_to_new, vid_new_to_old :
            Dictionaries mapping between original ids and contiguous ids.
        vertex_coords :
            Array of shape ``(nV, D)`` of vertex coordinates.
        landmarks :
            Same as ``vertex_coords`` (for nerve visualization).
        """
        old_vids = sorted(int(v) for v in self.vertex_coords_dict.keys())
        self.vid_old_to_new = {v: i for i, v in enumerate(old_vids)}
        self.vid_new_to_old = {i: v for v, i in self.vid_old_to_new.items()}
        self.vertex_coords = np.stack([self.vertex_coords_dict[v] for v in old_vids], axis=0).astype(float)
        self.landmarks = self.vertex_coords.copy()

    def _extract_triangles(self) -> None:
        """
        Extract 2-simplices from ``K`` and store them in canonical form.
        """
        assert self.vid_old_to_new is not None
        tris: set[Tri] = set()
        for s, _ in self.K.get_simplices():
            if len(s) == 3:
                a, b, c = (self.vid_old_to_new[int(v)] for v in s)
                tris.add(canon_tri(a, b, c))
        self.triangles = sorted(tris)

    def _extract_tetrahedra(self) -> None:
        """
        Extract 3-simplices from ``K`` (if present) and store them.

        Note
        ----
        This does not change how ``U`` or ``pou`` are constructed (still triangle-based).
        The tetrahedra are stored only for possible future use.
        """
        assert self.vid_old_to_new is not None
        tets: set[Tet] = set()
        for s, _ in self.K.get_simplices():
            if len(s) == 4:
                a, b, c, d = (self.vid_old_to_new[int(v)] for v in s)
                tets.add(canon_tet(a, b, c, d))
        self.tetrahedra = sorted(tets)

    def _assign_samples_to_triangles(self, tol: float = 1e-8) -> None:
        """
        Assign each sample to a triangle by barycentric containment.

        Parameters
        ----------
        tol :
            Numerical tolerance used when deciding whether barycentric coordinates
            are inside the triangle.

        Populates
        ---------
        sample_tri :
            Integer array of shape ``(n_samples,)`` giving the assigned triangle index.
        sample_bary :
            Float array of shape ``(n_samples, 3)`` giving barycentric coordinates
            relative to the assigned triangle.

        Raises
        ------
        ValueError
            If any sample cannot be assigned to any triangle.
        """
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
        """
        Build membership matrix ``U`` from the assigned triangle per sample.

        Each sample belongs to exactly three vertex stars: the vertices of its assigned triangle.
        """
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
        """
        Build partition of unity ``pou`` from stored barycentric coordinates.

        For each sample, the three incident vertices receive weights equal to the
        barycentric coordinates within the assigned triangle.
        """
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
