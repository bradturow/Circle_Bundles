# circle_bundles/summaries/nerve_summary.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ..nerve.combinatorics import Edge, Tri, canon_edge, canon_tri

Tet = tuple[int, int, int, int]


def canon_tet(a: int, b: int, c: int, d: int) -> Tet:
    return tuple(sorted((int(a), int(b), int(c), int(d))))  # type: ignore[return-value]


# ----------------------------
# Summary data container
# ----------------------------

@dataclass
class NerveSummary:
    """
    Pretty summary of a cover nerve (recorded or computed from U) up to dimension 3,
    plus overlap evidence from U.

    Optionally includes intersection cardinalities (#samples in ⋂ U_i over each simplex),
    enabling the same box-and-whisker diagnostic plots you had before.
    """
    n_sets: int
    n_samples: int

    n0: int
    n1: int
    n2: int
    n3: int

    # Optional intersection cardinalities (#samples in ⋂ U_i over each simplex)
    vert_card: Optional[np.ndarray] = None  # (n0,)
    edge_card: Optional[np.ndarray] = None  # (n1,)
    tri_card: Optional[np.ndarray] = None   # (n2,)
    tet_card: Optional[np.ndarray] = None   # (n3,)

    sample_overlap_counts: Optional[np.ndarray] = None
    max_overlap_order: Optional[int] = None
    n_samples_with_overlap_ge_5: Optional[int] = None
    warnings: Tuple[str, ...] = ()

    # ----------------------------
    # formatting
    # ----------------------------

    def to_text(self) -> str:
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
        counts = {0: int(self.n0), 1: int(self.n1), 2: int(self.n2), 3: int(self.n3)}
        md: List[str] = []
        md.append("### Cover And Nerve Summary")
        md.append(f"- $n_\\text{{sets}} = {self.n_sets}$, $n_\\text{{samples}} = {self.n_samples}$")

        if (self.max_overlap_order is not None) and (self.max_overlap_order > 4):
            md.append("")
            md.append("**Recorded Simplex Counts:**")
            md.append("")
            md.append("\n".join([f"- $\\#(\\text{{{d}-simplices}}) = {counts[d]}$" for d in (0, 1, 2, 3)]))
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
            md.append("\n".join([f"- $\\#(\\text{{{d}-simplices}}) = {counts[d]}$" for d in range(0, last_nonzero + 1)]))
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
    # rendering
    # ----------------------------

    def _has_cardinalities(self) -> bool:
        for arr in (self.vert_card, self.edge_card, self.tri_card, self.tet_card):
            if arr is not None and np.asarray(arr).size > 0:
                return True
        return False

    def show_summary(
        self,
        *,
        show: bool = True,
        mode: str = "auto",
        # NEW: bring back old behavior: show plots when they exist
        plot: str | bool = "auto",
        show_tets_plot: bool = True,
        dpi: int = 200,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Display the summary.

        - `mode` controls text/markdown display ("auto", "latex", "text", "both").
        - `plot` controls whether to display the box/whisker diagnostic plot:
            * "auto" (default): plot if cardinalities are present
            * True: always try to plot (requires cardinalities)
            * False: never plot
        """
        text = self.to_text()
        if not show:
            return text

        did_rich = False
        if mode in {"latex", "auto", "both"}:
            did_rich = _display_markdown(self)

        if mode == "both" or mode == "text" or (mode == "auto" and not did_rich):
            print("\n" + text + "\n")

        do_plot: bool
        if plot == "auto":
            do_plot = self._has_cardinalities()
        else:
            do_plot = bool(plot)

        if do_plot:
            # If the user asked for plotting but we have no cardinalities, fail loudly.
            if not self._has_cardinalities():
                raise ValueError("NerveSummary.show_summary(plot=True) requires cardinalities to be present.")
            self.plot_boxplot(
                show_tets=bool(show_tets_plot),
                dpi=int(dpi),
                figsize=figsize,
                save_path=save_path,
                show=True,
            )

        return text

    def plot_boxplot(
        self,
        *,
        show_tets: bool = True,
        dpi: int = 200,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        showfliers: bool = False,
        whis=(0, 100),
        sharey: bool = False,
        label_fmt: str = "{:g}",
        show: bool = True,  # NEW
    ):
        return plot_nerve_summary_boxplot(
            self,
            show_tets=show_tets,
            dpi=dpi,
            figsize=figsize,
            save_path=save_path,
            showfliers=showfliers,
            whis=whis,
            sharey=sharey,
            label_fmt=label_fmt,
            show=show,
        )


def _display_markdown(summary: NerveSummary) -> bool:
    try:
        from IPython.display import display, Markdown  # type: ignore
    except Exception:
        return False
    try:
        display(Markdown(summary.to_markdown()))
        return True
    except Exception:
        return False


# ----------------------------
# Helpers: derive simplices from U
# ----------------------------

def _edges_from_U(U: np.ndarray, *, min_points: int) -> list[Edge]:
    n_sets, _ = U.shape
    out: list[Edge] = []
    for j in range(n_sets):
        Uj = U[j]
        for k in range(j + 1, n_sets):
            if int(np.sum(Uj & U[k])) >= min_points:
                out.append(canon_edge(j, k))
    return out


def _triangles_from_U(U: np.ndarray, *, min_points: int) -> list[Tri]:
    n_sets, _ = U.shape
    out: list[Tri] = []
    for i in range(n_sets):
        Ui = U[i]
        for j in range(i + 1, n_sets):
            ij = Ui & U[j]
            if int(np.sum(ij)) < min_points:
                continue
            for k in range(j + 1, n_sets):
                if int(np.sum(ij & U[k])) >= min_points:
                    out.append(canon_tri(i, j, k))
    return out


def _tets_from_U(U: np.ndarray, *, min_points: int) -> list[Tet]:
    n_sets, _ = U.shape
    out: list[Tet] = []
    for i in range(n_sets):
        Ui = U[i]
        for j in range(i + 1, n_sets):
            ij = Ui & U[j]
            if int(np.sum(ij)) < min_points:
                continue
            for k in range(j + 1, n_sets):
                ijk = ij & U[k]
                if int(np.sum(ijk)) < min_points:
                    continue
                for l in range(k + 1, n_sets):
                    if int(np.sum(ijk & U[l])) >= min_points:
                        out.append(canon_tet(i, j, k, l))
    return out


def _simplex_intersection_cardinality(U: np.ndarray, sig: tuple[int, ...]) -> int:
    if len(sig) == 0:
        return int(U.shape[1])
    mask = np.logical_and.reduce(U[list(sig)], axis=0)
    return int(mask.sum())


def _cards_for_simplices(U: np.ndarray, simplices: List[tuple[int, ...]]) -> np.ndarray:
    out = np.empty(len(simplices), dtype=int)
    for idx, sig in enumerate(simplices):
        out[idx] = _simplex_intersection_cardinality(U, tuple(sig))
    return out


# ----------------------------
# Plot helper (box-and-whisker)
# ----------------------------

def plot_nerve_summary_boxplot(
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
    show: bool = True,  # NEW
):
    import matplotlib.pyplot as plt

    panels: List[Tuple[np.ndarray, str]] = []

    if summary.vert_card is not None and summary.vert_card.size > 0:
        panels.append((np.asarray(summary.vert_card, dtype=float), r"$0$-Simplices: $|U_i|$"))
    if summary.edge_card is not None and summary.edge_card.size > 0:
        panels.append((np.asarray(summary.edge_card, dtype=float), r"$1$-Simplices: $|U_i \cap U_j|$"))
    if summary.tri_card is not None and summary.tri_card.size > 0:
        panels.append((np.asarray(summary.tri_card, dtype=float), r"$2$-Simplices: $|U_i \cap U_j \cap U_k|$"))
    if show_tets and summary.tet_card is not None and summary.tet_card.size > 0:
        panels.append((np.asarray(summary.tet_card, dtype=float), r"$3$-Simplices: $|U_i \cap U_j \cap U_k \cap U_\ell|$"))

    if len(panels) == 0:
        return None, None

    n = len(panels)
    if figsize is None:
        figsize = (6.0 * n, 4.5)

    fig, axes = plt.subplots(
        1, n,
        figsize=figsize,
        dpi=int(dpi),
        sharey=bool(sharey),
        constrained_layout=True,
    )
    axes_list = [axes] if n == 1 else list(axes)

    for ax, (arr, title) in zip(axes_list, panels):
        bp = ax.boxplot([arr], labels=[""], showfliers=bool(showfliers), whis=whis)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("")

        # whisker endpoints (respects whis)
        w0, w1 = bp["whiskers"][0], bp["whiskers"][1]
        low = float(np.min(np.r_[w0.get_ydata(), w1.get_ydata()]))
        high = float(np.max(np.r_[w0.get_ydata(), w1.get_ydata()]))

        x_text = 1.03
        tmax = ax.text(x_text, high, f"max = {label_fmt.format(high)}", va="center", ha="left", fontsize=9, clip_on=True)
        tmin = ax.text(x_text, low,  f"min = {label_fmt.format(low)}",  va="center", ha="left", fontsize=9, clip_on=True)
        try:
            tmax.set_in_layout(False)
            tmin.set_in_layout(False)
        except Exception:
            pass

        ax.set_xlim(0.7, 1.35)
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

    if show:
        plt.show()

    return fig, (axes_list[0] if n == 1 else axes_list)


# ----------------------------
# Public API: summarize from U (+ optional recorded nerve)
# ----------------------------

def summarize_nerve_from_U(
    U: np.ndarray,
    *,
    edges: Optional[Iterable[Edge]] = None,
    tris: Optional[Iterable[Tri]] = None,
    tets: Optional[Iterable[Tet]] = None,
    max_simplex_dim: int = 3,
    min_points_simplex: int = 1,
    # NEW:
    force_compute_from_U: bool = False,
    compute_cardinalities: bool = True,
    plot: bool = False,
    show_tets_plot: bool = True,
    dpi: int = 200,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    # Printing
    verbose: bool = False,
    latex: str | bool = "auto",
) -> NerveSummary:
    U = np.asarray(U, dtype=bool)
    if U.ndim != 2:
        raise ValueError(f"U must be 2D. Got {U.shape}.")

    n_sets, n_samples = U.shape
    sample_overlap = U.sum(axis=0).astype(int)
    max_order = int(sample_overlap.max()) if sample_overlap.size else 0

    mp = int(min_points_simplex)
    max_dim_reported = int(max_simplex_dim)

    # Decide whether to use "recorded nerve" or compute from U
    use_recorded = (not force_compute_from_U) and (edges is not None or tris is not None or tets is not None)

    if use_recorded:
        E = list(edges) if edges is not None else []
        T = list(tris) if tris is not None else []
        Q = list(tets) if tets is not None else []
    else:
        E = _edges_from_U(U, min_points=mp) if max_simplex_dim >= 1 else []
        T = _triangles_from_U(U, min_points=mp) if max_simplex_dim >= 2 else []
        Q = _tets_from_U(U, min_points=mp) if max_simplex_dim >= 3 else []

    n1, n2, n3 = len(E), len(T), len(Q)

    warnings: list[str] = []

    max_order_show = max_order if max_order > 4 else None
    n_ge_5_show = int(np.sum(sample_overlap >= 5)) if max_order > 4 else None

    if max_order > (max_dim_reported + 1):
        warnings.append(
            f"U contains overlaps of order {max_order} (true nerve has simplices up to dim {max_order - 1}), "
            f"but summary only computed/recorded up to dim {max_dim_reported}."
        )

    # Cardinalities
    vert_card = edge_card = tri_card = tet_card = None
    if compute_cardinalities:
        vert_card = U.sum(axis=1).astype(int)
        edge_card = _cards_for_simplices(U, [tuple(e) for e in E]) if n1 > 0 else np.array([], dtype=int)
        tri_card  = _cards_for_simplices(U, [tuple(t) for t in T]) if n2 > 0 else np.array([], dtype=int)
        tet_card  = _cards_for_simplices(U, [tuple(tt) for tt in Q]) if n3 > 0 else np.array([], dtype=int)

    summ = NerveSummary(
        n_sets=int(n_sets),
        n_samples=int(n_samples),
        n0=int(n_sets),
        n1=int(n1),
        n2=int(n2),
        n3=int(n3),
        vert_card=vert_card,
        edge_card=edge_card,
        tri_card=tri_card,
        tet_card=tet_card,
        sample_overlap_counts=sample_overlap,
        max_overlap_order=max_order_show,
        n_samples_with_overlap_ge_5=n_ge_5_show,
        warnings=tuple(warnings),
    )

    if verbose:
        mode = "auto" if latex == "auto" else ("latex" if latex is True else "text")
        # show_summary now auto-plots if cardinalities exist
        summ.show_summary(show=True, mode=mode)

    if plot:
        if not compute_cardinalities:
            raise ValueError("plot=True requires compute_cardinalities=True.")
        summ.plot_boxplot(
            show_tets=bool(show_tets_plot),
            dpi=int(dpi),
            figsize=figsize,
            save_path=save_path,
            show=True,
        )

    return summ
