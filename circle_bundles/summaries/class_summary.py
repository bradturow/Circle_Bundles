# circle_bundles/summaries/class_summary.py
"""
Uniform summary for characteristic classes + persistence (Bundle.get_classes output).

This is the *modern* summary used by Bundle.summarize_classes(), and should be the only
summary shown in the docs for class/persistence output.

Design goals (Feb 2026)
----------------------
- Summarize "class representatives" + "restricted derived class report" + "persistence" cleanly.
- Do NOT print local-trivialization diagnostics (that's local_triv_summary's job).
- Provide a stable plain-text summary via `summary_text` for logging / unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


__all__ = [
    "ClassSummary",
    "summarize_classes_and_persistence",
    # exported so other summaries / docs can call it
    "summarize_edge_driven_persistence",
]


Edge = Tuple[int, int]
Tri = Tuple[int, int, int]
Tet = Tuple[int, int, int, int]


# ============================================================
# Small canon helpers (local; keep file self-contained)
# ============================================================

def canon_edge_tuple(e: Edge) -> Edge:
    a, b = int(e[0]), int(e[1])
    return (a, b) if a <= b else (b, a)


def canon_tri_tuple(t: Tri) -> Tri:
    i, j, k = int(t[0]), int(t[1]), int(t[2])
    return tuple(sorted((i, j, k)))  # type: ignore


def canon_tet_tuple(tt: Tet) -> Tet:
    a, b, c, d = int(tt[0]), int(tt[1]), int(tt[2]), int(tt[3])
    return tuple(sorted((a, b, c, d)))  # type: ignore


def induced_triangles_from_edges(tris: List[Tri], kept_edges: set[Edge]) -> List[Tri]:
    out: List[Tri] = []
    for (i, j, k) in tris:
        e1 = canon_edge_tuple((i, j))
        e2 = canon_edge_tuple((i, k))
        e3 = canon_edge_tuple((j, k))
        if (e1 in kept_edges) and (e2 in kept_edges) and (e3 in kept_edges):
            out.append(canon_tri_tuple((i, j, k)))
    return out


def induced_tetrahedra_from_edges(tets: List[Tet], kept_edges: set[Edge]) -> List[Tet]:
    out: List[Tet] = []
    for (a, b, c, d) in tets:
        edges6 = [
            canon_edge_tuple((a, b)),
            canon_edge_tuple((a, c)),
            canon_edge_tuple((a, d)),
            canon_edge_tuple((b, c)),
            canon_edge_tuple((b, d)),
            canon_edge_tuple((c, d)),
        ]
        if all(e in kept_edges for e in edges6):
            out.append(canon_tet_tuple((a, b, c, d)))
    return out


# ============================================================
# Formatting helpers
# ============================================================

def _safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    try:
        return bool(x)
    except Exception:
        return None


def _fmt_int_or_dash(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return str(int(x))
    except Exception:
        return "—"


def _fmt_float_or_dash(x: Any, *, decimals: int = 4) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "—"
        return f"{xf:.{decimals}f}"
    except Exception:
        return "—"


def _get_euler_rounding_distance(obj: Any) -> Optional[float]:
    for name in ("rounding_dist",):
        val = getattr(obj, name, None)
        if val is None:
            continue
        try:
            v = float(val)
            if np.isfinite(v):
                return v
        except Exception:
            pass
    return None


# ============================================================
# OLD persistence summary (verbatim, as requested)
# ============================================================

CobirthResult = Any
CodeathResult = Any


def summarize_edge_driven_persistence(
    p: Any,
    *,
    top_k: int = 10,
    show: bool = True,
    mode: str = "auto",  # {"auto","text","latex","both"}
    show_weight_hist: bool = False,
    hist_bins: int = 40,
) -> Dict[str, Any]:
    """
    Persistence summary in the same style as other summaries, with:
      - rows: full, w1 cobirth, w1 codeath, Euler cobirth, Euler codeath
      - columns: k, r, |W_r^(1)|, |W_r^(2)|, |W_r^(3)|
    where W_r is the induced subcomplex on the kept edges at cutoff r.

    Notes
    -----
    - "k" means number of edges removed (heaviest-first).
    - "r" means cutoff weight at that stage (max finite edge weight for the full complex).
    - |W_r^(d)| are the counts of d-simplices in the induced subcomplex.

    If show_weight_hist=True, renders a side-by-side Matplotlib figure with:
      - left: a table of the summary rows
      - right: a histogram of all edge weights, with markers for key events
    """
    # ----------------------------
    # Canonicalize full complex
    # ----------------------------
    E_all = sorted({canon_edge_tuple(e) for e in p.edges})
    T_all = sorted({canon_tri_tuple(t) for t in p.triangles})
    TT_all = sorted({canon_tet_tuple(tt) for tt in p.tets})

    ew = {canon_edge_tuple(e): float(w) for e, w in p.edge_weights.items()}

    # Filtration order (heaviest removed first)
    rem_order = [canon_edge_tuple(e) for e in list(p.sw1["removal_order"])]

    # ----------------------------
    # Helpers
    # ----------------------------
    def fmt_r_3(w: float) -> str:
        if np.isposinf(w):
            return "∞"
        if np.isneginf(w):
            return "-∞"
        return f"{float(w):.3f}"

    def worst(edges: List[Edge]) -> List[Tuple[Edge, float]]:
        arr = [(canon_edge_tuple(e), ew[canon_edge_tuple(e)]) for e in edges if canon_edge_tuple(e) in ew]
        arr.sort(key=lambda t: (-t[1], t[0]))
        return arr[:top_k]

    def stage_sizes(k_removed: int) -> Tuple[int, int, int]:
        k = int(k_removed)
        if k < 0:
            raise ValueError(f"k_removed must be >= 0 (got {k}). This should not happen.")
        k = max(0, min(k, len(rem_order)))

        removed = set(rem_order[:k])
        kept_edges = set(E_all) - removed

        kept_tris = induced_triangles_from_edges(T_all, kept_edges)
        kept_tets = induced_tetrahedra_from_edges(TT_all, kept_edges) if TT_all else []

        return (len(kept_edges), len(kept_tris), len(kept_tets))

    # "Full complex" cutoff: max finite edge weight (more truthful than ∞)
    finite_weights = [w for w in ew.values() if np.isfinite(w)]
    r_full = max(finite_weights) if finite_weights else float("inf")
    r_full_str = fmt_r_3(float(r_full))

    # ----------------------------
    # Pull events
    # ----------------------------
    sw1_cob: CobirthResult = p.sw1["cobirth"]
    sw1_cod: CodeathResult = p.sw1["codeath"]
    te_cob: CobirthResult = p.twisted_euler["cobirth"]
    te_cod: CodeathResult = p.twisted_euler["codeath"]

    for name, res in [
        ("sw1_cobirth", sw1_cob),
        ("sw1_codeath", sw1_cod),
        ("euler_cobirth", te_cob),
        ("euler_codeath", te_cod),
    ]:
        if int(res.k_removed) < 0:
            raise ValueError(f"{name}.k_removed is {res.k_removed}. Expected >= 0.")

    # ----------------------------
    # Stage rows
    # ----------------------------
    rows: List[Tuple[str, int, str, int, int, int, List[Tuple[Edge, float]]]] = []

    full_ke, full_kt, full_ktt = stage_sizes(0)
    rows.append(("Full nerve", 0, r_full_str, full_ke, full_kt, full_ktt, []))

    def add_row(label: str, res_obj: Union[CobirthResult, CodeathResult]):
        k = int(res_obj.k_removed)
        r_str = fmt_r_3(float(res_obj.cutoff_weight))
        ke, kt, ktt = stage_sizes(k)
        rows.append((label, k, r_str, int(ke), int(kt), int(ktt), worst(list(res_obj.removed_edges))))

    add_row("w₁ cobirth", sw1_cob)
    add_row("w₁ codeath", sw1_cod)
    add_row("Euler cobirth", te_cob)
    add_row("Euler codeath", te_cod)

    out: Dict[str, Any] = {
        "n_edges_total": int(len(E_all)),
        "n_triangles_total": int(len(T_all)),
        "n_tets_total": int(len(TT_all)),
        "rows": [
            {
                "stage": lab,
                "k_removed": int(k),
                "r_str": r_str,
                "W1": int(ne),
                "W2": int(nt),
                "W3": int(ntt),
                "removed_edges_top": red_top,
            }
            for (lab, k, r_str, ne, nt, ntt, red_top) in rows
        ],
        # Back-compat keys
        "SW1 cobirth": {
            "k_removed": int(sw1_cob.k_removed),
            "cutoff_weight": float(sw1_cob.cutoff_weight),
            "r_str": fmt_r_3(float(sw1_cob.cutoff_weight)),
            "|W_r^(1)|": stage_sizes(int(sw1_cob.k_removed))[0],
            "|W_r^(2)|": stage_sizes(int(sw1_cob.k_removed))[1],
            "|W_r^(3)|": stage_sizes(int(sw1_cob.k_removed))[2],
            "removed_edges_top": worst(list(sw1_cob.removed_edges)),
        },
        "SW1 codeath": {
            "k_removed": int(sw1_cod.k_removed),
            "cutoff_weight": float(sw1_cod.cutoff_weight),
            "r_str": fmt_r_3(float(sw1_cod.cutoff_weight)),
            "|W_r^(1)|": stage_sizes(int(sw1_cod.k_removed))[0],
            "|W_r^(2)|": stage_sizes(int(sw1_cod.k_removed))[1],
            "|W_r^(3)|": stage_sizes(int(sw1_cod.k_removed))[2],
            "removed_edges_top": worst(list(sw1_cod.removed_edges)),
        },
        "Euler cobirth": {
            "k_removed": int(te_cob.k_removed),
            "cutoff_weight": float(te_cob.cutoff_weight),
            "r_str": fmt_r_3(float(te_cob.cutoff_weight)),
            "|W_r^(1)|": stage_sizes(int(te_cob.k_removed))[0],
            "|W_r^(2)|": stage_sizes(int(te_cob.k_removed))[1],
            "|W_r^(3)|": stage_sizes(int(te_cob.k_removed))[2],
            "removed_edges_top": worst(list(te_cob.removed_edges)),
        },
        "Euler codeath": {
            "k_removed": int(te_cod.k_removed),
            "cutoff_weight": float(te_cod.cutoff_weight),
            "r_str": fmt_r_3(float(te_cod.cutoff_weight)),
            "|W_r^(1)|": stage_sizes(int(te_cod.k_removed))[0],
            "|W_r^(2)|": stage_sizes(int(te_cod.k_removed))[1],
            "|W_r^(3)|": stage_sizes(int(te_cod.k_removed))[2],
            "removed_edges_top": worst(list(te_cod.removed_edges)),
        },
    }

    # ----------------------------
    # Renderers
    # ----------------------------
    def _display_summary_latex_persistence(rows_for_tex) -> bool:
        try:
            from IPython.display import display, Math  # type: ignore
        except Exception:
            return False
    
        def _fmt_r_tex(r_str: str) -> str:
            if r_str == "∞":
                return r"\infty"
            if r_str == "-∞":
                return r"-\infty"
            return r_str
    
        # ---- Title (own display) ----
        display(Math(r"\textbf{Class Persistence}"))
    
        # ---- Table (own display) ----
        body = []
        for stage, k, r_str, ne, nt, ntt in rows_for_tex:
            body.append(
                rf"\text{{{stage}}} & {k} & {_fmt_r_tex(r_str)} & {ne} & {nt} & {ntt}"
            )
    
        table = (
            r"\begin{array}{lrrrrr}"
            r"\textbf{Stage} & k & r & |W_r^{(1)}| & |W_r^{(2)}| & |W_r^{(3)}| \\ \hline "
            + r" \\ ".join(body)
            + r"\end{array}"
        )
    
        display(Math(table))
        return True




    def _print_text_table(rows_for_txt):
        title = "=== Characteristic Class Persistence ==="
        print("\n" + title)

        stage_w = max(12, max(len(r[0]) for r in rows_for_txt))
        k_w = 6
        r_w = 10
        w1_w = 12
        w2_w = 12
        w3_w = 12

        header = (
            f"{'Stage':<{stage_w}}"
            f"{'k':>{k_w}}  "
            f"{'r':>{r_w}}  "
            f"{'|W_r^(1)|':>{w1_w}}  "
            f"{'|W_r^(2)|':>{w2_w}}  "
            f"{'|W_r^(3)|':>{w3_w}}"
        )
        print(header)
        print("-" * len(header))

        for stage, k, r_str, ne, nt, ntt in rows_for_txt:
            print(
                f"{stage:<{stage_w}}"
                f"{int(k):>{k_w}}  "
                f"{r_str:>{r_w}}  "
                f"{int(ne):>{w1_w}}  "
                f"{int(nt):>{w2_w}}  "
                f"{int(ntt):>{w3_w}}"
            )
        print("")

    def _show_side_table_and_hist(rows_for_tbl) -> bool:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return False

        weights = [float(w) for w in ew.values() if np.isfinite(float(w))]
        if len(weights) == 0:
            return False

        events = [
            ("w₁ cobirth", float(sw1_cob.cutoff_weight)),
            ("w₁ codeath", float(sw1_cod.cutoff_weight)),
            ("Euler cobirth", float(te_cob.cutoff_weight)),
            ("Euler codeath", float(te_cod.cutoff_weight)),
        ]

        fig, (ax_tbl, ax_h) = plt.subplots(
            1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
        )

        # Left: table
        ax_tbl.axis("off")
        col_labels = ["Stage", "k", "r", "|W¹|", "|W²|", "|W³|"]
        cell_text = [
            [stage, str(k), r_str, str(ne), str(nt), str(ntt)]
            for (stage, k, r_str, ne, nt, ntt) in rows_for_tbl
        ]
        table = ax_tbl.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 3.2)

        # Right: histogram
        ax_h.hist(weights, bins=int(hist_bins))
        ax_h.set_title("Edge-weight distribution")
        ax_h.set_xlabel("Edge weight")
        ax_h.set_ylabel("Count")

        for label, w in events:
            if np.isfinite(w):
                ax_h.axvline(w)
                y = ax_h.get_ylim()[1]
                ax_h.text(w, 0.95 * y, label, rotation=90, va="top", ha="right", fontsize=8)

        fig.tight_layout()
        plt.show()
        return True

    # ----------------------------
    # Display
    # ----------------------------
    if show:
        core_rows = [(lab, k, r_str, ne, nt, ntt) for (lab, k, r_str, ne, nt, ntt, _top) in rows]

        if show_weight_hist:
            ok = _show_side_table_and_hist(core_rows)
            if not ok:
                did_latex = False
                if mode in {"latex", "auto", "both"}:
                    did_latex = _display_summary_latex_persistence(core_rows)
                if mode == "both" or mode == "text" or (mode == "auto" and not did_latex):
                    _print_text_table(core_rows)
        else:
            did_latex = False
            if mode in {"latex", "auto", "both"}:
                did_latex = _display_summary_latex_persistence(core_rows)

            if mode == "both" or mode == "text" or (mode == "auto" and not did_latex):
                _print_text_table(core_rows)

    return out


# ============================================================
# Summary container
# ============================================================

@dataclass
class ClassSummary:
    """
    Summary output for class reps + persistence + restricted derived class data.
    """
    reps: Any
    restricted: Any
    persistence: Any
    summary_text: str

    # cached convenience
    euler_rounding_distance: Optional[float] = None

    # computed warning bits
    euler_not_cocycle_on_full: Optional[bool] = None
    euler_cobirth_k_removed: Optional[int] = None
    euler_cobirth_cutoff_weight: Optional[float] = None

    def show_summary(
        self,
        *,
        show_classes: bool = True,
        show_persistence: bool = True,
        show_rounding_distance: bool = False,
        persistence_mode: str = "auto",
    ) -> str:
        """
        Display the summary with clean toggles.

        - Classes first (LaTeX if possible, else text).
        - Then vertical space + "Class Persistence" + the legacy persistence table.
        """
        if (not show_classes) and (not show_persistence):
            return ""

        # ---- 1) Classes block first ----
        if show_classes:
            did_latex = self._display_latex(show_rounding_distance=bool(show_rounding_distance))
            txt = str(self.summary_text).strip()
            if (not did_latex) and txt:
                print("\n" + txt + "\n")

        # ---- 2) Persistence block underneath ----
        if show_persistence:
            if show_classes:
                print("\n")
            summarize_edge_driven_persistence(
                self.persistence,
                show=True,
                mode=str(persistence_mode),
                show_weight_hist=False,
            )

        return str(self.summary_text)

    def _display_latex(self, *, show_rounding_distance: bool = False) -> bool:
        try:
            from IPython.display import display, Math  # type: ignore
        except Exception:
            return False

        reps = self.reps
        r = self.restricted

        orientable = _safe_bool(getattr(r, "orientable", getattr(reps, "orientable", None)))

        w1_cob = getattr(r, "sw1_is_coboundary_Z2", None)
        if w1_cob is None:
            w1_cob = bool(orientable) if orientable is not None else None

        e_is_cob = getattr(r, "euler_is_coboundary_Z", None)
        eZ = getattr(r, "twisted_euler_number_Z", None)
        spin = getattr(r, "spin_on_this_complex", None)

        # Infer "Euler is zero" for printing (no special-casing for n_triangles==0)
        euler_rep = getattr(r, "euler_class", None)
        if not isinstance(euler_rep, dict):
            euler_rep = {}
        e_trivial_cochain = all(int(v) == 0 for v in euler_rep.values())  # empty => True
        if e_is_cob is not None:
            e_zero_for_print = bool(e_is_cob)
        else:
            e_zero_for_print = bool(e_trivial_cochain)

        # Warning row (based on persistence cobirth)
        warn = self.euler_not_cocycle_on_full
        warn_k = self.euler_cobirth_k_removed
        warn_w = self.euler_cobirth_cutoff_weight

        def _w1_tex() -> str:
            if w1_cob is None:
                return r"w_1\ \text{(unknown)}"
            return r"w_1 = 0\ (\text{orientable})" if bool(w1_cob) else r"w_1 \neq 0\ (\text{non-orientable})"

        def _e_tex() -> str:
            if e_zero_for_print:
                return r"e = 0\ (\text{trivial})" if bool(orientable) else r"\tilde{e}=0"
            if eZ is not None:
                k = abs(int(eZ))
                if bool(orientable):
                    parity_note = r"\ (\text{spin})" if (k % 2 == 0) else r"\ (\text{not spin})"
                    return rf"\pm {k}" + parity_note
                return rf"\pm {k}"
            return r"e \neq 0" if bool(orientable) else r"\tilde{e}\neq 0"

        rows: List[Tuple[str, str]] = []
        rows.append((r"\text{Stiefel--Whitney}", _w1_tex()))
        e_label = r"\text{Euler}" if bool(orientable) else r"\text{(twisted) Euler}"
        rows.append((e_label, _e_tex()))

        # Only report spin separately when Euler number is NOT provided
        # (If eZ is provided, we already show "(spin)/(not spin)" next to the Euler number.)
        if (spin is not None) and bool(orientable) and (eZ is None):
            w2_tex = r"w_2 = 0\ (\text{spin})" if bool(spin) else r"w_2 \neq 0\ (\text{not spin})"
            rows.append((r"\text{Spin}", w2_tex))

        if warn is True:
            kk = r"\text{?}" if warn_k is None else str(int(warn_k))
            ww = r"\text{?}" if warn_w is None or (not np.isfinite(float(warn_w))) else f"{float(warn_w):.3g}"
            rows.append((r"\text{Warning}", rf"\tilde{{e}}\ \text{{not cocycle on full nerve}}\ (k={kk},\, r={ww})"))

        if show_rounding_distance:
            rd = self.euler_rounding_distance
            if rd is not None and np.isfinite(float(rd)):
                rows.append((r"\text{Euler rounding dist.}", rf"{float(rd):.6g}"))
            else:
                rows.append((r"\text{Euler rounding dist.}", r"\text{—}"))

        body = r"\\[4pt]".join([r"\quad " + a + r" &:\quad " + b for (a, b) in rows])

        latex = (
            r"\begin{aligned}"
            r"\textbf{Characteristic Classes} & \\[8pt]"
            + body
            + r"\end{aligned}"
        )

        try:
            display(Math(latex))
            return True
        except Exception:
            return False


# ============================================================
# Builder
# ============================================================

def summarize_classes_and_persistence(*, reps: Any, restricted: Any, persistence: Any) -> ClassSummary:
    """
    Build a ClassSummary from the objects produced by Bundle.get_classes().

    The stable `summary_text` here is ONLY for the classes block; persistence is displayed
    via `summarize_edge_driven_persistence` to preserve the legacy output exactly.
    """
    r = restricted
    orientable = _safe_bool(getattr(r, "orientable", getattr(reps, "orientable", None)))

    w1_cob = getattr(r, "sw1_is_coboundary_Z2", None)
    if w1_cob is None:
        w1_cob = bool(orientable) if orientable is not None else None

    e_is_cob = getattr(r, "euler_is_coboundary_Z", None)
    eZ = getattr(r, "twisted_euler_number_Z", None)
    spin = getattr(r, "spin_on_this_complex", None)

    # Infer "Euler is zero" for printing (no special-casing for n_triangles==0)
    euler_rep = getattr(r, "euler_class", None)
    if not isinstance(euler_rep, dict):
        euler_rep = {}
    e_trivial_cochain = all(int(v) == 0 for v in euler_rep.values())  # empty => True
    if e_is_cob is not None:
        e_zero_for_print = bool(e_is_cob)
    else:
        e_zero_for_print = bool(e_trivial_cochain)

    # --- rounding distance (from reps or restricted; both have it in your pipeline)
    rd = _get_euler_rounding_distance(r)
    if rd is None:
        rd = _get_euler_rounding_distance(reps)

    # --- WARNING: detect if Euler rep is not a cocycle on the FULL nerve
    # Using persistence: if cobirth requires removing >=1 edges, then at k=0 (full) it was not a cocycle.
    euler_not_cocycle_on_full: Optional[bool] = None
    euler_cob_k: Optional[int] = None
    euler_cob_w: Optional[float] = None
    try:
        te_cob = persistence.twisted_euler["cobirth"]
        euler_cob_k = int(getattr(te_cob, "k_removed", None))
        euler_cob_w = float(getattr(te_cob, "cutoff_weight", None))
        if euler_cob_k is not None:
            euler_not_cocycle_on_full = bool(int(euler_cob_k) > 0)
    except Exception:
        pass

    IND = "  "
    LABEL = 28

    def _tline(label: str, content: str) -> str:
        return f"{IND}{label:<{LABEL}} {content}"

    # ---- TEXT summary (classes only) ----
    lines: List[str] = []
    lines.append("=== Characteristic Classes ===")

    lines.append(
        _tline(
            "Stiefel–Whitney:",
            "w₁ = 0 (orientable)"
            if (w1_cob is True)
            else ("w₁ ≠ 0 (non-orientable)" if (w1_cob is False) else "—"),
        )
    )

    if e_zero_for_print:
        if bool(orientable):
            lines.append(_tline("Euler class:", "e = 0 (trivial)"))
        else:
            lines.append(_tline("(twisted) Euler:", "ẽ = 0"))
    else:
        if eZ is not None:
            k = abs(int(eZ))
            if bool(orientable):
                parity_note = " (spin)" if (k % 2 == 0) else " (not spin)"
                lines.append(_tline("Euler number:", f"±{k}{parity_note}"))
            else:
                lines.append(_tline("(twisted) Euler #:", f"±{k}"))
        else:
            if bool(orientable):
                lines.append(_tline("Euler class:", "e ≠ 0 (non-trivial)"))
            else:
                lines.append(_tline("(twisted) Euler:", "ẽ ≠ 0"))

    # Only report spin separately when Euler number is NOT provided
    # (If eZ is provided, parity note "(spin)/(not spin)" is already shown.)
    if (spin is not None) and bool(orientable) and (eZ is None):
        if bool(spin):
            lines.append(_tline("Spin:", "w₂ = 0 (spin)"))
        else:
            lines.append(_tline("Spin:", "w₂ ≠ 0 (not spin)"))
    
    # --- requested warning ---
    if euler_not_cocycle_on_full is True:
        k_str = _fmt_int_or_dash(euler_cob_k)
        w_str = _fmt_float_or_dash(euler_cob_w, decimals=4)
        lines.append(
            _tline(
                "WARNING:",
                f"twisted Euler rep is not a cocycle on the full nerve (cobirth at k={k_str}, r={w_str}).",
            )
        )

    text = "\n".join(lines)

    return ClassSummary(
        reps=reps,
        restricted=restricted,
        persistence=persistence,
        summary_text=text,
        euler_rounding_distance=rd,
        euler_not_cocycle_on_full=euler_not_cocycle_on_full,
        euler_cobirth_k_removed=euler_cob_k,
        euler_cobirth_cutoff_weight=euler_cob_w,
    )
