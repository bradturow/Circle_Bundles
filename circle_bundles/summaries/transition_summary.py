# circle_bundles/summaries/transition_summary.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..nerve.combinatorics import Edge

if TYPE_CHECKING:
    from ..analysis.quality import BundleQualityReport


# --- helpers copied (or imported) from characteristic_class.py style ---
def _euc_to_geo_rad(d: Optional[float]) -> Optional[float]:
    """
    Convert chordal distance d in [0,2] on S^1 ⊂ C to geodesic angle in radians in [0, π]:
        d = 2 sin(theta/2)  =>  theta = 2 arcsin(d/2).
    """
    if d is None:
        return None
    d = float(d)
    if not np.isfinite(d):
        return None
    x = np.clip(d / 2.0, 0.0, 1.0)
    return float(2.0 * np.arcsin(x))


def _fmt_euc_with_geo_pi(d_euc: Optional[float], *, decimals: int = 3) -> str:
    if d_euc is None or not np.isfinite(float(d_euc)):
        return "—"
    d_euc = float(d_euc)
    theta = _euc_to_geo_rad(d_euc)
    if theta is None:
        return f"{d_euc:.{decimals}f}"
    return f"{d_euc:.{decimals}f} (\\varepsilon_{{\\text{{triv}}}}^{{\\text{{geo}}}}={theta/np.pi:.{decimals}f}\\pi)"


def _fmt_mean_euc_with_geo_pi(d_euc_mean: Optional[float], *, decimals: int = 3) -> str:
    if d_euc_mean is None or not np.isfinite(float(d_euc_mean)):
        return "—"
    d_euc_mean = float(d_euc_mean)
    theta = _euc_to_geo_rad(d_euc_mean)
    if theta is None:
        return f"{d_euc_mean:.{decimals}f}"
    return f"{d_euc_mean:.{decimals}f} (\\bar{{\\varepsilon}}_{{\\text{{triv}}}}^{{\\text{{geo}}}}={theta/np.pi:.{decimals}f}\\pi)"


def _canon_edge_weight_map(d: object) -> Optional[Dict[Edge, float]]:
    """
    Best-effort conversion of an edge->weight mapping into canonical form.

    Accepts mappings whose keys are (i,j) pairs in any order.
    Produces a dict keyed by Edge = (min(i,j), max(i,j)).
    Returns None if input is None or cannot be interpreted as a mapping.
    """
    if d is None:
        return None
    try:
        items = dict(d).items()
    except Exception:
        return None

    out: Dict[Edge, float] = {}
    for k, w in items:
        try:
            a, b = k
            e: Edge = (int(a), int(b))
            if e[0] > e[1]:
                e = (e[1], e[0])
            out[e] = float(w)
        except Exception:
            continue

    return out or None


@dataclass
class TransitionSummary:
    """
    A lightweight summary: (Diagnostics) + (Transitions).

    IMPORTANT (post-refactor)
    -------------------------
    The Bundle pipeline may no longer retain a full TransitionReport object.
    Therefore this summary is designed to work in two modes:

      1) If `report` is provided, transitions section uses it (backward compatible).
      2) If `report` is None, transitions section is derived from `quality`
         (requires BundleQualityReport to include rms aggregates and per-edge map).

    Diagnostics portion is aligned with characteristic_class.show_summary (minus Euler rounding).
    """
    n_sets: int
    n_samples: int
    quality: Optional["BundleQualityReport"] = None

    # Optional legacy payload (kept for backwards compatibility)
    report: Optional[Any] = None

    warnings: Tuple[str, ...] = ()

    # ----------------------------
    # Plain-text (Sphinx-friendly)
    # ----------------------------

    def to_text(self) -> str:
        IND = "  "
        LABEL_W = 28

        def _tline(label: str, content: str) -> str:
            return f"{IND}{label:<{LABEL_W}} {content}"

        q = self.quality
        rep = self.report

        lines: List[str] = []
        lines.append("=== Diagnostics ===")

        if q is None:
            lines.append(f"{IND}(no quality report provided)")
        else:
            eps_triv = getattr(q, "eps_align_euc", None)
            eps_triv_mean = getattr(q, "eps_align_euc_mean", None)
            delta = getattr(q, "delta", None)
            alpha = getattr(q, "alpha", None)
            eps_coc = getattr(q, "cocycle_defect", None)

            if eps_triv is not None:
                lines.append(
                    _tline(
                        "trivialization error:",
                        "ε_triv := sup_{(j k)∈N(U)} sup_{x∈π^{-1}(U_j∩U_k)} d_𝕮(Ω_{jk} f_k(x), f_j(x))"
                        f" = {_fmt_euc_with_geo_pi(eps_triv)}",
                    )
                )
            if eps_triv_mean is not None:
                lines.append(
                    _tline(
                        "mean triv error:",
                        f"\\bar{{ε}}_triv = {_fmt_mean_euc_with_geo_pi(eps_triv_mean)}",
                    )
                )
            if delta is not None:
                lines.append(
                    _tline(
                        "surjectivity defect:",
                        "δ := sup_{(i j k)∈N(U)} min_{v∈{i,j,k}} d_H(f_v(π^{-1}(U_i∩U_j∩U_k)), S^1)"
                        f" = {float(delta):.3f}",
                    )
                )
            if alpha is not None:
                if float(alpha) == float("inf"):
                    lines.append(_tline("stability ratio:", "α := ε_triv/(1-δ) = ∞  (since δ ≥ 1)"))
                else:
                    lines.append(_tline("stability ratio:", f"α := ε_triv/(1-δ) = {float(alpha):.3f}"))
            if eps_coc is not None:
                lines.append(
                    _tline(
                        "cocycle error:",
                        "ε_coc := sup_{(i j k)∈N(U)} ‖Ω_{ij}Ω_{jk}Ω_{ki} - I‖_F"
                        f" = {float(eps_coc):.3f}",
                    )
                )

        lines.append("")
        lines.append("=== Transitions (O(2) estimation) ===")

        # --- transitions block: prefer legacy report if present ---
        if rep is not None:
            # Best-effort access (don’t assume exact class)
            mp = getattr(rep, "min_points", None)
            n_req = getattr(rep, "n_edges_requested", None)
            n_est = getattr(rep, "n_edges_estimated", None)
            missing = getattr(rep, "missing_edges", None)
            mean_rms = getattr(rep, "mean_rms_angle_err", None)
            max_rms = getattr(rep, "max_rms_angle_err", None)

            lines.append(_tline("min_points:", f"{int(mp)}" if mp is not None else "—"))
            lines.append(_tline("edges requested:", f"{int(n_req)}" if n_req is not None else "—"))
            lines.append(_tline("edges estimated:", f"{int(n_est)}" if n_est is not None else "—"))
            if missing:
                try:
                    lines.append(_tline("missing edges:", f"{len(missing)}"))
                except Exception:
                    lines.append(_tline("missing edges:", "—"))
            if mean_rms is not None:
                lines.append(_tline("mean RMS angle err:", f"{float(mean_rms):.6g} rad"))
            else:
                lines.append(_tline("mean RMS angle err:", "—"))
            if max_rms is not None:
                lines.append(_tline("max  RMS angle err:", f"{float(max_rms):.6g} rad"))
            else:
                lines.append(_tline("max  RMS angle err:", "—"))

        # --- otherwise derive from quality ---
        else:
            if q is None:
                lines.append(_tline("(no transitions data)", "—"))
            else:
                rms_map = _canon_edge_weight_map(getattr(q, "rms_angle_err", None))
                mp = getattr(q, "min_points", None)  # usually not present; keep soft
                n_req = getattr(q, "n_edges_requested", None)
                n_est = getattr(q, "n_edges_estimated", None)
                mean_rms = getattr(q, "mean_edge_rms", None)
                max_rms = getattr(q, "max_edge_rms", None)

                lines.append(_tline("min_points:", f"{int(mp)}" if mp is not None else "—"))
                lines.append(_tline("edges requested:", f"{int(n_req)}" if n_req is not None else "—"))
                lines.append(_tline("edges estimated:", f"{int(n_est)}" if n_est is not None else "—"))

                # If we have the per-edge map, we can report its size too (helpful sanity check)
                if rms_map is not None:
                    lines.append(_tline("RMS map entries:", f"{len(rms_map)}"))

                lines.append(
                    _tline(
                        "mean RMS angle err:",
                        f"{float(mean_rms):.6g} rad" if mean_rms is not None else "—",
                    )
                )
                lines.append(
                    _tline(
                        "max  RMS angle err:",
                        f"{float(max_rms):.6g} rad" if max_rms is not None else "—",
                    )
                )

        for w in self.warnings:
            lines.append("")
            lines.append(f"{IND}WARNING: {w}")

        return "\n".join(lines)

    # ----------------------------
    # Rich display (Notebook-friendly)
    # ----------------------------

    def show_summary(self, *, show: bool = True, mode: str = "auto") -> str:
        """
        mode:
          - "auto": try LaTeX, else print text
          - "latex": LaTeX only (fallback to text if LaTeX display fails)
          - "text": print text only
          - "both": LaTeX (if possible) + also print text
        """
        text = self.to_text()
        if not show:
            return text

        did_latex = False
        if mode in {"latex", "auto", "both"}:
            did_latex = _display_summary_latex(self)

        if mode == "both" or mode == "text" or (mode == "auto" and not did_latex):
            print("\n" + text + "\n")

        return text


def _display_summary_latex(summary: TransitionSummary) -> bool:
    """
    Best-effort IPython Math display.

    Mirrors the characteristic_class summary *Diagnostics* rows (minus Euler rounding),
    then adds a small Transitions block.

    Works with either a legacy `report` OR quality-derived transition stats.
    """
    try:
        from IPython.display import display, Math  # type: ignore
    except Exception:
        return False

    q = summary.quality
    rep = summary.report

    # Pull diagnostics
    eps_triv = getattr(q, "eps_align_euc", None) if q is not None else None
    eps_triv_mean = getattr(q, "eps_align_euc_mean", None) if q is not None else None
    delta = getattr(q, "delta", None) if q is not None else None
    alpha = getattr(q, "alpha", None) if q is not None else None
    eps_coc = getattr(q, "cocycle_defect", None) if q is not None else None

    # helper for latex numeric strings with geo-in-π parentheses
    def _latex_eps_with_geo_pi(d_euc: Optional[float], *, decimals: int = 3, mean: bool = False) -> str:
        if d_euc is None or not np.isfinite(float(d_euc)):
            return r"\text{—}"
        d_euc = float(d_euc)
        theta = _euc_to_geo_rad(d_euc)
        if theta is None:
            return f"{d_euc:.{decimals}f}"
        if mean:
            return (
                f"{d_euc:.{decimals}f}"
                + r"\ \left(\bar{\varepsilon}_{\text{triv}}^{\text{geo}}="
                + f"{theta/np.pi:.{decimals}f}"
                + r"\pi\right)"
            )
        return (
            f"{d_euc:.{decimals}f}"
            + r"\ \left(\varepsilon_{\text{triv}}^{\text{geo}}="
            + f"{theta/np.pi:.{decimals}f}"
            + r"\pi\right)"
        )

    diag_rows: List[Tuple[str, str]] = []
    if q is None:
        diag_rows.append((r"\text{(no quality report provided)}", r""))
    else:
        if eps_triv is not None:
            diag_rows.append(
                (
                    r"\text{Trivialization error}",
                    r"\varepsilon_{\text{triv}} := "
                    r"\sup_{(j\,k)\in\mathcal{N}(\mathcal{U})}\sup_{x\in\pi^{-1}(U_j\cap U_k)} "
                    r"d_{\mathbb{C}}(\Omega_{jk}f_k(x),f_j(x))"
                    + r" = "
                    + _latex_eps_with_geo_pi(eps_triv, mean=False),
                )
            )
        if eps_triv_mean is not None:
            diag_rows.append(
                (
                    r"\text{Mean triv error}",
                    r"\bar{\varepsilon}_{\text{triv}}"
                    + r" = "
                    + _latex_eps_with_geo_pi(eps_triv_mean, mean=True),
                )
            )
        if delta is not None:
            diag_rows.append(
                (
                    r"\text{Surjectivity defect}",
                    r"\delta := \sup_{(i\,j\,k)\in\mathcal{N}(\mathcal{U})}\min_{v\in\{i,j,k\}} "
                    r"d_H\!\left(f_v(\pi^{-1}(U_i\cap U_j\cap U_k)),\mathbb{S}^1\right)"
                    + r" = "
                    + f"{float(delta):.3f}",
                )
            )
        if alpha is not None:
            if float(alpha) == float("inf"):
                diag_rows.append((r"\text{Stability ratio}", r"\alpha := \varepsilon_{\text{triv}}/(1-\delta) = \infty"))
            else:
                diag_rows.append(
                    (
                        r"\text{Stability ratio}",
                        r"\alpha := \varepsilon_{\text{triv}}/(1-\delta) = " + f"{float(alpha):.3f}",
                    )
                )
        if eps_coc is not None:
            diag_rows.append(
                (
                    r"\text{Cocycle error}",
                    r"\varepsilon_{\mathrm{coc}} := "
                    r"\sup_{(i\,j\,k)\in\mathcal{N}(\mathcal{U})}\left\|\Omega_{ij}\Omega_{jk}\Omega_{ki}-I\right\|_F"
                    + r" = "
                    + f"{float(eps_coc):.3f}",
                )
            )

    # --- transitions rows: report-first, else quality-derived ---
    trans_rows: List[Tuple[str, str]] = []

    def _num_or_dash(x: Any) -> str:
        if x is None:
            return r"\text{—}"
        try:
            return f"{int(x)}"
        except Exception:
            return r"\text{—}"

    if rep is not None:
        trans_rows.append((r"\text{min\_points}", _num_or_dash(getattr(rep, "min_points", None))))
        trans_rows.append((r"\text{edges requested}", _num_or_dash(getattr(rep, "n_edges_requested", None))))
        trans_rows.append((r"\text{edges estimated}", _num_or_dash(getattr(rep, "n_edges_estimated", None))))
        missing = getattr(rep, "missing_edges", None)
        if missing:
            try:
                trans_rows.append((r"\text{missing edges}", f"{len(missing)}"))
            except Exception:
                pass
        mean_rms = getattr(rep, "mean_rms_angle_err", None)
        max_rms = getattr(rep, "max_rms_angle_err", None)
        trans_rows.append(
            (r"\text{mean RMS angle err}", f"{float(mean_rms):.6g}\ \mathrm{{rad}}" if mean_rms is not None else r"\text{—}")
        )
        trans_rows.append(
            (r"\text{max RMS angle err}", f"{float(max_rms):.6g}\ \mathrm{{rad}}" if max_rms is not None else r"\text{—}")
        )
    else:
        if q is None:
            trans_rows.append((r"\text{(no transitions data)}", r""))
        else:
            trans_rows.append((r"\text{min\_points}", _num_or_dash(getattr(q, "min_points", None))))
            trans_rows.append((r"\text{edges requested}", _num_or_dash(getattr(q, "n_edges_requested", None))))
            trans_rows.append((r"\text{edges estimated}", _num_or_dash(getattr(q, "n_edges_estimated", None))))

            rms_map = _canon_edge_weight_map(getattr(q, "rms_angle_err", None))
            if rms_map is not None:
                trans_rows.append((r"\text{RMS map entries}", f"{len(rms_map)}"))

            mean_rms = getattr(q, "mean_edge_rms", None)
            max_rms = getattr(q, "max_edge_rms", None)
            trans_rows.append(
                (r"\text{mean RMS angle err}", f"{float(mean_rms):.6g}\ \mathrm{{rad}}" if mean_rms is not None else r"\text{—}")
            )
            trans_rows.append(
                (r"\text{max RMS angle err}", f"{float(max_rms):.6g}\ \mathrm{{rad}}" if max_rms is not None else r"\text{—}")
            )

    def _rows_to_aligned(rows: List[Tuple[str, str]]) -> str:
        out: List[str] = []
        for label, expr in rows:
            if expr.strip() == "":
                out.append(r"\quad " + label + r" &")
            else:
                out.append(r"\quad " + label + r" &:\quad " + expr)
        return r"\\[3pt]".join(out)

    latex = (
        r"\begin{aligned}"
        r"\textbf{Diagnostics} & \\[6pt]"
        + _rows_to_aligned(diag_rows)
        + r"\\[14pt]"
        r"\textbf{Transitions (O(2) estimation)} & \\[6pt]"
        + _rows_to_aligned(trans_rows)
        + r"\end{aligned}"
    )

    try:
        display(Math(latex))
        return True
    except Exception:
        return False


def summarize_transitions(
    rep: Optional[Any] = None,
    *,
    quality: Optional["BundleQualityReport"] = None,
    n_sets: Optional[int] = None,
    n_samples: Optional[int] = None,
    warnings: Tuple[str, ...] = (),
) -> TransitionSummary:
    """
    Factory helper.

    Backward compatible:
      - Old call: summarize_transitions(rep, quality=...)
      - New call: summarize_transitions(None, quality=..., n_sets=..., n_samples=...)

    If rep is provided, n_sets/n_samples default to attributes on rep if present.
    Otherwise, you should pass n_sets/n_samples explicitly (or accept 0 defaults).
    """
    if rep is not None:
        ns = int(getattr(rep, "n_sets", n_sets or 0))
        nsm = int(getattr(rep, "n_samples", n_samples or 0))
    else:
        ns = int(n_sets or 0)
        nsm = int(n_samples or 0)

    return TransitionSummary(
        n_sets=ns,
        n_samples=nsm,
        quality=quality,
        report=rep,
        warnings=tuple(warnings),
    )
