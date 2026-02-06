"""
Class summary (Characteristic Classes + Class Persistence)

Design goals (Feb 2026)
----------------------
- NO local trivialization diagnostics here (those live in local_triv_summary).
- The ONLY diagnostic we show here is the Euler rounding distance.
- Show characteristic-class info in the same “nice” style as the other summaries:
    * text block when mode includes text
    * LaTeX block when mode includes latex (IPython available)
- Include persistence summary immediately after, with its own title:
    "Class Persistence" (same style as "Characteristic Classes")
- No "bundle trivial" line.
- No special-case "(no 2-simplices)"; treat as Euler class being trivial as usual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import io
import contextlib
import textwrap

from ..analysis.class_persistence import summarize_edge_driven_persistence


__all__ = [
    "ClassSummary",
    "summarize_classes_and_persistence",
]


# ============================================================
# Container
# ============================================================

@dataclass
class ClassSummary:
    reps: Any
    restricted: Any
    persistence: Any
    summary_text: str

    def show_summary(
        self,
        *,
        show: bool = True,
        mode: str = "auto",
        top_k: int = 10,
        show_weight_hist: bool = False,
        hist_bins: int = 40,
    ) -> None:
        if not show:
            return

        mode = str(mode)

        # ----------------------------
        # Characteristic classes block
        # ----------------------------
        did_latex = False

        if mode in {"latex", "auto", "both"}:
            did_latex = _display_class_summary_latex(self.reps, self.restricted)

        if mode in {"text", "both"} or (mode == "auto" and not did_latex):
            print("\n" + self.summary_text + "\n")

        # ----------------------------
        # Persistence title
        # ----------------------------
        if mode in {"latex", "auto", "both"}:
            _display_title_latex("Class Persistence")

        if mode in {"text", "both"} or (mode == "auto" and not did_latex):
            print("\n=== Class Persistence ===\n")

        # ----------------------------
        # Persistence table (indented)
        # ----------------------------

        want_text = (mode in {"text", "both"}) or (mode == "auto" and not did_latex)

        if want_text:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                summarize_edge_driven_persistence(
                    self.persistence,
                    top_k=int(top_k),
                    show=True,
                    mode="text",
                    show_weight_hist=bool(show_weight_hist),
                    hist_bins=int(hist_bins),
                )

            table_text = buf.getvalue().rstrip("\n")

            if table_text.strip():
                # Shift whole table right (clean + robust)
                print(textwrap.indent(table_text, "    "))

        else:
            summarize_edge_driven_persistence(
                self.persistence,
                top_k=int(top_k),
                show=True,
                mode=mode,
                show_weight_hist=bool(show_weight_hist),
                hist_bins=int(hist_bins),
            )


# ============================================================
# Public constructor
# ============================================================

def summarize_classes_and_persistence(
    reps: Any,
    restricted: Any,
    persistence: Any,
) -> ClassSummary:

    text = _build_class_summary_text(reps, restricted)

    return ClassSummary(
        reps=reps,
        restricted=restricted,
        persistence=persistence,
        summary_text=text,
    )


# ============================================================
# Text summary builder
# ============================================================

def _build_class_summary_text(reps: Any, restricted: Any) -> str:
    orientable = bool(getattr(reps, "orientable", False))
    rounding_dist = float(getattr(reps, "rounding_dist", 0.0))

    w1_is_cob = getattr(restricted, "sw1_is_coboundary_Z2", None)
    if w1_is_cob is None:
        w1_is_cob = bool(orientable)

    e_is_cob = getattr(restricted, "euler_is_coboundary_Z", None)
    eZ = getattr(restricted, "twisted_euler_number_Z", None)

    e_rep = getattr(reps, "euler_class", {}) or {}
    e_trivial_cochain = all(int(v) == 0 for v in e_rep.values()) if e_rep else True
    e_zero_for_print = bool(e_is_cob) if e_is_cob is not None else bool(e_trivial_cochain)

    lines: List[str] = []
    lines.append("=== Characteristic Classes ===")

    lines.append(
        "  Stiefel–Whitney:           "
        + ("w₁ = 0 (orientable)" if bool(w1_is_cob) else "w₁ ≠ 0 (non-orientable)")
    )

    if e_zero_for_print:
        if orientable:
            lines.append("  Euler class:               e = 0 (trivial)")
        else:
            lines.append("  (twisted) Euler class:     ẽ = 0")
    else:
        if eZ is not None:
            k = abs(int(eZ))
            if orientable:
                parity_note = " (spin)" if (k % 2 == 0) else " (not spin)"
                lines.append(f"  Euler number:              ±{k}{parity_note}")
            else:
                lines.append(f"  (twisted) Euler number:    ±{k}")
        else:
            if orientable:
                lines.append("  Euler class:               e ≠ 0 (non-trivial)")
                sp = getattr(restricted, "spin_on_this_complex", None)
                if sp is not None:
                    lines.append(
                        f"  Spin class:                {'w₂ = 0 (spin)' if bool(sp) else 'w₂ ≠ 0 (not spin)'}"
                    )
            else:
                lines.append("  (twisted) Euler class:     ẽ ≠ 0")

    # Euler rounding distance LAST
    if orientable:
        lines.append(f"  Euler rounding dist.:      d_∞(δθ, e) = {rounding_dist:.6g}")
    else:
        lines.append(f"  Euler rounding dist.:      d_∞(δ_ωθ, ẽ) = {rounding_dist:.6g}")

    return "\n".join(lines)


# ============================================================
# LaTeX helpers
# ============================================================

def _display_title_latex(title: str, *, space_above_pt: int = 12) -> bool:
    try:
        from IPython.display import display, Math, HTML  # type: ignore
    except Exception:
        return False

    try:
        if space_above_pt and int(space_above_pt) > 0:
            display(HTML(f"<div style='height:{int(space_above_pt)}pt'></div>"))

        display(Math(r"\textbf{" + _latex_escape_text(title) + r"}"))
        return True
    except Exception:
        return False


def _display_class_summary_latex(reps: Any, restricted: Any) -> bool:
    try:
        from IPython.display import display, Math  # type: ignore
    except Exception:
        return False

    orientable = bool(getattr(reps, "orientable", False))
    rounding_dist = float(getattr(reps, "rounding_dist", 0.0))

    w1_is_cob = getattr(restricted, "sw1_is_coboundary_Z2", None)
    if w1_is_cob is None:
        w1_is_cob = bool(orientable)

    e_is_cob = getattr(restricted, "euler_is_coboundary_Z", None)
    eZ = getattr(restricted, "twisted_euler_number_Z", None)

    e_rep = getattr(reps, "euler_class", {}) or {}
    e_trivial_cochain = all(int(v) == 0 for v in e_rep.values()) if e_rep else True
    e_zero_for_print = bool(e_is_cob) if e_is_cob is not None else bool(e_trivial_cochain)

    rows: List[Tuple[str, str]] = []

    rows.append(
        (r"\text{Stiefel--Whitney}",
         r"w_1 = 0\ (\text{orientable})" if bool(w1_is_cob) else r"w_1 \neq 0\ (\text{non-orientable})")
    )

    if e_zero_for_print:
        if orientable:
            rows.append((r"\text{Euler class}", r"e = 0\ (\text{trivial})"))
        else:
            rows.append((r"\text{(twisted) Euler class}", r"\tilde{e} = 0"))
    else:
        if eZ is not None:
            k = abs(int(eZ))
            if orientable:
                parity_note = r"\ (\text{spin})" if (k % 2 == 0) else r"\ (\text{not spin})"
                rows.append((r"\text{Euler number}", rf"\pm {k}" + parity_note))
            else:
                rows.append((r"\text{(twisted) Euler number}", rf"\pm {k}"))
        else:
            if orientable:
                rows.append((r"\text{Euler class}", r"e \neq 0\ (\text{non-trivial})"))
                sp = getattr(restricted, "spin_on_this_complex", None)
                if sp is not None:
                    rows.append(
                        (r"\text{Spin class}",
                         r"w_2 = 0\ (\text{spin})" if bool(sp) else r"w_2 \neq 0\ (\text{not spin})")
                    )
            else:
                rows.append((r"\text{(twisted) Euler class}", r"\tilde{e} \neq 0"))

    if orientable:
        rows.append(
            (r"\text{Euler rounding dist.}",
             r"d_\infty(\delta\theta, e) = " + f"{rounding_dist:.3g}")
        )
    else:
        rows.append(
            (r"\text{Euler rounding dist.}",
             r"d_\infty(\delta_\omega\theta, \tilde{e}) = " + f"{rounding_dist:.3g}")
        )

    def _rows(rs):
        return r"\\[3pt]".join(
            r"\quad " + l + r" &:\quad " + r for l, r in rs
        )

    latex = (
        r"\begin{aligned}"
        r"\textbf{Characteristic Classes} & \\[6pt]"
        + _rows(rows)
        + r"\end{aligned}"
    )

    try:
        display(Math(latex))
        return True
    except Exception:
        return False


def _latex_escape_text(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash ")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("_", r"\_")
    )
