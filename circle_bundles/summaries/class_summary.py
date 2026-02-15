"""
Uniform summary for characteristic classes + persistence (Bundle.get_classes output).

This is the *modern* summary used by Bundle.summarize_classes(), and should be the only
summary shown in the docs for class/persistence output.

Design goals (Feb 2026)
----------------------
- Summarize characteristic classes + persistence cleanly.
- Do NOT print local-trivialization diagnostics.
- Stable plain-text summary for logging / tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ClassSummary",
    "summarize_classes_and_persistence",
]

Edge = Tuple[int, int]


# ============================================================
# Helpers
# ============================================================

def _canon_edge(a: int, b: int) -> Edge:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _safe_bool(x: Any) -> Optional[bool]:
    try:
        return None if x is None else bool(x)
    except Exception:
        return None


def _fmt_float_or_dash(x: Any, *, decimals: int = 4) -> str:
    try:
        xf = float(x)
        return "—" if not np.isfinite(xf) else f"{xf:.{decimals}f}"
    except Exception:
        return "—"


def _get_euler_rounding_distance(restricted: Any) -> Optional[float]:
    try:
        v = float(getattr(restricted, "rounding_dist", None))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _euler_not_cocycle_on_full_complex(persistence: Any) -> bool:
    """
    True iff the twisted Euler representative fails to be a cocycle on the full nerve.
    This is detected via Euler cobirth > 0.
    """
    try:
        cob = persistence.twisted_euler["cobirth"]
        return int(cob.k_removed) > 0
    except Exception:
        return False


# ============================================================
# Summary container
# ============================================================

@dataclass
class ClassSummary:
    reps: Any
    restricted: Any
    persistence: Any
    summary_text: str
    euler_rounding_distance: Optional[float] = None
    euler_not_cocycle_full: bool = False

    def show_summary(
        self,
        *,
        show_classes: bool = True,
        show_persistence: bool = True,
        show_rounding_distance: bool = False,
    ) -> str:
        if not (show_classes or show_persistence):
            return ""

        text = self.summary_text

        did_latex = False
        if show_classes:
            did_latex = self._display_latex(
                show_rounding_distance=show_rounding_distance
            )

        if not did_latex:
            print("\n" + text + "\n")

        return text

    # ----------------------------
    # LaTeX rendering
    # ----------------------------

    def _display_latex(self, *, show_rounding_distance: bool) -> bool:
        try:
            from IPython.display import display, Math  # type: ignore
        except Exception:
            return False

        r = self.restricted
        reps = self.reps

        orientable = _safe_bool(getattr(r, "orientable", getattr(reps, "orientable", None)))
        w1_cob = getattr(r, "sw1_is_coboundary_Z2", None)
        if w1_cob is None and orientable is not None:
            w1_cob = bool(orientable)

        e_is_cob = getattr(r, "euler_is_coboundary_Z", None)
        eZ = getattr(r, "twisted_euler_number_Z", None)
        spin = getattr(r, "spin_on_this_complex", None)
        nT = getattr(r, "n_triangles", 0)

        rows: List[Tuple[str, str]] = []

        # warning
        if self.euler_not_cocycle_full:
            rows.append(
                (
                    r"\text{Warning}",
                    r"\text{Euler representative is not a cocycle on the full nerve}",
                )
            )

        # w1
        if w1_cob is None:
            w1_tex = r"w_1\ \text{(unknown)}"
        else:
            w1_tex = r"w_1 = 0\ (\text{orientable})" if bool(w1_cob) else r"w_1 \neq 0\ (\text{non-orientable})"
        rows.append((r"\text{Stiefel--Whitney}", w1_tex))

        # Euler
        if int(nT or 0) == 0:
            rows.append((r"\text{Euler class}", r"e = 0\ \text{(no 2-simplices)}"))
        elif e_is_cob is True:
            rows.append(
                (
                    r"\text{Euler class}" if orientable else r"\text{(twisted) Euler}",
                    r"0\ (\text{trivial})",
                )
            )
        elif eZ is not None:
            k = abs(int(eZ))
            if orientable:
                parity = r"\ (\text{spin})" if (k % 2 == 0) else r"\ (\text{not spin})"
                rows.append((r"\text{Euler number}", rf"\pm {k}{parity}"))
            else:
                rows.append((r"\text{(twisted) Euler number}", rf"\pm {k}"))
        else:
            rows.append(
                (
                    r"\text{Euler class}" if orientable else r"\text{(twisted) Euler}",
                    r"\neq 0",
                )
            )

        if spin is not None and orientable:
            rows.append((r"\text{Spin}", r"\text{True}" if spin else r"\text{False}"))

        if show_rounding_distance:
            rd = self.euler_rounding_distance
            rows.append(
                (
                    r"\text{Euler rounding dist.}",
                    r"\text{—}" if rd is None else f"{float(rd):.6g}",
                )
            )

        def _rows(rr):
            return r"\\[4pt]".join(r"\quad " + a + r" &:\quad " + b for a, b in rr)

        latex = (
            r"\begin{aligned}"
            r"\textbf{Characteristic Classes} & \\[8pt]"
            + _rows(rows)
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
    rd = _get_euler_rounding_distance(restricted)
    warn = _euler_not_cocycle_on_full_complex(persistence)

    IND = "  "
    LABEL = 30

    def _t(label: str, value: str) -> str:
        return f"{IND}{label:<{LABEL}} {value}"

    lines: List[str] = []
    lines.append("=== Characteristic Classes ===")

    if warn:
        lines.append(_t("WARNING:", "Euler rep is not a cocycle on the full nerve"))

    w1_cob = getattr(restricted, "sw1_is_coboundary_Z2", None)
    orientable = bool(getattr(restricted, "orientable", False))

    lines.append(
        _t(
            "Stiefel–Whitney:",
            "w₁ = 0 (orientable)" if w1_cob else "w₁ ≠ 0 (non-orientable)",
        )
    )

    nT = int(getattr(restricted, "n_triangles", 0))
    e_is_cob = getattr(restricted, "euler_is_coboundary_Z", None)
    eZ = getattr(restricted, "twisted_euler_number_Z", None)

    if nT == 0:
        lines.append(_t("Euler class:", "0 (no 2-simplices)"))
    elif e_is_cob:
        lines.append(_t("Euler class:", "0 (trivial)"))
    elif eZ is not None:
        k = abs(int(eZ))
        if orientable:
            parity = " (spin)" if (k % 2 == 0) else " (not spin)"
            lines.append(_t("Euler number:", f"±{k}{parity}"))
        else:
            lines.append(_t("(twisted) Euler #:", f"±{k}"))
    else:
        lines.append(_t("Euler class:", "non-trivial"))

    return ClassSummary(
        reps=reps,
        restricted=restricted,
        persistence=persistence,
        summary_text="\n".join(lines),
        euler_rounding_distance=rd,
        euler_not_cocycle_full=warn,
    )
