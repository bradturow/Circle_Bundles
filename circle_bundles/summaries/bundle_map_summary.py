# circle_bundles/summaries/bundle_map_summary.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BundleMapSummary:
    """
    Summary wrapper for the bundle-map solver output.

    This mirrors the style of the other summary objects:
      - always carries a plain-text summary
      - can render a polished table/visual summary via show_summary()
    """
    report: Any
    meta: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""

    def show_summary(
        self,
        *,
        show: bool = True,
        mode: str = "auto",
        rounding: int = 3,
        extra_rows: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Render the bundle-map summary using the existing pretty renderer.

        Parameters
        ----------
        show:
            Whether to display the summary.
        mode:
            "auto", "latex", or "text" (passed through to the renderer).
        rounding:
            Numeric rounding in the renderer.
        extra_rows:
            Optional extra (label, value) rows to append to the summary table.
        """
        # Import lazily to avoid any import cycles / optional deps
        from ..trivializations.bundle_map import show_bundle_map_summary

        return show_bundle_map_summary(
            self.report,
            show=bool(show),
            mode=str(mode),
            rounding=int(rounding),
            extra_rows=extra_rows,
        )


def summarize_bundle_map(
    report: Any,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> BundleMapSummary:
    """
    Create a BundleMapSummary object from a solver report + meta.
    """
    # Try to derive a stable plain-text snippet (never required to be perfect)
    text = ""
    try:
        # common patterns: report may have to_text(), summary_text, or __str__
        if hasattr(report, "to_text") and callable(getattr(report, "to_text")):
            text = str(report.to_text())
        elif hasattr(report, "summary_text"):
            text = str(getattr(report, "summary_text"))
        else:
            text = str(report)
    except Exception:
        text = ""

    return BundleMapSummary(
        report=report,
        meta=dict(meta or {}),
        summary_text=text,
    )
