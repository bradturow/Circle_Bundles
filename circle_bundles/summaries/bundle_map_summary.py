# circle_bundles/summaries/bundle_map_summary.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BundleMapSummary:
    """
    Summary wrapper for the bundle-map solver output.

    Mirrors the style of the other summary objects:
      - always carries a plain-text summary
      - can render a polished table/visual summary via show_summary()
    """
    report: Any
    meta: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""

    def _ambient_dim_row(self) -> Optional[Tuple[str, str]]:
        """
        Try to infer an ambient dimension display row from meta.
        Expected meta keys (prefer earlier):
          - "ambient_dim"  (recommended)
          - "D_used"
          - "D"
        """
        for key in ("ambient_dim", "D_used", "D"):
            if key in self.meta and self.meta[key] is not None:
                try:
                    D = int(self.meta[key])
                    return (
                        r"\text{Ambient dimension}",
                        rf"{D}",
                    )
                except Exception:
                    return (
                        r"\text{Ambient dimension}",
                        rf"\text{{{self.meta[key]}}}",
                    )
        return None

    def show_summary(
        self,
        *,
        show: bool = True,
        # Keep these parameters only for backwards compatibility;
        # we intentionally force the uniform "auto" behavior now.
        mode: str = "auto",
        rounding: int = 3,
        extra_rows: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Render the bundle-map summary via the shared pretty renderer.

        Notes
        -----
        - We intentionally enforce the uniform behavior used elsewhere:
            mode="auto" and rounding=3.
        - We automatically append an "Ambient dimension" row when available in meta.
        """
        # Import lazily to avoid any import cycles / optional deps
        from ..trivializations.bundle_map import show_bundle_map_summary

        rows: List[Tuple[str, str]] = []
        if extra_rows:
            rows.extend(list(extra_rows))

        amb = self._ambient_dim_row()
        if amb is not None:
            rows.append(amb)

        return show_bundle_map_summary(
            self.report,
            show=bool(show),
            mode="auto",      # force
            rounding=3,       # force
            extra_rows=(rows if rows else None),
        )


def summarize_bundle_map(
    report: Any,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> BundleMapSummary:
    """
    Create a BundleMapSummary object from a solver report + meta.
    """
    text = ""
    try:
        if hasattr(report, "to_text") and callable(getattr(report, "to_text")):
            # Use default formatting; we keep this as a stable snippet
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
