"""Regression tests for optional imports and Matplotlib compatibility."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import circle_bundles as cb
from circle_bundles.base_covers import (
    NerveSummary as CoverNerveSummary,
    plot_cover_summary_boxplot,
)
from circle_bundles.summaries.nerve_summary import (
    NerveSummary,
    plot_nerve_summary_boxplot,
)


def test_public_api_contains_notebook_helpers():
    """Optional installations must not change the documented top-level API."""
    expected = {
        "get_patch_sample",
        "get_predominant_dirs",
        "make_star_pyramid",
        "sample_sphere",
        "show_data_vis",
    }

    assert expected <= set(cb.__all__)
    for name in expected:
        assert callable(getattr(cb, name))


def test_cover_summary_boxplot_supports_current_matplotlib():
    summary = CoverNerveSummary(
        n_sets=2,
        n_samples=3,
        n0=2,
        n1=1,
        n2=0,
        n3=0,
        vert_card=np.array([2, 3]),
        edge_card=np.array([1]),
    )

    fig, axes = plot_cover_summary_boxplot(summary, show=False)
    assert fig is not None
    assert axes is not None
    plt.close(fig)


def test_bundle_nerve_summary_boxplot_supports_current_matplotlib():
    summary = NerveSummary(
        n_sets=2,
        n_samples=3,
        n0=2,
        n1=1,
        n2=0,
        n3=0,
        vert_card=np.array([2, 3]),
        edge_card=np.array([1]),
    )

    fig, axes = plot_nerve_summary_boxplot(summary, show=False)
    assert fig is not None
    assert axes is not None
    plt.close(fig)
