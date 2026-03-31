# tests/test_covers.py
"""Tests for cover construction utilities."""
from __future__ import annotations

import numpy as np
import pytest

from circle_bundles.covers.covers import CoverData, get_metric_ball_cover
from circle_bundles.covers.fibonacci_covers import (
    fibonacci_sphere,
    hemisphere_rep,
    get_s2_fibonacci_cover,
)


# ---------------------------------------------------------------------------
# CoverData
# ---------------------------------------------------------------------------

class TestCoverData:
    def test_construction(self):
        U = np.ones((3, 10), dtype=bool)
        cover = CoverData(U=U)
        assert cover.U.shape == (3, 10)
        assert cover.pou is None
        assert cover.landmarks is None

    def test_frozen(self):
        cover = CoverData(U=np.ones((2, 5), dtype=bool))
        with pytest.raises(AttributeError):
            cover.U = np.zeros((2, 5), dtype=bool)


# ---------------------------------------------------------------------------
# get_metric_ball_cover
# ---------------------------------------------------------------------------

class TestMetricBallCover:
    def test_shapes(self, rng):
        base = rng.normal(size=(100, 3))
        landmarks = rng.normal(size=(10, 3))
        cover = get_metric_ball_cover(base, landmarks, radius=5.0)
        assert cover.U.shape == (10, 100)
        assert cover.pou.shape == (10, 100)
        assert cover.landmarks.shape == (10, 3)

    def test_U_is_boolean(self, rng):
        base = rng.normal(size=(50, 2))
        landmarks = rng.normal(size=(5, 2))
        cover = get_metric_ball_cover(base, landmarks, radius=3.0)
        assert cover.U.dtype == bool

    def test_pou_sums_to_one(self, rng):
        """Partition of unity should sum to 1 for all covered points."""
        base = rng.normal(size=(80, 2))
        landmarks = rng.normal(size=(8, 2))
        cover = get_metric_ball_cover(base, landmarks, radius=5.0)
        covered = cover.U.any(axis=0)
        pou_sum = cover.pou[:, covered].sum(axis=0)
        np.testing.assert_allclose(pou_sum, 1.0, atol=1e-12)

    def test_pou_nonnegative(self, rng):
        base = rng.normal(size=(50, 3))
        landmarks = rng.normal(size=(5, 3))
        cover = get_metric_ball_cover(base, landmarks, radius=5.0)
        assert np.all(cover.pou >= 0.0)

    def test_dim_mismatch_raises(self):
        base = np.ones((10, 2))
        landmarks = np.ones((5, 3))
        with pytest.raises(ValueError, match="mismatch"):
            get_metric_ball_cover(base, landmarks, radius=1.0)


# ---------------------------------------------------------------------------
# fibonacci_sphere
# ---------------------------------------------------------------------------

class TestFibonacciSphere:
    def test_shape(self):
        pts = fibonacci_sphere(100)
        assert pts.shape == (100, 3)

    def test_unit_norms(self):
        pts = fibonacci_sphere(200)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_deterministic(self):
        a = fibonacci_sphere(50)
        b = fibonacci_sphere(50)
        np.testing.assert_array_equal(a, b)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            fibonacci_sphere(0)


# ---------------------------------------------------------------------------
# hemisphere_rep
# ---------------------------------------------------------------------------

class TestHemisphereRep:
    def test_upper_hemisphere(self):
        v = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(hemisphere_rep(v), v)

    def test_lower_hemisphere_flipped(self):
        v = np.array([0.0, 0.0, -1.0])
        np.testing.assert_array_equal(hemisphere_rep(v), -v)

    def test_equator_tiebreak(self):
        v = np.array([1.0, 0.0, 0.0])
        rep = hemisphere_rep(v)
        # Should stay the same (positive x)
        np.testing.assert_array_equal(rep, v)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="3"):
            hemisphere_rep(np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# get_s2_fibonacci_cover
# ---------------------------------------------------------------------------

class TestS2FibonacciCover:
    def test_returns_cover_data(self, s2_points):
        cover = get_s2_fibonacci_cover(s2_points, n_vertices=30)
        assert isinstance(cover, CoverData)
        assert cover.U.shape[1] == s2_points.shape[0]
        assert cover.U.dtype == bool

    def test_full_coverage(self, s2_points):
        """Every point should be covered by at least one set."""
        cover = get_s2_fibonacci_cover(s2_points, n_vertices=30)
        assert np.all(cover.U.any(axis=0))
