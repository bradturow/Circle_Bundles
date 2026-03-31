# tests/test_metrics.py
"""Tests for vectorized metric objects."""
from __future__ import annotations

import numpy as np

from circle_bundles.metrics import (
    EuclideanMetric,
    S1AngleMetric,
    RP1AngleMetric,
)


# ---------------------------------------------------------------------------
# EuclideanMetric
# ---------------------------------------------------------------------------

class TestEuclideanMetric:
    def test_pairwise_shape(self):
        m = EuclideanMetric()
        X = np.random.default_rng(0).normal(size=(10, 3))
        Y = np.random.default_rng(1).normal(size=(7, 3))
        D = m.pairwise(X, Y)
        assert D.shape == (10, 7)

    def test_self_distance_zero(self):
        m = EuclideanMetric()
        X = np.random.default_rng(0).normal(size=(10, 3))
        D = m.pairwise(X)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_symmetric(self):
        m = EuclideanMetric()
        X = np.random.default_rng(0).normal(size=(8, 4))
        D = m.pairwise(X)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_non_negative(self):
        m = EuclideanMetric()
        X = np.random.default_rng(0).normal(size=(15, 2))
        D = m.pairwise(X)
        assert np.all(D >= -1e-15)

    def test_known_distance(self):
        m = EuclideanMetric()
        X = np.array([[0.0, 0.0]])
        Y = np.array([[3.0, 4.0]])
        D = m.pairwise(X, Y)
        np.testing.assert_allclose(D[0, 0], 5.0, atol=1e-12)

    def test_has_name(self):
        m = EuclideanMetric()
        assert isinstance(m.name, str)
        assert len(m.name) > 0


# ---------------------------------------------------------------------------
# S1AngleMetric
# ---------------------------------------------------------------------------

class TestS1AngleMetric:
    def test_same_angle(self):
        m = S1AngleMetric()
        X = np.array([[0.0], [1.0], [2.0]])
        D = m.pairwise(X)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_opposite_points(self):
        """Distance between 0 and pi on S^1 should be pi."""
        m = S1AngleMetric()
        X = np.array([[0.0]])
        Y = np.array([[np.pi]])
        D = m.pairwise(X, Y)
        np.testing.assert_allclose(D[0, 0], np.pi, atol=1e-12)

    def test_wrap_around(self):
        """Distance should wrap around: d(0, 2pi - 0.1) ~ 0.1."""
        m = S1AngleMetric()
        X = np.array([[0.0]])
        Y = np.array([[2 * np.pi - 0.1]])
        D = m.pairwise(X, Y)
        np.testing.assert_allclose(D[0, 0], 0.1, atol=1e-12)

    def test_symmetric(self):
        m = S1AngleMetric()
        X = np.array([[0.5], [1.5], [3.0], [5.0]])
        D = m.pairwise(X)
        np.testing.assert_allclose(D, D.T, atol=1e-12)


# ---------------------------------------------------------------------------
# RP1AngleMetric
# ---------------------------------------------------------------------------

class TestRP1AngleMetric:
    def test_same_angle(self):
        m = RP1AngleMetric()
        X = np.array([[0.5]])
        D = m.pairwise(X)
        np.testing.assert_allclose(D[0, 0], 0.0, atol=1e-12)

    def test_antipodal_identification(self):
        """On RP^1, angles differing by pi should have distance 0."""
        m = RP1AngleMetric()
        X = np.array([[0.0]])
        Y = np.array([[np.pi]])
        D = m.pairwise(X, Y)
        np.testing.assert_allclose(D[0, 0], 0.0, atol=1e-12)

    def test_max_distance(self):
        """Maximum distance on RP^1 should be pi/2."""
        m = RP1AngleMetric()
        X = np.array([[0.0]])
        Y = np.array([[np.pi / 2]])
        D = m.pairwise(X, Y)
        np.testing.assert_allclose(D[0, 0], np.pi / 2, atol=1e-12)
