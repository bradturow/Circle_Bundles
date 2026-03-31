# tests/test_synthetic.py
"""Tests for circle_bundles.synthetic data generators."""
from __future__ import annotations

import numpy as np
import pytest

from circle_bundles.synthetic import (
    sample_sphere,
    hopf_projection,
    sample_s2_trivial,
    sample_R3_torus,
    sample_foldy_klein_bottle,
)
from circle_bundles.synthetic.tori_and_kb import sample_C2_torus


# ---------------------------------------------------------------------------
# sample_sphere
# ---------------------------------------------------------------------------

class TestSampleSphere:
    """Tests for uniform sphere sampling via Gaussian normalization."""

    def test_shape_s2(self, rng):
        pts = sample_sphere(100, dim=2, rng=rng)
        assert pts.shape == (100, 3)

    def test_shape_s3(self, rng):
        pts = sample_sphere(200, dim=3, rng=rng)
        assert pts.shape == (200, 4)

    def test_shape_s1(self, rng):
        pts = sample_sphere(50, dim=1, rng=rng)
        assert pts.shape == (50, 2)

    def test_unit_norms(self, rng):
        """All returned points should lie on the unit sphere."""
        pts = sample_sphere(500, dim=3, rng=rng)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_reproducibility(self):
        """Same seed should give identical output."""
        a = sample_sphere(50, dim=2, rng=np.random.default_rng(0))
        b = sample_sphere(50, dim=2, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(a, b)

    def test_invalid_n_raises(self, rng):
        with pytest.raises(ValueError, match="positive"):
            sample_sphere(0, dim=2, rng=rng)

    def test_invalid_dim_raises(self, rng):
        with pytest.raises(ValueError, match="dim"):
            sample_sphere(10, dim=-1, rng=rng)


# ---------------------------------------------------------------------------
# hopf_projection
# ---------------------------------------------------------------------------

class TestHopfProjection:
    """Tests for the Hopf fibration S^3 -> S^2."""

    def test_output_shape(self, s3_points):
        base = hopf_projection(s3_points)
        assert base.shape == (s3_points.shape[0], 3)

    def test_output_on_s2(self, s3_points):
        """Hopf projection should map S^3 to S^2."""
        base = hopf_projection(s3_points)
        norms = np.linalg.norm(base, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_fiber_invariance(self, rng):
        """Points on the same Hopf fiber should project to the same base point.

        The Hopf projection is q -> q v q^{-1} with v = i (default).
        The fiber action is right-multiplication by e^{i*t}, where
        q = z1 + z2*j with z1 = a+ib, z2 = c+id.

        Using the quaternion identity j * e^{it} = e^{-it} * j:
            (z1 + z2*j) * e^{it} = z1*e^{it} + z2*e^{-it}*j

        In real coordinates:
            (a', b') = R(+t) * (a, b)
            (c', d') = R(-t) * (c, d)
        """
        q = sample_sphere(1, dim=3, rng=rng)  # (1, 4)
        a, b, c, d = q[0]

        t = np.pi / 3
        ct, st = np.cos(t), np.sin(t)
        # Right-multiply by e^{it}: R(+t) on (a,b), R(-t) on (c,d)
        q_rot = np.array([[
            a * ct - b * st,
            a * st + b * ct,
            c * ct + d * st,
            -c * st + d * ct,
        ]])

        base_q = hopf_projection(q)
        base_rot = hopf_projection(q_rot)
        np.testing.assert_allclose(base_q, base_rot, atol=1e-10)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            hopf_projection(np.ones((10, 5)))

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            hopf_projection(np.ones(4))


# ---------------------------------------------------------------------------
# sample_s2_trivial  (product bundle S^2 x S^1)
# ---------------------------------------------------------------------------

class TestSampleS2Trivial:
    def test_shapes(self, rng):
        data, base, angles = sample_s2_trivial(200, rng=rng)
        assert data.shape == (200, 5)
        assert base.shape == (200, 3)
        assert angles.shape == (200,)

    def test_base_on_s2(self, rng):
        _, base, _ = sample_s2_trivial(200, rng=rng)
        norms = np.linalg.norm(base, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_angles_in_range(self, rng):
        _, _, angles = sample_s2_trivial(200, rng=rng)
        assert np.all(angles >= 0.0)
        assert np.all(angles < 2 * np.pi)


# ---------------------------------------------------------------------------
# Torus samplers
# ---------------------------------------------------------------------------

class TestSampleC2Torus:
    def test_shape(self, rng):
        data, base, alpha = sample_C2_torus(300, rng=rng)
        assert data.shape == (300, 4)
        assert base.shape == (300, 2)
        assert alpha.shape == (300,)

    def test_base_on_s1(self, rng):
        _, base, _ = sample_C2_torus(300, rng=rng)
        norms = np.linalg.norm(base, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_invalid_n(self, rng):
        with pytest.raises(ValueError, match="positive"):
            sample_C2_torus(0, rng=rng)


class TestSampleR3Torus:
    def test_shape(self, rng):
        data, base, alpha = sample_R3_torus(300, rng=rng)
        assert data.shape == (300, 3)
        assert base.shape == (300, 2)
        assert alpha.shape == (300,)

    def test_base_on_s1(self, rng):
        _, base, _ = sample_R3_torus(300, rng=rng)
        norms = np.linalg.norm(base, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_ring_torus_violation_raises(self, rng):
        """If r_center > R, the ring torus condition fails."""
        with pytest.raises(ValueError, match="ring_torus"):
            sample_R3_torus(100, R=1.0, r_center=2.0, r_amplitude=0.0, rng=rng)


# ---------------------------------------------------------------------------
# Klein bottle sampler
# ---------------------------------------------------------------------------

class TestSampleFoldyKleinBottle:
    def test_shape(self, rng):
        X, t = sample_foldy_klein_bottle(200, rng=rng)
        assert X.shape[0] == 200
        assert X.ndim == 2
        assert t.shape == (200,)

    def test_base_param_range(self, rng):
        _, t = sample_foldy_klein_bottle(200, rng=rng)
        assert np.all(t >= 0.0)
        assert np.all(t < 2 * np.pi)
