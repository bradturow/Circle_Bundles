# tests/test_o2_cocycle.py
"""Tests for O(2) matrix helpers and cocycle utilities."""
from __future__ import annotations

import numpy as np
import pytest

from circle_bundles.o2_cocycle import (
    angles_to_unit,
    rotation_matrix,
    reflection_axis_matrix,
    r_from_det,
    project_to_O2,
    det_sign,
    decompose_O2_as_R_times_r,
)


# ---------------------------------------------------------------------------
# angles_to_unit
# ---------------------------------------------------------------------------

class TestAnglesToUnit:
    def test_zero_angle(self):
        out = angles_to_unit(np.array([0.0]))
        np.testing.assert_allclose(out, [[1.0, 0.0]], atol=1e-15)

    def test_pi_half(self):
        out = angles_to_unit(np.array([np.pi / 2]))
        np.testing.assert_allclose(out, [[0.0, 1.0]], atol=1e-15)

    def test_shape(self):
        out = angles_to_unit(np.linspace(0, 2 * np.pi, 50))
        assert out.shape == (50, 2)

    def test_unit_norms(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        out = angles_to_unit(theta)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-15)


# ---------------------------------------------------------------------------
# rotation_matrix
# ---------------------------------------------------------------------------

class TestRotationMatrix:
    def test_identity(self):
        R = rotation_matrix(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-15)

    def test_90_degrees(self):
        R = rotation_matrix(np.pi / 2)
        expected = np.array([[0, -1], [1, 0]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-15)

    def test_orthogonal(self):
        R = rotation_matrix(1.23)
        np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-15)

    def test_det_one(self):
        R = rotation_matrix(2.71)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-15)


# ---------------------------------------------------------------------------
# reflection_axis_matrix
# ---------------------------------------------------------------------------

class TestReflectionAxisMatrix:
    def test_is_orthogonal(self):
        r = reflection_axis_matrix(0.5)
        np.testing.assert_allclose(r.T @ r, np.eye(2), atol=1e-15)

    def test_det_minus_one(self):
        r = reflection_axis_matrix(0.5)
        np.testing.assert_allclose(np.linalg.det(r), -1.0, atol=1e-15)

    def test_involution(self):
        """Reflection is its own inverse: r^2 = I."""
        r = reflection_axis_matrix(1.7)
        np.testing.assert_allclose(r @ r, np.eye(2), atol=1e-15)

    def test_ref_angle_zero(self):
        """ref_angle=0 should reflect across the x-axis: (x,y) -> (x,-y)."""
        r = reflection_axis_matrix(0.0)
        expected = np.array([[1, 0], [0, -1]], dtype=float)
        np.testing.assert_allclose(r, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# r_from_det
# ---------------------------------------------------------------------------

class TestRFromDet:
    def test_det_plus_one(self):
        np.testing.assert_array_equal(r_from_det(+1, 0.0), np.eye(2))

    def test_det_minus_one(self):
        r = r_from_det(-1, 0.5)
        np.testing.assert_allclose(np.linalg.det(r), -1.0, atol=1e-15)

    def test_invalid_det_raises(self):
        with pytest.raises(ValueError, match="det must be"):
            r_from_det(0, 0.0)


# ---------------------------------------------------------------------------
# project_to_O2
# ---------------------------------------------------------------------------

class TestProjectToO2:
    def test_already_orthogonal(self):
        R = rotation_matrix(0.7)
        proj = project_to_O2(R)
        np.testing.assert_allclose(proj, R, atol=1e-12)

    def test_noisy_matrix(self):
        """A slightly perturbed rotation should project back to O(2)."""
        R = rotation_matrix(1.5)
        noisy = R + 0.01 * np.array([[0.3, -0.1], [0.2, 0.4]])
        proj = project_to_O2(noisy)
        np.testing.assert_allclose(proj.T @ proj, np.eye(2), atol=1e-12)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="2x2"):
            project_to_O2(np.eye(3))


# ---------------------------------------------------------------------------
# det_sign
# ---------------------------------------------------------------------------

class TestDetSign:
    def test_rotation(self):
        assert det_sign(rotation_matrix(1.0)) == 1

    def test_reflection(self):
        assert det_sign(reflection_axis_matrix(0.0)) == -1


# ---------------------------------------------------------------------------
# decompose_O2_as_R_times_r
# ---------------------------------------------------------------------------

class TestDecomposeO2:
    def test_pure_rotation(self):
        """A pure rotation should decompose as (theta, +1)."""
        theta_in = 1.23
        R = rotation_matrix(theta_in)
        theta_out, det_out = decompose_O2_as_R_times_r(R, ref_angle=0.0)
        assert det_out == 1
        np.testing.assert_allclose(theta_out, theta_in, atol=1e-12)

    def test_pure_reflection(self):
        """A reflection should decompose with det = -1."""
        r = reflection_axis_matrix(0.0)
        _, det_out = decompose_O2_as_R_times_r(r, ref_angle=0.0)
        assert det_out == -1

    def test_roundtrip(self):
        """O = R(theta) * r(ref) should decompose back to the same theta and det."""
        ref = 0.7
        theta_in = 2.1
        O = rotation_matrix(theta_in) @ reflection_axis_matrix(ref)
        theta_out, det_out = decompose_O2_as_R_times_r(O, ref_angle=ref)
        assert det_out == -1
        np.testing.assert_allclose(theta_out, theta_in % (2 * np.pi), atol=1e-12)

    def test_non_orthogonal_raises(self):
        with pytest.raises(ValueError, match="orthogonal"):
            decompose_O2_as_R_times_r(np.array([[2.0, 0.0], [0.0, 1.0]]))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="2x2"):
            decompose_O2_as_R_times_r(np.eye(3))
