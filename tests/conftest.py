# tests/conftest.py
"""Shared fixtures for circle_bundles test suite."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Deterministic NumPy random generator for reproducible tests."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Small point clouds used across multiple test modules
# ---------------------------------------------------------------------------

@pytest.fixture
def s3_points(rng):
    """500 points sampled uniformly on S^3 (in R^4)."""
    from circle_bundles.synthetic import sample_sphere
    return sample_sphere(500, dim=3, rng=rng)


@pytest.fixture
def s2_points(rng):
    """500 points sampled uniformly on S^2 (in R^3)."""
    from circle_bundles.synthetic import sample_sphere
    return sample_sphere(500, dim=2, rng=rng)
