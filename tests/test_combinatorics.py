# tests/test_combinatorics.py
"""Tests for nerve/combinatorics simplex canonicalization helpers."""
from __future__ import annotations

from circle_bundles.nerve.combinatorics import canon_edge, canon_tri, canon_tet


class TestCanonEdge:
    def test_already_sorted(self):
        assert canon_edge(1, 3) == (1, 3)

    def test_swap(self):
        assert canon_edge(5, 2) == (2, 5)

    def test_equal(self):
        assert canon_edge(4, 4) == (4, 4)


class TestCanonTri:
    def test_already_sorted(self):
        assert canon_tri(0, 1, 2) == (0, 1, 2)

    def test_unsorted(self):
        assert canon_tri(3, 1, 2) == (1, 2, 3)

    def test_reverse(self):
        assert canon_tri(9, 5, 1) == (1, 5, 9)


class TestCanonTet:
    def test_already_sorted(self):
        assert canon_tet(0, 1, 2, 3) == (0, 1, 2, 3)

    def test_unsorted(self):
        assert canon_tet(7, 2, 5, 0) == (0, 2, 5, 7)
