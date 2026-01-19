from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Set

try:
    import gudhi
except ImportError as e:
    raise ImportError("This function requires `gudhi`. Install with `pip install gudhi`.") from e

from .combinatorics import Edge, Tri, canon_edge, canon_tri
from .bundle import MaxTrivialSubcomplex

def get_simplices(st, n):
    return [
        tuple(s)
        for s, _ in st.get_skeleton(n)
        if len(s) == n + 1
    ]

def max_trivial_to_simplex_tree(
    subcx: MaxTrivialSubcomplex,
    *,
    filtration_value: float = 0.0,
    include_isolated_vertices: Optional[Sequence[int]] = None,
) -> "gudhi.SimplexTree":
    """
    Build a Gudhi SimplexTree from a MaxTrivialSubcomplex.

    Inserts:
      - all vertices appearing in kept_edges or kept_triangles
      - all kept_edges
      - all kept_triangles

    Parameters
    ----------
    subcx : MaxTrivialSubcomplex
    filtration_value : float
        Filtration assigned to every inserted simplex (static complex).
    include_isolated_vertices : optional list of ints
        If you want to force certain vertices to be present even if they appear
        in no kept edge/triangle.

    Returns
    -------
    st : gudhi.SimplexTree
    """
    st = gudhi.SimplexTree()

    verts: Set[int] = set()

    # --- triangles (also collects their vertices) ---
    for t in subcx.kept_triangles:
        i, j, k = canon_tri(*t)
        st.insert([i, j, k], filtration=filtration_value)
        verts.update([i, j, k])

    # --- edges (also collects their vertices) ---
    for e in subcx.kept_edges:
        a, b = canon_edge(*e)
        st.insert([a, b], filtration=filtration_value)
        verts.update([a, b])

    # --- vertices ---
    if include_isolated_vertices is not None:
        verts.update(int(v) for v in include_isolated_vertices)

    for v in verts:
        st.insert([int(v)], filtration=filtration_value)

    st.initialize_filtration()
    return st
