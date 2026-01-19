# combinatorics.py
from typing import Tuple

Edge = Tuple[int, int]
Tri  = Tuple[int, int, int]
Tet = Tuple[int, int, int, int]

def canon_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)

def canon_tri(a: int, b: int, c: int) -> Tri:
    return tuple(sorted((int(a), int(b), int(c))))

def canon_tet(a: int, b: int, c: int, d: int) -> Tet:
    return tuple(sorted((int(a), int(b), int(c), int(d))))
