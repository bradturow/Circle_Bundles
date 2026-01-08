from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import networkx as nx

from .combinatorics import canon_edge

Edge = Tuple[int, int]


__all__ = [
    "get_cocycle_dict",
    "lift_base_points",
]


def get_cocycle_dict(
    G: nx.Graph,
    *,
    value_rule: str = "different_cluster",
    require_consistent: bool = True,
) -> Dict[Edge, int]:
    """
    Construct a Z2-valued Čech 1-cocycle dictionary from a fiberwise cluster graph G.

    Nodes: (fiber_index, cluster_label).
    Edges connect clusters across different fibers.

    Default rule:
      value on edge (j,k) is 1 iff the cluster labels differ (mu != nu), else 0.

    Returns
    -------
    cocycle_dict : dict[(j,k), int] with j<k, values in {0,1}

    Notes
    -----
    Multiple graph edges can map to the same fiber-pair (j,k). If they induce
    different values, that's either an error (default) or we take OR.
    """
    if value_rule not in {"different_cluster"}:
        raise ValueError("value_rule currently only supports 'different_cluster'.")

    cocycle: Dict[Edge, int] = {}

    for (j1, mu), (k1, nu) in G.edges():
        j1 = int(j1); k1 = int(k1)
        mu = int(mu); nu = int(nu)

        if j1 == k1:
            continue  # skip intra-fiber edges

        e = canon_edge(j1, k1)
        val = int(mu != nu)  # Z2

        if e not in cocycle:
            cocycle[e] = val
        else:
            if cocycle[e] != val:
                if require_consistent:
                    raise ValueError(
                        f"Inconsistent cocycle assignment on fiber-pair {e}: "
                        f"existing={cocycle[e]}, new={val} from edge ({(j1,mu)} <-> {(k1,nu)})"
                    )
                # If not requiring consistency, take OR (a mild "union" convention)
                cocycle[e] = int(bool(cocycle[e]) or bool(val))

    return cocycle


def lift_base_points(
    G: nx.Graph,
    cl: np.ndarray,
    base_points: np.ndarray,
    *,
    seed_fiber: int = 0,
    seed_cluster: int = 0,
    return_assigned: bool = False,
    eps: float = 1e-12,
    noise_label: int = -1,
    assume_two_clusters: bool = True,
    prefer_edge_shared_indices: bool = True,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Assign consistent signs to representative vectors of points in a double cover
    (e.g. S^2 -> RP^2). Output is a sign choice ± per point.

    Parameters
    ----------
    G : nx.Graph
        Cluster graph with nodes (fiber_index, cluster_label). If edges have
        attribute 'indices_shared', we use that to propagate faster/cleaner.
    cl : (n_fibers, n_points) int
        Fiberwise cluster labels; noise indicated by `noise_label`.
    base_points : (n_points, d) float
        Vector representative for each base point.
    seed_fiber, seed_cluster : ints
        Seed a starting cluster to orient.
    assume_two_clusters : bool
        If True, we apply the "other label opposite" rule in the seed fiber and
        enforce that each fiber has <= 2 non-noise labels (when present).

    Returns
    -------
    oriented : (n_points, d) float
        Flipped representatives.
    assigned : (n_points,) bool   (only if return_assigned=True)
        Which points were oriented during propagation.

    Notes
    -----
    - Noise points are ignored for seeding/propagation.
    - Orientation propagation uses a BFS over cluster nodes in G.
    """

    cl = np.asarray(cl)
    base_points = np.asarray(base_points, dtype=float)

    if cl.ndim != 2:
        raise ValueError(f"`cl` must be 2D (n_fibers, n_points). Got cl.ndim={cl.ndim}.")
    n_fibers, n_points = cl.shape

    if base_points.ndim != 2 or base_points.shape[0] != n_points:
        raise ValueError(
            f"`base_points` must have shape (n_points, d). "
            f"Got {base_points.shape}, expected ({n_points}, d)."
        )

    seed_fiber = int(seed_fiber)
    seed_cluster = int(seed_cluster)
    if not (0 <= seed_fiber < n_fibers):
        raise ValueError(f"seed_fiber={seed_fiber} out of range [0,{n_fibers-1}]")
    if seed_cluster == noise_label:
        raise ValueError("seed_cluster cannot be the noise label.")

    oriented = np.zeros_like(base_points)
    assigned = np.zeros(n_points, dtype=bool)

    def _orient_to_ref(v: np.ndarray, r: np.ndarray, want_positive: bool = True) -> np.ndarray:
        dot = float(np.dot(v, r))
        if abs(dot) <= float(eps):
            return v  # ambiguous; don't flip
        ok = (dot > 0) if want_positive else (dot < 0)
        return v if ok else -v

    # ---------- seed cluster ----------
    seed_idx = np.where(cl[seed_fiber] == seed_cluster)[0]
    if seed_idx.size == 0:
        raise ValueError(f"Seed cluster (fiber={seed_fiber}, cluster={seed_cluster}) is empty.")

    ref_idx = int(seed_idx[0])
    ref_vec = base_points[ref_idx]
    oriented[ref_idx] = ref_vec
    assigned[ref_idx] = True

    for idx in seed_idx[1:]:
        idx = int(idx)
        oriented[idx] = _orient_to_ref(base_points[idx], ref_vec, want_positive=True)
        assigned[idx] = True

    # ---------- two-sheet rule in seed fiber ----------
    seed_labels = [int(x) for x in np.unique(cl[seed_fiber]) if int(x) != noise_label]
    if assume_two_clusters and len(seed_labels) > 2:
        raise ValueError(
            f"assume_two_clusters=True but seed fiber has {len(seed_labels)} non-noise labels: {seed_labels}"
        )

    if assume_two_clusters:
        for other_label in seed_labels:
            if other_label == seed_cluster:
                continue
            other_idx = np.where(cl[seed_fiber] == other_label)[0]
            for idx in other_idx:
                idx = int(idx)
                oriented[idx] = _orient_to_ref(base_points[idx], ref_vec, want_positive=False)
                assigned[idx] = True

    # ---------- BFS over cluster nodes ----------
    q: deque[Tuple[Tuple[int, int], int]] = deque()
    visited: set[Tuple[int, int]] = set()

    # seed BFS with the seed fiber's non-noise clusters that exist in G
    for mu in seed_labels:
        node = (seed_fiber, int(mu))
        if node not in G:
            continue
        idxs = np.where(cl[seed_fiber] == mu)[0]
        if idxs.size == 0:
            continue
        idx0 = int(idxs[0])
        if assigned[idx0]:
            q.append((node, idx0))
            visited.add(node)

    while q:
        (j, mu), j_ref_idx = q.popleft()
        j_ref_vec = oriented[int(j_ref_idx)]

        for node_k in G.neighbors((j, mu)):
            k, nu = int(node_k[0]), int(node_k[1])
            if nu == noise_label:
                continue
            if node_k in visited:
                continue

            # choose a shared index to anchor orientation
            shared = None
            if prefer_edge_shared_indices and G.has_edge((j, mu), node_k):
                shared = G.edges[(j, mu), node_k].get("indices_shared", None)

            if shared is None:
                # fallback: intersect membership lists (slower)
                idx_j = np.where(cl[j] == mu)[0]
                idx_k = np.where(cl[k] == nu)[0]
                shared = np.intersect1d(idx_j, idx_k, assume_unique=False)

            if len(shared) == 0:
                continue

            k_ref_idx = int(np.asarray(shared, dtype=int)[0])
            oriented[k_ref_idx] = _orient_to_ref(base_points[k_ref_idx], j_ref_vec, want_positive=True)
            assigned[k_ref_idx] = True
            k_ref_vec = oriented[k_ref_idx]

            # orient all points in (k,nu)
            idx_k = np.where(cl[k] == nu)[0]
            if assume_two_clusters:
                # optional safety: enforce <=2 labels on encountered fibers (when present)
                k_labels = [int(x) for x in np.unique(cl[k]) if int(x) != noise_label]
                if len(k_labels) > 2:
                    raise ValueError(
                        f"assume_two_clusters=True but encountered fiber {k} has {len(k_labels)} labels: {k_labels}"
                    )

            for idx in idx_k:
                idx = int(idx)
                if assigned[idx]:
                    continue
                oriented[idx] = _orient_to_ref(base_points[idx], k_ref_vec, want_positive=True)
                assigned[idx] = True

            visited.add(node_k)
            q.append((node_k, k_ref_idx))

    if return_assigned:
        return oriented, assigned
    return oriented
