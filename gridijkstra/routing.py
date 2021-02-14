from typing import Optional, List, Tuple, Union, Dict
import scipy.sparse as scs
import numpy as np


def _transform_node_input(nodes):
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes)
    if nodes.dtype == np.bool:
        # Sources provided as a boolean map. Should cast these to indices (Nx2)
        nodes = np.argwhere(nodes)
    elif nodes.ndim == 1:
        nodes = nodes.reshape(1, 2)
    else:
        assert nodes.ndim == 2
        assert nodes.shape[1] == 2  # For now, we restrict to 2D
        # Perhaps just return if sources.size == 0?
    return nodes


def plan(costs: Union[np.ndarray, Dict[Tuple[int, int]], np.ndarray],
         indices_start: np.ndarray,
         indices_target: np.ndarray,
         stencil: Optional[List[Tuple[int, int]]] = None,
         return_length: bool = True,
         return_total_cost: bool = False):
    """
    Dijkstra's algorithm (scipy) for grid-based-graphs
    :param costs: Traversal costs. If an numpy.ndarray is provided, costs[i, j] represents the traversal intensity
        (cost-per-length) of traversing from (i, j) to either of its neighbors (defined by 'stencil'). If a dict is
        provided, the keys are a (di, dj) pair and the values are traversal intensity maps such that
        costs[(di, dj)][i, j] is the cost of traversing from (i, j) to (i + di, j + dj). Assuming a ni x nj traversal
        grid, the dimensionality of the intensity maps should be ni x nj.
    :param indices_start: numpy.ndarray-like. Compute the paths from these indices. Shape must be either (2,) or (N, 2),
        with N being the number of paths to compute.
    :param indices_target: numpy.ndarray-like. Compute the paths to these indices. Similar to indices_start. Must have
        the same shape as indices_start.
    :param stencil: Only used if 'costs' is provided as an numpy.ndarray. Defines the neighborhood stencil as a list of
        tuples. Defaults to [(0, 1), (1, 0), (-1, 0), (0, -1)].
    :param return_length:
    :param return_total_cost:
    :return:
    """
    # TODO: default stencil when costs is provided
    # Combinations are not assumed, only pairwise paths:
    assert len(indices_start) == len(indices_target)  # TODO: Add option to extend with all combinations
    indices_start = _transform_node_input(indices_start)
    indices_target = _transform_node_input(indices_target)

    # Handle costs
    if isinstance(costs, np.ndarray):
        # Costs are provided as an intensity map, we need the stencil though.
        assert stencil is not None
        costs = {s: costs for s in stencil}
    else:
        assert isinstance(costs, dict)
        # If stencil is provided, warn that it will not be used

    # Define index transformation functions
    ni, nj = next(iter(costs.values())).shape  # TODO: verify that all costs-elements have same shape
    ns = len(costs)

    def _to_ij(_q):
        return np.unravel_index(_q, shape=(ni, nj))

    def _from_ij(_i, _j):
        return np.ravel_multi_index((_i, _j), dims=(ni, nj))

    # Validate input
    # 1. Check that the dimension of the cost arrays and the provided sources and destinations are consistent.
    # Warn/raise if that is not the case
    # 2. Check values of costs. Non-positive values are not allowed

    # Execute path planning (scipy-based)

    # Find all valid nodes and edges
    # TODO: if costs is inf or nan: ignore
    g0_ij = np.repeat(np.argwhere(np.ones((ni, nj))), ns, axis=0)
    g1_ij = g0_ij.copy()
    g_costs = np.zeros(g0_ij.shape[0])
    for i, s in enumerate(costs):
        g1_ij[i::ns] += s
        # Costs are interpreted as intensity, thus we need to scale with stencil length
        g_costs[i::ns] = costs[s].flatten() * np.sqrt(s[0] ** 2 + s[1] ** 2)

    invalid = np.any(g0_ij < 0, axis=1)
    invalid |= np.any(g1_ij < 0, axis=1)
    invalid |= (g0_ij[:, 0] >= ni)
    invalid |= (g0_ij[:, 1] >= nj)
    invalid |= (g1_ij[:, 0] >= ni)
    invalid |= (g1_ij[:, 1] >= nj)

    # Filter invalid edges
    g0_ij = g0_ij[~invalid]
    g1_ij = g1_ij[~invalid]
    g_costs = g_costs[~invalid]

    # Convert to flat index
    g0_q = _from_ij(g0_ij[:, 0], g0_ij[:, 1])
    g1_q = _from_ij(g1_ij[:, 0], g1_ij[:, 1])
    sources_q = _from_ij(indices_start[:, 0], indices_start[:, 1])
    destinations_q = _from_ij(indices_target[:, 0], indices_target[:, 1])

    # Define traversal graph
    g = scs.coo_matrix((g_costs, (g0_q, g1_q)), shape=(ni * nj, ni * nj))
    total_costs, predecessors = scs.csgraph.dijkstra(g, indices=sources_q, return_predecessors=True)

    # Reconstruct paths
    paths = []
    for i, (s, d) in enumerate(zip(sources_q, destinations_q)):
        p = [d]
        while p[-1] != -9999 and p[-1] != s:
            p.append(predecessors[i, p[-1]])
        paths.append(p[::-1])

    # Convert paths to ij indices
    paths_ij = [np.array(_to_ij(p)).T for p in paths]

    # Pack output
    output = [paths_ij]
    if return_length:
        lengths = [np.sum(np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))) for p in paths_ij]
        output.append(lengths)
    if return_total_cost:
        output.append(total_costs[np.arange(destinations_q.size), destinations_q])

    if indices_start.shape[0] == 1:
        output = [ou[0] for ou in output]

    return tuple(output)
