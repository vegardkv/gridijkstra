from typing import Optional, List, Tuple, Union, Dict
import scipy.sparse as scs
import numpy as np


Costs = Union[
    np.ndarray,
    Dict[Tuple[int, int], np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]


def plan(
    costs: Costs,
    indices_start: Union[np.ndarray, List],
    indices_target: Union[np.ndarray, List],
    stencil: Optional[List[Tuple[int, int]]] = None,
    return_path: bool = False,
    return_length: bool = False
):
    """
    Dijkstra's algorithm (scipy) for grid-based-graphs
    :param costs: Traversal costs.
        numpy.ndarray: costs[i, j] represents the traversal intensity (cost-per-length) of traversing from (i, j) to
            either of its neighbors (defined by 'stencil').
        dict: keys are a (di, dj) pair and the values are traversal intensity maps such that costs[(di, dj)][i, j] is
            the cost of traversing from (i, j) to (i + di, j + dj). Assuming a ni x nj traversal grid, the
            dimensionality of the intensity maps should be ni x nj.
        tuple: TODO
    :param indices_start: numpy.ndarray-like. Compute the paths from these indices. Shape must be either (2,) or (N, 2),
        with N being the number of paths to compute.
    :param indices_target: numpy.ndarray-like. Compute the paths to these indices. Similar to indices_start. Must have
        the same shape as indices_start.
    :param stencil: Only used if 'costs' is provided as an numpy.ndarray. Defines the neighborhood stencil as a list of
        tuples. Defaults to [(0, 1), (1, 0), (-1, 0), (0, -1)].
    :param return_path:
    :param return_length:
    :return:
    """
    # Combinations are not assumed, only pairwise paths:
    assert len(indices_start) == len(indices_target)  # TODO: Add option to extend with all combinations
    indices_start = _transform_node_input(indices_start)
    indices_target = _transform_node_input(indices_target)

    r_graph = _RoutingGraph(costs, stencil)

    # Validate input
    # 1. Check that the dimension of the cost arrays and the provided sources and destinations are consistent.
    # Warn/raise if that is not the case
    # 2. Check values of costs. Non-positive values are not allowed

    # Execute path planning (scipy-based)
    sources_q = r_graph.from_ij(indices_start[:, 0], indices_start[:, 1])
    destinations_q = r_graph.from_ij(indices_target[:, 0], indices_target[:, 1])

    # Define traversal graph
    # TODO: handle duplicates in indices_start
    sq_uniq, sq_inv = np.unique(sources_q, return_inverse=True)
    if return_path is True or return_length is True:
        total_costs, predecessors = scs.csgraph.dijkstra(r_graph.graph, indices=sq_uniq, return_predecessors=True)
        # Reconstruct paths
        paths = []
        for i, (s, d) in enumerate(zip(sources_q, destinations_q)):
            p = [d]
            while p[-1] != -9999 and p[-1] != s:
                p.append(predecessors[sq_inv[i], p[-1]])
            paths.append(p[::-1])
        # Convert paths to ij indices
        paths_ij = [np.array(r_graph.to_ij(p)).T for p in paths]
    else:
        total_costs = scs.csgraph.dijkstra(r_graph.graph, indices=sq_uniq)
        paths_ij = None

    # Pack output
    output = [total_costs[sq_inv, destinations_q]]
    if return_path:
        output.append(paths_ij)
    if return_length:
        lengths = [np.sum(np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))) for p in paths_ij]
        output.append(lengths)

    # Reduce dimensionality if input was 1D
    if indices_start.shape[0] == 1:
        output = [ou[0] for ou in output]

    if return_length is False and return_path is False:
        return output[0]
    else:
        return tuple(output)


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


class _RoutingGraph:
    def __init__(self, costs, stencil):
        # Handle costs
        if isinstance(costs, tuple) and len(costs) == 3:
            g0_ij, g1_ij, g_costs = costs
            self._ni = max(np.max(g0_ij[:, 0]), np.max(g1_ij[:, 0]))
            self._nj = max(np.max(g0_ij[:, 1]), np.max(g1_ij[:, 1]))
            # TODO: check for duplicates
        else:
            if isinstance(costs, np.ndarray):
                # Costs are provided as an intensity map, we need the stencil though.
                if stencil is None:
                    stencil = [(0, 1), (1, 0), (-1, 0), (0, -1)]
                costs = {s: costs for s in stencil}
            else:
                assert isinstance(costs, dict)

            # Define index transformation functions
            # TODO: verify that all costs-elements have same shape
            self._ni, self._nj = next(iter(costs.values())).shape

            # Filter invalid edges
            g0_ij, g1_ij, g_costs = _extract_stencil_costs(costs, self._ni, self._nj)

        # Convert to flat index
        g0_q = self.from_ij(g0_ij[:, 0], g0_ij[:, 1])
        g1_q = self.from_ij(g1_ij[:, 0], g1_ij[:, 1])

        # Define traversal graph
        self.graph = scs.coo_matrix((g_costs, (g0_q, g1_q)), shape=(self._ni * self._nj, self._ni * self._nj))

    def to_ij(self, _q):
        return np.unravel_index(_q, shape=(self._ni, self._nj))

    def from_ij(self, _i, _j):
        return np.ravel_multi_index((_i, _j), dims=(self._ni, self._nj))


def _extract_stencil_costs(costs, ni, nj):
    # Find all valid nodes and edges
    g0_ij = []
    g1_ij = []
    g_costs = []
    for di, dj in costs.keys():
        can_leave = ~np.isnan(costs[(di, dj)])
        can_enter = np.roll(can_leave, shift=(-di, -dj), axis=(0, 1))
        for si in range(abs(di)):
            can_enter[si if np.sign(di) < 0 else (si - 1), :] = 0
        for sj in range(abs(dj)):
            can_enter[:, sj if np.sign(dj) < 0 else (sj - 1)] = 0
        can_traverse = can_leave & can_enter
        can_traverse_ij = np.argwhere(can_traverse)
        g0_ij.append(can_traverse_ij)
        g1_ij.append(can_traverse_ij + np.array([[di, dj]]))
        g_costs.append(
            costs[(di, dj)][can_traverse]
            * np.sqrt(dj ** 2 + di ** 2)
        )
    g0_ij = np.vstack(g0_ij)
    g1_ij = np.vstack(g1_ij)
    g_costs = np.hstack(g_costs)

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
    return g0_ij, g1_ij, g_costs
