import numpy as np
import gridijkstra
import unittest


# TODO: create test for calculating upland area


class TestIntensityCosts(unittest.TestCase):
    def test_return_combinations(self):
        costs = np.ones((10, 10))
        t = gridijkstra.plan(costs, [1, 3], [3, 7])
        self.assertIsInstance(t, np.float64)
        t, p = gridijkstra.plan(costs, [1, 3], [3, 7], return_path=True)
        self.assertIsInstance(t, np.float64)
        self.assertTrue(p.dtype == np.int64)
        self.assertEqual(p.ndim, 2)
        self.assertEqual(p.shape[1], 2)
        t, p, le = gridijkstra.plan(costs, [1, 3], [3, 7], return_path=True, return_length=True)
        self.assertIsInstance(t, np.float64)
        self.assertTrue(p.dtype == np.int64)
        self.assertEqual(p.ndim, 2)
        self.assertEqual(p.shape[1], 2)
        self.assertIsInstance(le, np.float64)
        t, le = gridijkstra.plan(costs, [1, 3], [3, 7], return_length=True)
        self.assertIsInstance(t, np.float64)
        self.assertIsInstance(le, np.float64)

    def test_basic(self):
        costs = np.ones((10, 10))
        stencil = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # Test diagonal
        src = np.array([2, 2])
        tar = np.array([8, 8])
        _, path, length = gridijkstra.plan(costs, src, tar, stencil, return_path=True, return_length=True)
        self.assertEqual(length, 12.0)
        self.assertLessEqual(np.max(path), 8)
        self.assertGreaterEqual(np.min(path), 2)

        # Test j-direction
        src = np.array([2, 2])
        tar = np.array([2, 8])
        _, path, length = gridijkstra.plan(costs, src, tar, stencil, return_path=True, return_length=True)
        self.assertEqual(length, 6.0)
        self.assertTrue(np.allclose(path[:, 0], 2))

        # Test i-direction
        src = np.array([8, 2])
        tar = np.array([2, 2])
        _, path, length = gridijkstra.plan(costs, src, tar, stencil, return_path=True, return_length=True)
        self.assertEqual(length, 6.0)
        self.assertTrue(np.allclose(path[:, 1], 2))

        # Test uneven diagonal
        src = np.array([2, 3])
        tar = np.array([4, 8])
        _, path, length = gridijkstra.plan(costs, src, tar, stencil, return_path=True, return_length=True)
        self.assertEqual(length, 7.0)
        self.assertLessEqual(np.max(path[:, 0]), 4)
        self.assertGreaterEqual(np.min(path[:, 0]), 2)
        self.assertLessEqual(np.max(path[:, 1]), 8)
        self.assertGreaterEqual(np.min(path[:, 1]), 3)

    def test_ij_costs(self):
        bw = np.zeros((10, 10), dtype=np.bool)
        bw[2, 2:8] = 1
        bw[2:6, 8] = 1
        bw[6, 4:8] = 1
        bw[6:9, 4] = 1
        stencil = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        src = np.array([9, 4])
        tar = np.array([2, 2])
        _, path, length = gridijkstra.plan(bw, src, tar, stencil, return_path=True, return_length=True)
        self.assertEqual(length, 11)

    def test_rectangular_grid(self):
        costs = 2 * np.ones((5, 7))
        stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        src = np.array([1, 1])
        tar = np.array([3, 6])
        total, path, length = gridijkstra.plan(costs, src, tar, stencil, return_path=True, return_length=True)
        self.assertEqual(length, 7)
        self.assertEqual(total, 14)

    def test_invalid_nan_planning(self):
        costs = np.ones((10, 10))
        costs[:, 5] = np.nan
        total = gridijkstra.plan(costs, [1, 1], [8, 8])
        self.assertTrue(np.isinf(total))

    def test_inf_planning(self):
        costs = np.ones((10, 10))
        costs[:, 5] = np.inf
        total = gridijkstra.plan(costs, [1, 1], [8, 8])
        self.assertTrue(np.isinf(total))


class TestAsymmetricCosts(unittest.TestCase):
    def test_basic(self):
        costs = {
            (0, 1): np.ones((9, 9)),
            (0, -1): np.ones((9, 9)),
            (1, 0): np.ones((9, 9)),
            (-1, 0): np.ones((9, 9)),
        }

        src = np.array([2, 2])
        tar = np.array([8, 8])
        total, path, length = gridijkstra.plan(costs, src, tar, return_path=True, return_length=True)
        self.assertEqual(length, 12.0)
        self.assertEqual(length, total)

        # Make j-incremental traversal expensive
        costs[(0, 1)][4:, 4:] = 10000.0
        total, path, length = gridijkstra.plan(costs, src, tar, return_path=True, return_length=True)
        self.assertEqual(length, 12.0)
        self.assertEqual(length, total)

        # Make i-incremental traversal expensive
        costs[(1, 0)][4:, 4:] = 10000.0
        total, path, length = gridijkstra.plan(costs, src, tar, return_path=True, return_length=True)
        self.assertEqual(length, 12.0)
        self.assertEqual(total, 40000.0 + 8.0)

        # Uneven planning 1
        tar = np.array([8, 6])
        total, path, length = gridijkstra.plan(costs, src, tar, return_path=True, return_length=True)
        self.assertEqual(total, 20000.0 + 8.0)

        # Uneven planning 2
        tar = np.array([6, 8])
        total, path, length = gridijkstra.plan(costs, src, tar, return_path=True, return_length=True)
        self.assertEqual(total, 20000.0 + 8.0)

    def test_only_downhill_allowed(self):
        d = np.linspace(-10, 10, 21).reshape(-1, 1) ** 2
        d2 = d + d.T
        z = np.exp(-d2/100)
        i_pos_costs = np.diff(z, axis=0, append=np.nan) < 0
        i_neg_costs = np.diff(z[::-1], axis=0, prepend=np.nan)[::-1] < 0
        j_pos_costs = np.diff(z, axis=1, append=np.nan) < 0
        j_neg_costs = np.diff(z[:, ::-1], axis=1, prepend=np.nan)[:, ::-1] < 0
        costs = {
            (1, 0): np.where(i_pos_costs, 1.0, np.inf),
            (-1, 0): np.where(i_neg_costs, 1.0, np.inf),
            (0, 1): np.where(j_pos_costs, 1.0, np.inf),
            (0, -1): np.where(j_neg_costs, 1.0, np.inf),
        }
        np.random.seed(123)
        ij0 = np.random.randint(0, 10, size=(20, 2))
        ij0[5:10, :] += 11
        ij0[10:15, 0] += 11
        ij0[15:, 1] += 11
        ij1 = np.random.randint(0, 10, size=(20, 2))
        ij1[5:10, :] += 11
        ij1[10:15, 0] += 11
        ij1[15:, 1] += 11
        path_exists = np.zeros(ij0.shape[0], dtype=np.bool)
        path_exists[:5] = (ij0[:5, 0] >= ij1[:5, 0]) & (ij0[:5, 1] >= ij1[:5, 1])
        path_exists[5:10] = (ij0[5:10, 0] <= ij1[5:10, 0]) & (ij0[5:10, 1] <= ij1[5:10, 1])
        path_exists[10:15] = (ij0[10:15, 0] <= ij1[10:15, 0]) & (ij0[10:15, 1] >= ij1[10:15, 1])
        path_exists[15:] = (ij0[15:, 0] >= ij1[15:, 0]) & (ij0[15:, 1] <= ij1[15:, 1])
        totals = gridijkstra.plan(costs, ij0, ij1)
        self.assertListEqual(path_exists.tolist(), (~np.isinf(totals)).tolist())

    def test_valid_nan_planning(self):
        traversable = np.zeros((10, 12), dtype=bool)
        traversable[1:8, 1:4] = 1
        traversable[1:4, 1:8] = 1
        traversable[6:8, 1:8] = 1
        stencil = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if not (i == j == 0)]
        length = gridijkstra.plan(np.where(traversable, 1, np.nan), [1, 7], [7, 7], stencil=stencil)
        self.assertAlmostEqual(length, 4 + 5 * np.sqrt(2))
