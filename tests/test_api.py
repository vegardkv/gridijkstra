import numpy as np
import gridijkstra
import unittest


class TestIntensityCosts(unittest.TestCase):
    def test_basic(self):
        costs = np.ones((10, 10))
        stencil = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # Test diagonal
        src = np.array([2, 2])
        tar = np.array([8, 8])
        path, length = gridijkstra.plan(costs, src, tar, stencil)
        self.assertEqual(length, 12.0)
        self.assertLessEqual(np.max(path), 8)
        self.assertGreaterEqual(np.min(path), 2)

        # Test j-direction
        src = np.array([2, 2])
        tar = np.array([2, 8])
        path, length = gridijkstra.plan(costs, src, tar, stencil)
        self.assertEqual(length, 6.0)
        self.assertTrue(np.allclose(path[:, 0], 2))

        # Test i-direction
        src = np.array([8, 2])
        tar = np.array([2, 2])
        path, length = gridijkstra.plan(costs, src, tar, stencil)
        self.assertEqual(length, 6.0)
        self.assertTrue(np.allclose(path[:, 1], 2))

        # Test uneven diagonal
        src = np.array([2, 3])
        tar = np.array([4, 8])
        path, length = gridijkstra.plan(costs, src, tar, stencil)
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
        path, length = gridijkstra.plan(bw, src, tar, stencil)
        self.assertEqual(length, 11)

    def test_rectangular_grid(self):
        costs = 2 * np.ones((5, 7))
        stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        src = np.array([1, 1])
        tar = np.array([3, 6])
        path, length, total = gridijkstra.plan(costs, src, tar, stencil, return_total_cost=True)
        self.assertEqual(length, 7)
        self.assertEqual(total, 14)


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
        path, length, total = gridijkstra.plan(costs, src, tar, return_total_cost=True)
        self.assertEqual(length, 12.0)
        self.assertEqual(length, total)

        # Make j-incremental traversal expensive
        costs[(0, 1)][4:, 4:] = 10000.0
        path, length, total = gridijkstra.plan(costs, src, tar, return_total_cost=True)
        self.assertEqual(length, 12.0)
        self.assertEqual(length, total)

        # Make i-incremental traversal expensive
        costs[(1, 0)][4:, 4:] = 10000.0
        path, length, total = gridijkstra.plan(costs, src, tar, return_total_cost=True)
        self.assertEqual(length, 12.0)
        self.assertEqual(total, 40000.0 + 8.0)

        # Uneven planning 1
        tar = np.array([8, 6])
        path, length, total = gridijkstra.plan(costs, src, tar, return_total_cost=True)
        self.assertEqual(total, 20000.0 + 8.0)

        # Uneven planning 2
        tar = np.array([6, 8])
        path, length, total = gridijkstra.plan(costs, src, tar, return_total_cost=True)
        self.assertEqual(total, 20000.0 + 8.0)

