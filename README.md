
`gridijkstra` is a Python package for 2D grid-based path-planning

The package wraps scipy's dijkstra with a grid-oriented interface:

```python
>>> import gridijkstra
>>> import numpy as np
>>> costs = np.ones((50, 60))
>>> costs[10:15, :20] = 1e30  # np.inf also works, but is less convenient for plotting
>>> costs[20:25, 25:55] = 1e30
>>> costs[30:40, 30:40] = 1e30
>>> start = (2, 2)
>>> target = (48, 58)
>>> total_cost = gridijkstra.plan(costs, start, target)
>>> print(f'Full path length: {total_cost}')
'Full path length: 102.0'
```

#### Installation

<pre>
pip install gridijkstra
</pre>

# Examples

See scripts/examples.ipynb for the code:

![](scripts/example1.svg)

![](scripts/example2.svg)

![](scripts/example3.svg)
