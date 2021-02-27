# gridijkstra
Python package wrapping scipy's dijkstra with a grid-based interface

```python
>>> import gridijkstra
>>> import numpy as np
>>> costs = np.ones((50, 60))
>>> costs[10:15, :20] = 1e30  # np.inf also works, but is less convenient for plotting
>>> costs[20:25, 25:55] = 1e30
>>> costs[30:40, 30:40] = 1e30
>>> start = (2, 2)
>>> target = (48, 58)
>>> total_cost, path = gridijkstra.plan(costs, start, target, return_path=True)
>>> print(f'Full path length: {total_cost}')
Full path length: 102.0
```

Three use cases are shown below. See scripts/examples.ipynb for a notebook with examples

![](scripts/example1.svg)

![](scripts/example2.svg)

![](scripts/example3.svg)
