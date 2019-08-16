# FEALPy manual

## Mesh 

```python
node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)], dtype=np.float)

cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)  # tri mesh

mesh = TriangleMesh(node, cell)
mesh.uniform_refine(n)
```

* `mesh.ds.NN`: Number of nodes;
* `mesh.ds.NC`: Number of cells;
* `mesh.ds.cell_to_cell()`: An array, let `cell2cell = mesh.ds.cell_to_cell() `, then `cell2cell[cellidx,lidx]` gives the `cellidx-th`(index from 0) cell's `lidx-th` edge's neighbor cell. If `cellidx-th` is the boundary-cell, `cell2cell[cellidx,lidx]` is itself.
* `mesh.entity_measure(arg)`：`arg` can take `'cell', 'face', 'edge'`, and return the corresponding measure of `arg`.

