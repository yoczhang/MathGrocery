## Quadrature

### Gauss-Legendre Quadrature

The main purpose is to get the quadrature on edges. The details of Legendre quadrature, see [Polygon quadrature](../../Polygon_quadrature/polygon_quadrature.md).

```python
>>> from fealpy.quadrature import GaussLegendreQuadrature
>>> p = 2
>>> qf = GaussLegendreQuadrature(p + 1)
>>> bcs, ws = qf.quadpts, qf.weights
>>> bcs
array([[0.88729833, 0.11270167],
       [0.5       , 0.5       ],
       [0.11270167, 0.88729833]])
>>> ws
array([0.27777778, 0.44444444, 0.27777778])
>>> ws.sum()
1.0
```

- `bcs = qf.quadpts`: `[NQpoints x 2]`,

   `bcs[:,0]` gives the `NQpoints` quadrature points in $$[0,1]$$, and actually `bcs[:,1] = 1 - bcs[:,0] `. This special setting of `bcs` can be used to compute **physicalquadrature points**, such as,

  Given $$t\in[0,1]$$ is the quadrature point, we want to compute the physical quadrature point $$x$$ in $$[a,b]$$, then we have $$x = a + (b-a)t = (b)(t) + (a)(1-t)$$.

- We can use the following codes to give the all physical quadrature points on all edges 

  ```python
  # ...
  # we have get mesh
  node = mesh.entity('node')
  edge = mesh.entity('edge')
  
  qf = GaussLegendreQuadrature(p + 1)  # get quadrature
  bcs, ws = qf.quadpts, qf.weights
  
  allEdgeNode = node[edge]  # [Nedges x 2 x 2]
  ps = np.einsum('ij, kjm->ikm', bcs, allEdgeNode)  # get physical quadrature points
  ```

  - If we set `oneEdgeNode = allEdgeNode[i,:,:]`, `[2 x 2]`, this gives us the $$i$$-th edge's vertices coordinates, such as, `oneEdgeNode[0,:]` is the **first** vertex's $$(x_0,y_0)$$-coordinate of $$i$$-th edge, and `oneEdgeNode[1,:]` is the **second** vertex's $$(x_1,y_1)$$-coordinate of $$i$$-th edge. So 
    $$
    \begin{align}
    \begin{bmatrix} (t) \ \ (1-t) \end{bmatrix}
    \begin{bmatrix} (x_0) \ \ (y_0)\\ (x_1) \ \ (y_1) \end{bmatrix} = 
    \begin{bmatrix} (x_0)(t)+(x_1)(1-t) \ \ \  (y_0)(t)+(y_1)(1-t)\end{bmatrix}
    \end{align}
    $$

  - To understand the `np.einsum`.

