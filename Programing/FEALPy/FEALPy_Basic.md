# FEALPy Basic

#### 1. 

p = int(sys.argv[2])  #  p 表示空间的分片多项式阶数，如 p=2 即采用 P2 元.

n = int(sys.argv[3]) #  n 表示网格初始化时加密的次数.



#### 2. 

in `class TriangleMesh(Mesh2d)`:

...
`N = node.shape[0]`, `node.shape` returns the numbers of rows and cols:	

```python
In [23]: import numpy as np
In [24]: node = np.array([
...:             (0, 0),
...:             (1, 0),
...:             (1, 1),
...:             (0, 1)], dtype=np.float)
...:             
In [25]: node.shape
Out[25]: (4, 2)
```



#### 3.

In  `LagrangeFiniteElementSpace.py`, in the class `class LagrangeFiniteElementSpace()`

```python
class LagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C'):
        self.mesh = mesh
        self.p = p
        if len(mesh.node.shape) == 1:
            self.GD = 1
        else:
            self.GD = mesh.node.shape[1]
        if spacetype is 'C':
            if mesh.meshtype is 'interval':
                self.dof = CPLFEMDof1d(mesh, p)
                self.TD = 1
            elif mesh.meshtype is 'tri':
                self.dof = CPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype is 'stri':
                self.dof = CPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype is 'tet':
                self.dof = CPLFEMDof3d(mesh, p)
                self.TD = 3
        elif spacetype is 'D':
            if mesh.meshtype is 'interval':
                self.dof = DPLFEMDof1d(mesh, p)
                self.TD = 1
            elif mesh.meshtype is 'tri':
                self.dof = DPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype is 'tet':
                self.dof = DPLFEMDof3d(mesh, p)
                self.TD = 3
  ...
  ...
  ...
```

where

* `GD`: grid-dimension;
* `TD`: topology-dimension;
* `CPLFEMDof1d`: continuous piecewise Lagrange FEM Dof in 1D;
* `DPLFEMDof1d`: discontinuous piecewise Lagrange FEM Dof in 1D.



#### 4.

```python
class DPLFEMDof1d(DPLFEMDof):
    def __init__(self, mesh, p):
        super(DPLFEMDof1d, self).__init__(mesh, p)

    def multi_index_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex
```

where the usage of `super` function can be found in [Python super 函数详解](https://www.jianshu.com/p/6d7cce41dc65) and [python中的super函数及MRO](https://blog.csdn.net/m0_38063172/article/details/82250865).



#### 5. Plot the mesh

```python
import matplotlib.pyplot as plt
from fealpy.fem.PoissonFEMModel import PoissonFEMModel
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.pde.poisson_2d import CosCosData as PDE

d = 2
n = 1
pde = PDE()
mesh = pde.init_mesh(n)
from fealpy.mesh.mesh_tools import unique_row, find_node, find_entity, show_mesh_2d
fig = plt.figure()
axes = fig.gca()
show_mesh_2d(axes,mesh)
plt.show()

## TODO:
find_node(axes,mesh.node,showindex=True)
```

![fealpy_plotmesh0](fealpy_files/fealpy_plotmesh0.png)



#### 6. Get barycentric points (bcs) and basis

```python
#!/usr/bin/env python3
#
# yc test file
#

# ------------------------------------------------- #
# --- project 1: get barycentric points (bcs)   --- #

import sys
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.functionspace.dof import CPLFEMDof2d
from fealpy.mesh.mesh_tools import find_node, find_entity

# init settings
n = 0  # refine times
p = 2  # polynomial order of FEM space
q = p + 1  # integration order

node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)], dtype=np.float)

cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)  # tri mesh

mesh = TriangleMesh(node, cell)
mesh.uniform_refine(n)

dof = CPLFEMDof2d(mesh, p)

# plot mesh
ipoint = dof.interpolation_points()
cell2dof = dof.cell2dof

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='w')
find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', fontsize=15)
find_node(axes, ipoint, showindex=True, fontsize=12, markersize=25)
plt.show()

# get bcs
integrator = mesh.integrator(q)
qf = integrator
bcs, ws = qf.quadpts, qf.weights
shape = bcs.shape

print(shape)
# ------------------------------------------------- #

# ------------------------------------------------- #
# ---      project 2: get basis at bcs          --- #

# Ref: lagrange_fem_space.py -- basis
bc = bcs
ftype = mesh.ftype
TD = 2  # topological dimension
multiIndex = dof.multiIndex

c = np.arange(1, p+1, dtype=np.int)
P = 1.0/np.multiply.accumulate(c)
t = np.arange(0, p)
shape = bc.shape[:-1]+(p+1, TD+1)
A = np.ones(shape, dtype=ftype)
A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
np.cumprod(A, axis=-2, out=A)
A[..., 1:, :] *= P.reshape(-1, 1)
idx = np.arange(TD+1)
phi = np.prod(A[..., multiIndex, idx], axis=-1)


# ------------------------------------------------- #
print("End of this test file")
```

In the above (and the following $tdim, gdim \geq 2$)

- `bcs`: the intergal points, the shape is `(NQ,tdim+1)`, the `NQ` means the number of intergal points; `tdim` is topology-dimension, in 2D, means $\lambda_0, \lambda_1, \lambda_2$. 
- `ws`: the corresponding Gauss weights, the shape is `(NQ,)`.
- `phi`: the shape of 'phi' can be `(NQ, ldof)`, the `ldof` is the number of local basis functions.



In the `LagrangeFiniteElementSpace.py`, defined the `grad_basis()` function which returen the `gphi`

- `gphi`: the gradient basis function values at barycentric point `bcs`, the shape of 'gphi' is `(NQ,NC,ldof,gdim)`, `NC` is the number of cells, `gdim` is the grid-dimension.