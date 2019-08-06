# FEALPy Basic

#### 1. 

p = int(sys.argv[2])  #  p 表示空间的分片多项式阶数，如 p=2 即采用 P2 元.

n = int(sys.argv[3]) #  n 表示网格初始化时加密的次数.



#### 2. 

in `class TriangleMesh(Mesh2d)`:

...
`N = node.shape[0]`, `node.shape` returns the numbers of rows and cols:	

```pyth
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

In  `lagrange_fem_space.py`, in the class `class LagrangeFiniteElementSpace()`

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

where the usage of `super` function can be found in [Python super 函数详解](https://www.jianshu.com/p/6d7cce41dc65).



