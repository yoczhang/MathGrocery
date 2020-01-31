```python
import numpy as np


class Function(np.ndarray):
    def __new__(cls, space, dim=None, array=None):
      	# 这里 space 指的就是之前构建的 '空间类' 所创建的对象,
        # (例如: space = ScaledMonomialSpace2d(mesh, p))
        if array is None:
            self = space.array(dim=dim).view(cls)
            # 例如: space = ScaledMonomialSpace2d(mesh, p),
            # 则, 由 ScaledMonomialSpace2d() 中有如下
            # def array(self, dim=None): ...
        else:
            self = array.view(cls)
        self.space = space
        return self

    def index(self, i):
        return Function(self.space, array=self[:, i])

    def __call__(self, bc, index=None):
        space = self.space
        return space.value(self, bc, index=index)

    def value(self, bc, index=None):
        space = self.space
        return space.value(self, bc, index=index)

    def grad_value(self, bc, index=None):
        space = self.space
        return space.grad_value(self, bc, index=index)

    def laplace_value(self, bc, index=None):
        space = self.space
        return space.laplace_value(self, bc, index=index)

    def div_value(self, bc, index=None):
        space = self.space
        return space.div_value(self, bc, index=index)

    def hessian_value(self, bc, index=None):
        space = self.space
        return space.hessian_value(self, bc, index=index)

    def edge_value(self, bc, index=None):
        space = self.space
        return space.edge_value(self, bc)

    def add_plot(self, plt):
        mesh = self.space.mesh
        if mesh.meshtype is 'tri':
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            fig1 = plt.figure()
            fig1.set_facecolor('white')
            axes = fig1.gca(projection='3d')
            axes.plot_trisurf(
                    node[:, 0], node[:, 1],
                    cell, self, cmap=plt.cm.jet, lw=0.0)
            return axes
        else:
            return None
```