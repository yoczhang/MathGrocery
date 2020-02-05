1.

```python
def source_vector(self, f):
    PI0 = self.PI0  # list 变量, (NC,)
    # PI0[i].shape: (sdof, ldof), sdof 是 ScaledMonomialSpace 下的 local 自由度个数, ldof 则是每个单元中 vem 空间的 local 自由度个数.
    phi = self.smspace.basis
    def u(x, index):
        return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
    bb = self.integralalg.integral(u, celltype=True)
    g = lambda x: x[0].T@x[1]
    bb = np.concatenate(list(map(g, zip(PI0, bb))))
    gdof = self.number_of_global_dofs()
    b = np.bincount(self.dof.cell2dof, weights=bb, minlength=gdof)
    return b
```

