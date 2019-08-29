# Something basically

---

[TOC]

---

## The scale of stiff-matrix and mass-matrix

1. Let $$d\, (d = 1, 2, 3)$$ denote the dimension. $$\mathcal{E}_h$$ be the mesh, $$E\in \mathcal{E}_h$$ is one element ($$E$$ denotes: `edge` when $$d=1$$, `face` when $$d=2$$ and `cell` when $$d=3$$), $$h_E$$ is the diameter, $$\mathbf{x}_E$$ is the barycentre of $$E$$.

2. Set the scaled monomials on $$E$$:
   $$
   \mathcal{M}_k(E) = \{ (\frac{\mathbf{x}-\mathbf{x}_E}{h_E})^{\mathbf{s}}, |\mathbf{s}|\leq k \},
   $$
   and let $$m_i, m_j\in \mathcal{M}_k$$ be the basis on $$E$$. 

3. The stiff-matrix and mass-matrix have the following scales:

   * $$(m_i,m_j)_E \simeq h_E^d$$;
* $$(\nabla m_i, \nabla m_j)_E \simeq h_E^{d-2}$$.



---

## Verifying Numerical Convergence Rates

One can find the details paper in `NutstoreSync/PAPERS/Basics/.`



