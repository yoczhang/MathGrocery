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

   * $$(\nabla m_i, \nabla m_j)_E \simeq h_E^{d-2}$$;

   * $$(m_i,m_j)_E \simeq h_E^d$$.

     

4. So if we directly use the non-scaled basis such as: 
   $$
   \begin{align}
   & \phi_i \in \{ 1, x, y\} \text{ in DG}; \\
   & \phi_i \in \{ x, y, 1-x-y\} \text{ in FEM}.
   \end{align}
   $$
   The sitff-matirx and mass-matrix have the following scales:

   - $$(\nabla\phi_i,\nabla\phi_j)_E \simeq h_E^d $$;
   - $$(\phi_i,\phi_j)_E \simeq h_E^{d+2}$$.

   

5. n the FEM triangle mesh, we have compute the stiff-matrix by using the barycentric coordinates,
   $$
   \lambda_1, \lambda_2, \lambda_3, ...
   $$
   From the result in $$2D$$ (The details see [exercise_ch2](../..//Exercise_LongLectures/ch2_FEM/exercise_ch2.md)): Let $$c_i = \cot\theta_i, i=1,2,3$$. If we define the local stiffness matrix $$\pmb A_\tau$$ as $$3\times 3$$ matrix formed by $$\int_\tau \nabla\lambda_i\cdot\nabla\lambda_j {\rm d}x, \ i,j = 1,2,3$$. Then 
   $$
   \pmb A_\tau = \frac{1}{2}\begin{bmatrix}&c_2+c_3 \quad &-c_3 \quad &-c_2 \\&-c_3 \quad &c_3+c_1 \quad &-c_1 \\&-c_2 \quad &-c_1 \quad &c_1+c_2\end{bmatrix}.
   $$
   So this matrix $$\pmb A_\tau$$ is independent of $$h$$. We infer that the sitff-matirx and mass-matrix have the following scales:

   - $$(\nabla\lambda_i,\nabla\lambda_j)_E \simeq h_E^{d-2} $$;
   - $$(\lambda_i,\lambda_j)_E \simeq h_E^{d} $$;







---

## Verifying Numerical Convergence Rates

One can find the details paper in `NutstoreSync/PAPERS/Basics/`, which named as "Verifying Numerical Convergence Rates.pdf".



