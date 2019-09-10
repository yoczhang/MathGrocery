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

     

4. ~~So if we directly use the non-scaled basis such as:~~ 
   $$
   \begin{align}
   & \phi_i \in \{ 1, x, y\} \text{ in DG}; \\
   & \phi_i \in \{ x, y, 1-x-y\} \text{ in FEM}.
   \end{align}
   $$
   ~~The sitff-matirx and mass-matrix have the following scales:~~

   - ~~$$(\nabla\phi_i,\nabla\phi_j)_E \simeq h_E^d $$;~~
   - ~~$$(\phi_i,\phi_j)_E \simeq h_E^{d+2}$$.~~

   

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

## Clockwise and anticlockwise $$90^\circ$$ rotation

In 2D, let $$\pmb v = (v_1,v_2)$$, then

- clockwise $$90^\circ$$ rotation of $$\pmb v$$: $$(v_2, -v_1)$$;
- anticlockwise $$90^\circ$$ rotation of $$\pmb v$$: $$(-v_2, v_1).$$



---

## Verifying Numerical Convergence Rates

One can find the details paper in `NutstoreSync/PAPERS/Basics/`, which named as "Verifying Numerical Convergence Rates.pdf".



---

## Differential operators in 2D

- In 3D, $$\pmb v = (v_1,v_2, v_3)$$, 
  $$
  {\rm curl}\,\pmb v = \left| \begin{array}{ccc}
  \pmb i & \pmb j & \pmb k \\
  \partial_x & \partial_y & \partial_z \\ 
   v_1 & v_2 & v_3
  \end{array} \right| = \pmb i \left| \begin{array}{cc}
  \partial_y & \partial_z \\ 
   v_2 & v_3
  \end{array} \right| - \pmb j \left| \begin{array}{cc}
  \partial_x & \partial_z \\ 
   v_1 & v_3
  \end{array} \right| + \pmb k \left| \begin{array}{cc}
  \partial_x & \partial_y \\ 
   v_1 & v_2
  \end{array} \right|,
  $$
  where $$\pmb i, \pmb j, \pmb k$$ is the coordinate vectors in space.

  

- In 2D, 

  Besides the gradient and divergence operator, we will use two more differential operators $$\rm curl$$ and $$\rm rot$$ in $$\mathbb R^2$$. The $$\rm curl$$ operator is unambiguously defined for a 3D vector fields $$\pmb v$$ and the result $${\rm curl}\, \pmb v$$ is still a 3D vector field. When restricted to 2D, we have two variants. 

  - For a scalar function $$\phi$$, treating it as $$(0,0,\phi)$$ and taking $$\rm curl$$, we get a 3D vector field which can be defined as a 2D vector field since the third component is always zero
    $$
    {\rm curl }\, \phi = (\partial_y\phi, -\partial_x \phi).
    $$
    

  - For a 2D vector field $$\pmb v = (v_1(x,y),v_2(x,y))$$, treating as $$(v_1,v_2,0)$$ and taking $$\rm curl$$, we get a 3D vector with only nonzero component in the third coordinate and thus can be identified as a scalar
    $$
    {\rm rot}\, \pmb v = \partial_x v_2 - \partial_y v_1.
    $$

  In 2D, if we introduce the clockwise 

---





