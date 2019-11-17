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

  In 2D, if we introduce the clockwise $$90^\circ$$ rotation and use $$\perp$$ in the superscript to denote this rotation, then
  $$
  {\rm curl}\, f = ({\rm grad}\,f)^{\perp}, \quad {\rm rot}\, \pmb v = {\rm div}(\pmb v^{\perp}).
  $$

---



