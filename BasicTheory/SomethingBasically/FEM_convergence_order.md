# FEM convergence order

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

   

5. In the FEM triangle mesh, we have compute the stiff-matrix by using the barycentric coordinates,
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

## Convergence order

Something like in the file [The_difference_l2_L2_norms](./The_difference_l2_L2_norms.md). In 2D, the number of Dofs is $N\approx \frac{1}{h^2}$ and in 3D, the number of Dofs is $N\approx \frac{1}{h^3}$. 

==注意: 这里 $N$ 更准确的应该是正则剖分的网格的顶点数==, 一个简单的理解可以考虑 2D 下的四边形网格剖分 $N = n_x\times n_y = 1/h_x\times 1/h_y$.

For simple, we just give the order $r$
$$
\frac{\log\vert\frac{ e_1}{ e_2}\vert}{\log\vert\frac{ h_1}{ h_2}\vert} = r. \tag{1}
$$

- 2D

  So in 2D, we have 
  $$
  \frac{\log\vert\frac{ e_1}{ e_2}\vert}{\log\vert\frac{ N_1^{-1/2}}{ N_2^{-1/2}}\vert} \approx r, \ \Rightarrow\  \frac{\log\vert\frac{ e_1}{ e_2}\vert}{\log\vert\frac{ N_1}{ N_2}\vert^{(-1/2)}} \approx r, \ \Rightarrow \  \frac{\log\vert\frac{ e_1}{ e_2}\vert}{(-1/2)\log\vert\frac{ N_1}{ N_2}\vert} \approx r, \ \Rightarrow \ \frac{\log\vert\frac{ e_1}{ e_2}\vert}{\log\vert\frac{ N_1}{ N_2}\vert} \approx -\frac{r}{2}. \tag{2}
  $$
  

  上面 `(2)` 中给出了剖分网格顶点数 $N_1$ 和 $N_2$ 之间的关系.

  数值方法中自由度 $N_{dof1}$ 和 $N_{dof2}$ 之间也有 `(2)` 中的关系, 即
  $$
  \frac{\log\vert\frac{ e_1}{ e_2}\vert}{\log\vert\frac{ N_{dof1}}{ N_{dof2}}\vert} \approx -\frac{r}{2}, \tag{3}
  $$
  但是 `(2)` 和 `(3)` 之间是如何推导的暂时还没弄清楚.

  

- 3D

  类似 2D 
  $$
  \frac{\log\vert\frac{ e_1}{ e_2}\vert}{\log\vert\frac{ N_1}{ N_2}\vert} \approx -\frac{r}{3}. \tag{4}
  $$



---

以上是关于有精确解时, 可以得到在==某种范数下==的误差  $e=O(h^r)$, 所以在 $n$ 层网格上 $e_n=O(h_n^r)$, 第 $n+1$ 层网格上 $e_{n+1}=O(h_{n+1}^r)$, 一般设定 $n$ 越大, 网格尺度越小, 即
$$
\frac{e_{n}}{e_{n+1}} = (\frac{h_n}{h_{n+1}})^r \ \Rightarrow \ \frac{\log\vert\frac{ e_{n}}{ e_{n+1}}\vert}{\log\vert\frac{h_{n}}{h_{n+1}}\vert} = r. \tag{5}
$$
当没有精确解时, 令 $x_n$ 为第 $n$ 层网格的数值解, $x_{n+1}$ 为第 $n+1$ 层网格的数值解, $x^*$ 为(假设的)真解, 我们有 $e_n = x_n-x^*$, $e_{n+1} = x_{n+1}-x^*$, 且
$$
\begin{align}
&\frac{x_{n+1}-x_n}{x_n-x^*} \\
&=  \frac{x_{n+1}-x^* + x^*-x_n}{x_n-x^*}  \\
&= \frac{x_{n+1}-x^*}{x_n-x^*} + \frac{x^*-x_n}{x_n-x^*}\\
&= (\frac{h_{n+1}}{h_{n}})^r - 1,
\end{align}
$$
因此有 
$$
\begin{align}
 x_{n+1}-x_n = ((\frac{h_{n+1}}{h_{n}})^r - 1) e_n,
\end{align} \tag{6}
$$
同理有 
$$
\begin{align}
 x_{n+2}-x_{n+1} = ((\frac{h_{n+2}}{h_{n+1}})^r - 1) e_{n+1},
\end{align} \tag{7}
$$
则有
$$
\begin{align}
\frac{x_{n+1}-x_n}{x_{n+2}-x_{n+1}} &= \frac{((\frac{h_{n+1}}{h_{n}})^r - 1) e_n}{((\frac{h_{n+2}}{h_{n+1}})^r - 1) e_{n+1}} \\
&= \frac{h_{n+1}^r-h_{n}^r}{h_{n+2}^r-h_{n+1}^r}.
\end{align}\tag{8}
$$
最后一个等号利用了 $(5)$ 式中的第一个等号.



其实从 $(8)$ 中可以看出,

- 如果一般的网格在加密时会有: $\frac{h_{n+1}}{h_{n}} = \frac{h_{n+2}}{h_{n+1}}$, 那么此时就有
  $$
  \begin{align}
  &\frac{x_{n+1}-x_n}{x_{n+2}-x_{n+1}} = \frac{e_n}{e_{n+1}}= (\frac{h_{n}}{h_{n+1}})^r \\ 
  \Rightarrow \ & \frac{\log\vert\frac{x_{n+1}-x_n}{x_{n+2}-x_{n+1}}\vert}{\log\vert\frac{h_{n}}{h_{n+1}}\vert} = r .
  \end{align}
  $$
   

- 如果用自由度 Dof 来计算的话, 需要按照 $(8)$ 式推导一下
  $$
  \begin{align}
  &\frac{e_n}{e_{n+1}}= \frac{x_{n+1}-x_n}{x_{n+2}-x_{n+1}} = ....
  \end{align}
  $$
  

