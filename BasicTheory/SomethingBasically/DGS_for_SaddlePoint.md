

# Distributive Gauss-Seidel for General Saddle Point Problem

The DGS for Stokes problem has been introduced in [DGS_for_Stokes](./DGS_for_Stokes.md), here we focus on the DGS method for the general saddle-point problem.

---

The $$u$$, $$v$$ and $$p$$ are in the following figure

<img src="./subfiles/MAC_u_v.png" alt="MAC_u_v" style="zoom:50%;" />

---

## Step 1

Suppose we solve Stokes equations 
$$
\mathcal L U = F \tag{1}
$$
where 
$$
\begin{equation}
\mathcal L = \begin{pmatrix}
-\mu \Delta \quad \nabla \\
-\nabla\cdot \quad 0
\end{pmatrix}, \quad U = \begin{pmatrix}
\pmb u \\
p
\end{pmatrix}, \quad F = \begin{pmatrix}
\pmb f \\
g
\end{pmatrix}.
\end{equation}
$$
And we introduce the distributive matrix
$$
\mathcal M = \begin{pmatrix}
I \quad \nabla \\
0 \quad \mu\Delta_p
\end{pmatrix}.
$$
Since $$-\mu \Delta \nabla + \nabla \mu\Delta = (-\mu \Delta + \nabla \mu\nabla\cdot)\nabla = (\mu\nabla\times\nabla\times)\nabla = 0$$, therefore $$\mathcal L\mathcal M = \begin{pmatrix} -\mu\Delta \quad 0 \\ -\nabla\cdot \quad - \Delta_p \end{pmatrix}$$ is lower triangluar matrix. Here we have used the identity
$$
-\Delta = -\rm{grad\,div + curl\,curl \quad(in\ 3D)},\\-\Delta = -\rm{grad\,div + curl\,rot \quad(in\ 2D)},
$$
which holds in $$H^{-1}$$ topology, and the fact: $$\rm{curl\,grad = 0}$$.

---

## Step 2

Suppose we solve equations 
$$
\mathcal L U = F \tag{2}
$$
where 
$$
\begin{equation}\mathcal L = \begin{pmatrix}-\mu \Delta \quad \nabla \\-\nabla\cdot \quad -I\end{pmatrix}, \quad U = \begin{pmatrix}\pmb u \\p\end{pmatrix}, \quad F = \begin{pmatrix}\pmb f \\g\end{pmatrix},\end{equation}
$$
with $$I$$ is the identity operator.

We need to find the distributive matrix $$\mathcal M$$ such that $$\mathcal L \mathcal M$$ is the lower triangluar matrix. We choose the same $$\mathcal M$$ as in the DGS for Stokes problem:
$$
\mathcal M = \begin{pmatrix} &I &\nabla \\ &0 &\mu(\nabla\cdot) \nabla\end{pmatrix} = \begin{pmatrix} &I &\nabla \\ &0 &\mu\Delta\end{pmatrix},
$$
we check that
$$
\begin{align}
\mathcal L \mathcal M =&\begin{pmatrix}-\mu \Delta \quad \nabla \\-\nabla\cdot \quad -I\end{pmatrix}\begin{pmatrix} &I &\nabla \\ &0 &\mu(\nabla\cdot) \nabla\end{pmatrix} \\
=&\begin{pmatrix}
&-\mu\Delta   &-\mu\Delta \nabla + \nabla \mu \Delta \\
& -(\nabla\cdot)   &-(\nabla\cdot)\nabla-\mu\Delta
\end{pmatrix} \\
=&\begin{pmatrix}
&-\mu\Delta   &0 \\
& -(\nabla\cdot)   &-\Delta-\mu\Delta
\end{pmatrix},
\end{align}
$$
and here $$\mu>0$$.

Thus 
$$
\begin{align}
\mathcal L^{-1} &= \mathcal M(\mathcal L\mathcal M)^{-1} \tag{3} \\
&= \mathcal M \begin{pmatrix} -\mu\Delta&  &0 \\ -(\nabla\cdot)&  &- \Delta_p \end{pmatrix}^{-1} \tag{4}
\end{align}
$$
will be a good candidate for $$\mathcal B$$, where $$\mathcal B$$ is an approximation of $$\mathcal L^{-1}$$.

Next we give the residual ($$r = F - \mathcal L U$$), $$r_{\pmb u}$$ and $$r_p$$ as 
$$
r_{\pmb u} = \pmb f + \mu\Delta \pmb u^k - \nabla p^k, \quad r_p = g+ \nabla\cdot \pmb u^k + p^k.
$$
Then we have 
$$
\mathcal L \begin{pmatrix}
\delta\pmb u^k \\ \delta p^k
\end{pmatrix} = \begin{pmatrix}
r_{\pmb u} \\ r_p
\end{pmatrix} \quad \Longrightarrow \quad
\begin{pmatrix}
\delta\pmb u^k \\ \delta p^k
\end{pmatrix} = \mathcal L^{-1} \begin{pmatrix}
r_{\pmb u} \\ r_p
\end{pmatrix},
$$
by using $$(4)$$,
$$
\begin{pmatrix}
\delta\pmb u^k \\ \delta p^k
\end{pmatrix} = \mathcal M(\mathcal L \mathcal M)^{-1} \begin{pmatrix}
r_{\pmb u} \\ r_p
\end{pmatrix}.
$$
If we set 
$$
\mathcal L \mathcal M \begin{pmatrix}
\delta w_{\pmb u}^k \\ \delta w_p^k
\end{pmatrix} =  \begin{pmatrix}
r_{\pmb u} \\ r_p
\end{pmatrix}, \quad \text{i.e.,} \quad \begin{pmatrix}
\delta w_{\pmb u}^k \\ \delta w_p^k
\end{pmatrix} = (\mathcal L \mathcal M )^{-1} \begin{pmatrix}
r_{\pmb u} \\ r_p
\end{pmatrix}, \tag{5}
$$
then
$$
\begin{pmatrix}
\delta\pmb u^k \\ \delta p^k
\end{pmatrix} = \mathcal M \begin{pmatrix}
\delta w_{\pmb u}^k \\ \delta w_p^k
\end{pmatrix} = \begin{pmatrix}
\delta w_{\pmb u}^k + \nabla \delta w_p^k \\ \mu\Delta \delta w_p^k
\end{pmatrix}.
$$
Hence $$\pmb u^{k+1}$$ and $$p^{k+1}$$ can be updated as 
$$
\begin{align}
&\pmb u^{k+1} = \pmb u^{k} + \delta\pmb u^{k} = \underline{\pmb u^{k} + \delta w_{\pmb u}^k} + \nabla \delta w_p^k, \tag{6}\\
&p^{k+1} = p^k + \delta p^{k} = p^k + \mu\Delta \delta w_p^k, \tag{7}
\end{align}
$$
here, we need note the underline term $$\pmb u^{k} + \delta w_{\pmb u}^k$$, which will be reformed in the following.

We now give detail calculation of equation $$(5)$$.

- The first equation of $$(5)$$ is $$-\mu \Delta \delta w_{\pmb u}^k = r_{\pmb u}$$, combined with the definition of $$r_{\pmb u}$$
  $$
  \begin{align}
  &\ -\mu \Delta \delta w_{\pmb u}^k = \pmb f - \nabla p^k + \mu\Delta \pmb u^k, \tag{8.1} \\
  \Longrightarrow &\ -\mu \Delta (\pmb u^k + \delta w_{\pmb u}^k ) =\pmb f - \nabla p^k,\tag{8.2}  \\
  :\Longrightarrow &\ -\mu \Delta \pmb u^{k+\frac{1}{2}} =\pmb f - \nabla p^k, \tag{8.3}
  \end{align}
  $$
  where, we define the intermedia velocity $$\pmb u^{k+\frac{1}{2}}:=\pmb u^k + \delta w_{\pmb u}^k$$.

  So for equation $$(6)$$, we have 
  $$
  \pmb u^{k+1} = \pmb u^{k+\frac{1}{2}} + \nabla \delta w_p^k,
  $$
  where we can get $$\pmb u^{k+\frac{1}{2}}$$ by solving $$(8.3)$$ with one Gauss-Seidel relaxation. 

  The next thing is to solve $$\delta w_p^k$$, this involves the second equation of $$(5)$$.

- The second equation of $$(5)$$ is 
  $$
  \begin{align}
    (-1-\mu)\Delta \delta w_p^k &= r_p + \nabla\cdot\delta w_{\pmb u}^k \\
    &= g + p^k + \nabla\cdot \pmb u^k + \nabla\cdot\delta w_{\pmb u}^k \\
    &= g + p^k + \nabla\cdot(\pmb u^k+\delta w_{\pmb u}^k)\\
    &= g + p^k + \nabla\cdot \pmb u^{k+\frac{1}{2}}.
  \end{align}
  $$
   So for equation $$(7)$$, we have 
  $$
  p^{k+1} = p^{k} + \mu\Delta \delta w_p^k = p^{k} + \frac{\mu}{-1-\mu}(g + p^k + \nabla\cdot \pmb u^{k+\frac{1}{2}}) \tag{9}.
  $$

Finally, we can see that even though we introduce operators such as $$\mathcal L$$ and $$\mathcal M$$, we don't use it in the implementation. The algorithm can be summarized as the following:

**Algorithm**: Distributive Gauss-Seidel: Given $$(\pmb u^k, p^k)$$

- **Step 1**: Relax momentum equation to get intermedia velocity $$\pmb u^{k+\frac{1}{2}}$$. 

  Solve momentum equation 
  $$
  -\mu \Delta \pmb u^{k+\frac{1}{2}} =\pmb f - \nabla p^k,
  $$
  approximately by one Gauss-Seidel relaxation.

- **Step 2**: Update velocity cellwisely and pressure patchwisely.

  - **Step 2.1**: For each cell $$T$$, solve the following equation to get $$\delta w_p^k$$
    $$
    \begin{align}
    (-1-\mu)\Delta \delta w_p^k = g + p^k + \nabla\cdot \pmb u^{k+\frac{1}{2}},
    \end{align}
    $$
    then project $$\pmb u^{k+1}|_T$$ on to local divergence free sapce on $$T$$
    $$
    \pmb u^{k+1}|_T = \pmb u^{k+\frac{1}{2}}|_T + \nabla \delta w_p^k.
    $$

  - **Step 2.2**: Correct pressure for the current cell $$T$$ and its neighboring cells.
    $$
    p^{k+1} = p^{k} + \mu\Delta \delta w_p^k,
    $$
    or using $$\pmb{u}^{k+\frac{1}{2}}$$ and $$(9)$$, 
    $$
    p^{k+1} = p^{k} + \mu\Delta \delta w_p^k = p^{k} + \frac{\mu}{-1-\mu}(g + p^k + \nabla\cdot \pmb u^{k+\frac{1}{2}}).
    $$



---

## Step 3

Suppose we solve equations 
$$
\mathcal L U = F \tag{10}
$$
where 
$$
\begin{equation}\mathcal L = \begin{pmatrix}-\mu \Delta + \gamma I \quad \nabla \\-\nabla\cdot \quad -I\end{pmatrix}, \quad U = \begin{pmatrix}\pmb u \\p\end{pmatrix}, \quad F = \begin{pmatrix}\pmb f \\g\end{pmatrix},\end{equation}
$$
with $$I$$ is the identity operator.

We need to find the distributive matrix $$\mathcal M$$ such that $$\mathcal L \mathcal M$$ is the lower triangluar matrix. We choose the $$\mathcal M$$ as:
$$
\mathcal M = \begin{pmatrix} &I &\nabla \\ &0 &\mu(\nabla\cdot)\nabla - \gamma I\end{pmatrix} = \begin{pmatrix} &I &\nabla \\ &0 &\mu\Delta - \gamma I \end{pmatrix},
$$
we check that
$$
\begin{align}
\mathcal L \mathcal M =&\begin{pmatrix}-\mu \Delta + \gamma I \quad \nabla \\-\nabla\cdot \quad -I\end{pmatrix}\begin{pmatrix} &I &\nabla \\ &0 &\mu(\nabla\cdot) \nabla - \gamma I\end{pmatrix} \\
=&\begin{pmatrix}
&-\mu\Delta + \gamma I   &(-\mu\Delta+\gamma I) \nabla + \nabla (\mu \Delta-\gamma I) \\
& -(\nabla\cdot)   &-(\nabla\cdot)\nabla-(\mu\Delta-\gamma I)
\end{pmatrix} \\
=&\begin{pmatrix}
&-\mu\Delta + \gamma I   &0 \\
& -(\nabla\cdot)   &(-1-\mu)\Delta + \gamma I
\end{pmatrix},
\end{align}
$$
where $$\mu>0$$ and we have used $$-\mu \Delta \nabla + \nabla \mu\Delta =0$$.

Thus 
$$
\begin{align}
\mathcal L^{-1} &= \mathcal M(\mathcal L\mathcal M)^{-1} \tag{11.1} \\
&= \mathcal M \begin{pmatrix}
&-\mu\Delta + \gamma I &0 \\
& -(\nabla\cdot)   &(-1-\mu)\Delta + \gamma I
\end{pmatrix}^{-1} \tag{11.2}
\end{align}
$$
will be a good candidate for $$\mathcal B$$, where $$\mathcal B$$ is an approximation of $$\mathcal L^{-1}$$.

Next we give the residual ($$r = F - \mathcal L U$$), $$r_{\pmb u}$$ and $$r_p$$ as 
$$
r_{\pmb u} = \pmb f + \mu\Delta \pmb u^k - \gamma \pmb u^k  - \nabla p^k, \quad r_p = g+ \nabla\cdot \pmb u^k + p^k.
$$
Then we have 
$$
\mathcal L \begin{pmatrix}
\delta\pmb u^k \\ \delta p^k
\end{pmatrix} = \begin{pmatrix}
r_{\pmb u} \\ r_p
\end{pmatrix} \quad \Longrightarrow \quad
\begin{pmatrix}
\delta\pmb u^k \\ \delta p^k
\end{pmatrix} = \mathcal L^{-1} \begin{pmatrix}
r_{\pmb u} \\ r_p
\end{pmatrix},
$$
by using $$(4)$$,
$$
\begin{pmatrix}
\delta\pmb u^k \\ \delta p^k
\end{pmatrix} = \mathcal M(\mathcal L \mathcal M)^{-1} \begin{pmatrix}
r_{\pmb u} \\ r_p
\end{pmatrix}.
$$
If we set 
$$
\mathcal L \mathcal M \begin{pmatrix}\delta w_{\pmb u}^k \\ \delta w_p^k\end{pmatrix} =  \begin{pmatrix}r_{\pmb u} \\ r_p\end{pmatrix}, \quad \text{i.e.,} \quad \begin{pmatrix}\delta w_{\pmb u}^k \\ \delta w_p^k\end{pmatrix} = (\mathcal L \mathcal M )^{-1} \begin{pmatrix}r_{\pmb u} \\ r_p\end{pmatrix}, \tag{12}
$$
then
$$
\begin{pmatrix}
\delta\pmb u^k \\ \delta p^k
\end{pmatrix} = \mathcal M \begin{pmatrix}
\delta w_{\pmb u}^k \\ \delta w_p^k
\end{pmatrix} = \begin{pmatrix}
\delta w_{\pmb u}^k + \nabla \delta w_p^k \\ \mu\Delta \delta w_p^k - \gamma \delta w_p^k
\end{pmatrix}.
$$
Hence $$\pmb u^{k+1}$$ and $$p^{k+1}$$ can be updated as 
$$
\begin{align}
&\pmb u^{k+1} = \pmb u^{k} + \delta\pmb u^{k} = \underline{\pmb u^{k} + \delta w_{\pmb u}^k} + \nabla \delta w_p^k, \tag{13}\\
&p^{k+1} = p^k + \delta p^{k} = p^k + \mu\Delta \delta w_p^k - \gamma \delta w_p^k, \tag{14}
\end{align}
$$
here, we need note the underline term $$\pmb u^{k} + \delta w_{\pmb u}^k$$, which will be reformed in the following.

We now give detail calculation of equation $$(12)$$.

- The first equation of $$(12)$$ is $$-\mu \Delta \delta w_{\pmb u}^k + \gamma \delta w_{\pmb u}^k = r_{\pmb u}$$, combined with the definition of $$r_{\pmb u}$$
  $$
  \begin{align}
  &\ -\mu \Delta \delta w_{\pmb u}^k +\gamma\delta w_{\pmb u}^k = \pmb f+ \mu\Delta \pmb u^k - \gamma\pmb u^k - \nabla p^k, \tag{15}
  \end{align}
  $$
  where, we define the intermedia velocity $$\pmb u^{k+\frac{1}{2}}:=\pmb u^k + \delta w_{\pmb u}^k$$ (actually, there is no need to define the intermedia $$\pmb u^{k+\frac{1}{2}}$$). 

  So for equation $$(13)$$, we have 
  $$
  \pmb u^{k+1} = \pmb u^{k+\frac{1}{2}} + \nabla \delta w_p^k,
  $$
  where we can get $$\pmb u^{k+\frac{1}{2}}$$ by solving $$(15.3)$$ with one Gauss-Seidel relaxation. 

  The next thing is to solve $$\delta w_p^k$$, this involves the second equation of $$(12)$$.

- The second equation of $$(12)$$ is 
  $$
  \begin{align}
    (-1-\mu)\Delta \delta w_p^k + \gamma\delta w_p^k &= r_p + \nabla\cdot\delta w_{\pmb u}^k \\
    &= g + p^k + \nabla\cdot \pmb u^k + \nabla\cdot\delta w_{\pmb u}^k \\
    &= g + p^k + \nabla\cdot(\pmb u^k+\delta w_{\pmb u}^k)\\
    &= g + p^k + \nabla\cdot \pmb u^{k+\frac{1}{2}}. \tag{16}
  \end{align}
  $$
   So for equation $$(14)$$, we have (the red formulation is not needed)
  $$
  p^{k+1} = p^{k} + \mu\Delta \delta w_p^k - \gamma\delta w_p^k \color{red}{= p^{k} -(g + p^k + \nabla\cdot \pmb u^{k+\frac{1}{2}} + \Delta\delta w_p^k) } \tag{17}.
  $$

Finally, we can see that even though we introduce operators such as $$\mathcal L$$ and $$\mathcal M$$, we don't use it in the implementation. The algorithm can be summarized as the following:

**Algorithm**: Distributive Gauss-Seidel: Given $$(\pmb u^k, p^k)$$

- **Step 1**: Solve momentum equation $$(15)$$ to get $$\Delta \delta w_{\pmb u}^k$$, i.e.,
  $$
  \begin{align}
  &\ -\mu \Delta \delta w_{\pmb u}^k +\gamma\delta w_{\pmb u}^k = \pmb f+ \mu\Delta \pmb u^k - \gamma\pmb u^k - \nabla p^k, \tag{18}
  \end{align}
  $$
  approximately by one Gauss-Seidel relaxation. 

  One more thing is that, in right-hand side of $$(17)$$, one need to compute $$\mu \Delta \pmb u^k$$, we should be careful to compute $$\mu \Delta \pmb u^k$$ on the boundary.

- **Step 2**: Solve equation $$(16)$$ to get $$\Delta \delta w_p^k$$, i.e.,
  $$
  \begin{align}
    (-1-\mu)\Delta \delta w_p^k + \gamma\delta w_p^k &= r_p + \nabla\cdot\delta w_{\pmb u}^k \\
    &= g + p^k + \nabla\cdot \pmb u^k + \nabla\cdot\delta w_{\pmb u}^k, \tag{19}
  \end{align}
  $$

- **Step 3**: Using $$(13)$$, $$(14)$$ to update $$\pmb u^{k+1}$$ and $$p^{k+1}$$
  $$
  \begin{align}
  &\pmb u^{k+1} = \pmb u^{k} + \delta\pmb u^{k} = \pmb u^{k} + \delta w_{\pmb u}^k + \nabla \delta w_p^k, \tag{20.1}\\
  &p^{k+1} = p^k + \delta p^{k} = p^k + \mu\Delta \delta w_p^k - \gamma \delta w_p^k,\tag{20.2}
  \end{align}
  $$
  



**Discrete $$(18)$$, $$(19)$$**: For simplicity, we let $$U:=(\delta w_{\pmb u}^k)|_u$$, $$V:=(\delta w_{\pmb u}^k)|_v$$ and $$P:=\delta w_p^k$$ 

- Discrete $$(18)$$:

  - Inner nodes, at $$(i,j)$$, (we have omited the superscript $$k$$)
    $$
    \begin{align}
    &\mu \frac{4U_{i,j}-U_{i-1,j}-U_{i+1,j}-U_{i,j+1}-U_{i,j-1}}{h^2} + \gamma U_{i,j} \\ =& f^1_{i,j} + \mu \frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2} - \gamma u_{i,j} - \frac{p_{i,j}-p_{i-1,j}}{h}, \tag{21.1}
    \end{align}
    $$
    thus, we can get 
    $$
    \begin{align}
    (4+\frac{\gamma h^2}{\mu})U_{i,j} =& (U_{i-1,j}+U_{i+1,j}+U_{i,j+1}+U_{i,j-1}) \\
    & \quad +\frac{h^2}{\mu}f^1_{i,j} + (u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}) \\
    & \quad -\frac{\gamma h^2}{\mu}u_{i,j} - \frac{h}{\mu}(p_{i,j}-p_{i-1,j}). \tag{21.2}
    \end{align}
    $$
    

    Similarly, 
    $$
    \begin{align}
    &\mu \frac{4V_{i,j}-V_{i-1,j}-V_{i+1,j}-V_{i,j+1}-V_{i,j-1}}{h^2} + \gamma V_{i,j} \\ =& f^2_{i,j} + \mu \frac{v_{i+1,j}+v_{i-1,j}+v_{i,j+1}+v_{i,j-1}-4v_{i,j}}{h^2} - \gamma v_{i,j} - \frac{p_{i,j}-p_{i,j-1}}{h}, \tag{22.1}
    \end{align}
    $$
    and 
    $$
    \begin{align}
    (4+\frac{\gamma h^2}{\mu})V_{i,j} =& (V_{i-1,j}+V_{i+1,j}+V_{i,j+1}+V_{i,j-1}) \\
    & \quad +\frac{h^2}{\mu}f^1_{i,j} + (v_{i+1,j}+v_{i-1,j}+v_{i,j+1}+v_{i,j-1}-4v_{i,j}) \\
    & \quad -\frac{\gamma h^2}{\mu}v_{i,j} - \frac{h}{\mu}(p_{i,j}-p_{i,j-1}). \tag{22.2}
    \end{align}
    $$
  
- Boundary nodes, we have seen in the $$ u, v, p$$ figure, and we refer LongChen's note "MACStokes.pdf".
  
  > We then discuss discretization of boundary conditions. For Dirichlet boundary condition, one can impose it in one direction by fixing the value laying on the boundary and by extrapolation on the other direction. Let us take $$x$$-coordinate component velocity $$u$$ as an example. On edges $$x=0$$ and $$x=1$$, the value is given by the boundary condition and no equation is discretized on these points. On edges $$y=0$$ and $$y=1$$, however, there is no unknowns of $$u$$ on that edge and we need to modify the stencil at $$y=h/2, 1-h/2$$. As an example, consider the discretization at the index $$(1,j)$$. We introduce the gost value at $$y = 1+h/2$$, i.e., $$u(0,j)$$. Then we can discretize $$-\Delta u$$. The ghost value can be eliminated by the linear extrapolation, i.e., requiring $$(u^{0,j}+u^{1,j})/2 = u_D(x_j,1)$$. Therefore the modified discretization of $$-\Delta u$$ is 
    > $$
    > \begin{align}
    > -\Delta u =& \frac{4u^{1,j} - u^{0,j}-u^{2,j}-u^{1,j-1}-u^{2,j+1}}{h^2} \\
    > =& \frac{5u^{1,j} - 2u_D(x_j,1)-u^{2,j}-u^{1,j-1}-u^{2,j+1}}{h^2}.
    > \end{align}
    > $$
  
- But we need to find the $$-\Delta$$ opertor on the boundary, we take $$-\Delta U_{1,j}$$ as an example, but we should also note that here $$2\leq j \leq N-1$$, which meas that $$j$$ is the interior nodes,
    $$
    -\Delta U_{1,j} = \frac{5 U_{1,j} - 2U_{1,j}^D - U_{1,j-1} - U_{1,j+1} - U_{2,j}}{h^2}.
    $$
    
















