

# Distributive Gauss-Seidel for General Saddle Point Problem

The DGS for Stokes problem has been introduced in [DGS_for_Stokes](./DGS_for_Stokes.md), here we focus on the DGS method for the general saddle-point problem.

---

## Step 1 (Done)

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
which holds in $$H^{-1}$$ topology, and the fact $$\rm{curl\,grad = 0}$$.

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
\delta w_{\pmb u}^k + \nabla \delta w_p^k \\ \mu\Delta_p \delta w_p^k
\end{pmatrix}.
$$
Hence $$\pmb u^{k+1}$$ and $$p^{k+1}$$ can be updated as 
$$
\begin{align}
&\pmb u^{k+1} = \pmb u^{k} + \delta\pmb u^{k} = \underline{\pmb u^{k} + \delta w_{\pmb u}^k} + \nabla \delta w_p^k, \tag{6}\\
&p^{k+1} = p^k + \delta p^{k} = p^k + \mu\Delta_p \delta w_p^k, \tag{7}
\end{align}
$$
here, we need note the underline term $$\pmb u^{k} + \delta w_{\pmb u}^k$$, which will be reformed in the following.

