# Distributive Gauss-Seidel for Stokes Problem

## The operator form

Suppose we solve equations 
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
For one step of the iterative method is given by 
$$
U^{k+1} = U^k + \mathcal B(F-\mathcal L U^k), \tag{2}
$$
where $$\mathcal B$$ is an approximation of $$\mathcal L^{-1}$$.

In order to construct a good approximation $$\mathcal B$$, we first introduce a distributive matrix
$$
\mathcal M = \begin{pmatrix}
I \quad \nabla \\
0 \quad \mu\Delta_p
\end{pmatrix}.
$$
Since $$-\mu \Delta \nabla + \nabla \mu\Delta = (-\mu \Delta + \nabla \mu\nabla\cdot)\nabla = (\mu\nabla\times\nabla\times)\nabla = 0$$, therefore $$\mathcal L\mathcal M = \begin{pmatrix} -\mu\Delta \quad 0 \\ -\nabla\cdot \quad - \Delta_p \end{pmatrix}$$ is lower triangluar matrix. Thus 
$$
\begin{align}
\mathcal L^{-1} &= \mathcal M(\mathcal L\mathcal M)^{-1} \tag{3} \\
&= M \begin{pmatrix} -\mu\Delta \quad 0 \\ -\nabla\cdot \quad - \Delta_p \end{pmatrix}^{-1} \tag{4}
\end{align}
$$
will be a good candidate for $$\mathcal B$$.

**Update procedure**: Given $$\pmb u^k$$ and $$p^k$$, how to update it by computing the errors $$\delta\pmb u^k$$ and $$\delta p^k$$? 

We first form the residual $$r_{\pmb u}$$ and $$r_p$$ as 
$$
r_{\pmb u} = \pmb f - \nabla p^k + \mu\Delta \pmb u^k, \quad r_p = g+ \nabla\cdot \pmb u^k.
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
  -\Delta \delta w_p^k &= r_p + \nabla\cdot\delta w_{\pmb u}^k \\
  &= g + \nabla\cdot \pmb u^k + \nabla\cdot\delta w_{\pmb u}^k \\
  &= g + \nabla\cdot(u^k+\delta w_{\pmb u}^k)\\
  &= g + \nabla\cdot \pmb u^{k+\frac{1}{2}}.
  \end{align}
$$
  So for equation $$(7)$$, we have $$p^{k+1} = p^{k} + \Delta \delta w_p^k = p^{k} + (-g - \nabla\cdot \pmb u^{k+\frac{1}{2}})$$.

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
    -\Delta \delta w_p^k = g + \nabla\cdot \pmb u^{k+\frac{1}{2}},
    \end{align}
    $$
    then project $$\pmb u^{k+1}|_T$$ on to local divergence free sapce on $$T$$
    $$
    \pmb u^{k+1}|_T = \pmb u^{k+\frac{1}{2}}|_T + \nabla \delta w_p^k.
    $$
  
  - **Step 2.2**: Correct pressure for the current cell $$T$$ and its neighboring cells.
    $$
    p^{k+1} = p^{k} + \Delta \delta w_p^k,
    $$
    or using $$u^{k+\frac{1}{2}}$$, 
    $$
    p^{k+1} = p^{k} + \Delta \delta w_p^k = p^{k} + (-g - \nabla\cdot \pmb u^{k+\frac{1}{2}}).
    $$
    

