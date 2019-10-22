# The difference between $$l^2$$ norm and $$L^2$$ norm

This is from the answer [Difference between l2 norm and L2 norm](https://scicomp.stackexchange.com/questions/21761/difference-between-l2-norm-and-l2-norm), and just modify a little.

Both norms are similar in that they are induced by the scalar product of the respective Hilbert space, but they differ because the different  spaces are endowed with different inner products:

- For $$\mathbb R^N$$, the Euclidean norm of $$v = (v_1, ..., v_N)^\intercal\in\mathbb R^N$$ is defined by
  $$
  \Vert v \Vert_2^2 = (v,v)_2 = \sum_{i=1}^N v_i^2.
  $$

- For $$l^2$$ (the space of real sequences for which the following norm is finite), the norm of $$v=\{v_i\}_{i\in\mathbb N}\in l^2$$ is defined by 
  $$
  \Vert v\Vert _{l^2}^2 = (v,v)_{l^2} = \sum_{i=1}^{\infin}v_i^2.
  $$

- For $$L^2(\Omega)$$ (the space of Lebesgue measurable functions on a bounded domain $$\Omega\sub\mathbb R^d$$ for which the following norm is finite), the norm of $$v\in L^2(\Omega)$$ is defined by
  $$
  \Vert v\Vert _{L^2}^2 = (v,v)_{L^2} = \int_\Omega v(x)^2 {\rm d}x.
  $$

For example, for the finite element discretization. Let's say we have a finite-dimensional subspace $$V_h\sub L^2(\Omega)$$ which is the span of a finite number of basis functions $$\{ \varphi_1, ..., \varphi_N\}$$. Then any $$v_h\in V_h$$ can be written as 
$$
v_h = \sum_{i=1}^N v_i \varphi_i. \tag{1}
$$
Since $$V_h\sub L^2(\Omega)$$, we can of course measure $$v_h$$ by the $$L^2$$ norm. Alternatively, we can identify $$v_h$$ with the vector $$\vec{v}:=(v_1,...,v_N)^{\intercal}\in \mathbb R^N$$ (sometimes called **coordinate isomorphism**) and measure $$v_h$$ by the Euclidean norm of $$\vec v$$.

How do the two ways of measuring $$v_h$$ compare? Plugging in the definition $$(1)$$ yields
$$
\Vert v_h\Vert_{L^2}^2 = (v_h,v_h)_{L^2}=\sum_{i=1}^N\sum_{j=1}^N v_i v_j\int_{\Omega}\varphi_i\varphi_j {\rm d}x = \vec{u}^{\intercal}M_h\vec{v},
$$
where $$M_h\in\mathbb R^{N\times N}$$ is the mass matrix with entries $$M_{ij}=\int_\Omega\varphi_i\varphi_j{\rm d}x$$. By comparison, we have 
$$
\Vert v_h \Vert_{l^2}^2 := \Vert\vec{v}\Vert_2^2 = \vec{v}^\intercal \vec{v}.
$$
Both norms are therefore equivalent, i.e., there exist constants $$c_1, c_2 >0$$ such that
$$
c_1\Vert u_h \Vert_{l^2} \leq \Vert v_h \Vert_{L^2} \leq c_2\Vert u_h\Vert_{l^2} \quad \text{for all } v_h\in V_h.
$$
So in principle, you could use both norms interchangeably -- if the error goes to zero in one norm, it also goes to zero in the other norm, and with the same rate. However, note that while the constants $$c_1$$ and $$c_2$$ are independent of $$v_h$$, they do depend on $$V_h$$, and in particular on $$N$$ (here, we can set that $$N=n_x n_y$$ and $$n_x = \frac{1}{h_x}, n_y = \frac{1}{h_y}$$, if $$h_x=h_y$$, so $$N=\frac{1}{h^2}$$). This is important if you want to compare discretization errors for **different** spaces $$V_h$$ with (say) $$h_1>h_2$$ ($$N_1<N_2$$), in which case you should use a norm that does not itself depend on $$N_1$$ or $$N_2$$, i.e., the $$L^2$$ norm. (We can see this by taking $$v_h$$ as the constant function $$v_h=1$$ and comapre $$\Vert v_h\Vert_{l^2}$$ for different $$N$$ with $$\Vert v_h\Vert_{L^2}$$ -- the former scales as $$\sqrt{N}$$, while the latter has the same value for every $$N$$, since in **2D** mass matrix scaling as $$h^{d}$$, so if $$d=2$$, $$M_h\simeq h^2=\frac{1}{N}$$).

There's also a third -- intermediate -- alternative, where the mass matrix is approximated by a diagonal matrix $$D_h$$ (e.g., by taking as diagonal elements of $$D_h$$ the sum of the corresponding row of $$M_h$$), and the norm is taken as $$\Vert v_h \Vert_D^2 := \vec{v}^{\intercal} D_h \vec{v}=\sum_{i=1}^N(D_h)_{ii}v_i^2$$; this is usually referred to as *mass lumping*. This norm is also equivalent with both the $$l^2$$ and the $$L^2$$ norm -- and in this case, the constants $$c_1$$ and $$c_2$$ when comparing $$L^2$$ and mass lumping norm do *not* depend on $$N$$.