# Eigenvalue Problems

## Basci theory

Ref: 2018 (IMAJNA) Virtual Element Method for Second Order Elliptic Eigenvalue Problems.pdf.

We are interested in the problem of computing the eigenvalues of the Laplace operator, namely finding $$\lambda\in\mathbb{R}$$ such that there exists $$u$$, with $$\lVert u\rVert_0 =1$$, satisfying
$$
\begin{align}\tag{2.1}
-\Delta u &= \lambda u \quad \text{ in } \Omega \\
u &= 0 \quad\ \ \, \text{ on } \Gamma,
\end{align}
$$
where $$\Omega\subset\mathbb{R}^n (n=2,3)$$ is a bounded polygonal/polyhedral domain with Lipschitz boundary $$\Gamma$$.



For ease of exposition, we focus on the case of Dirichlet boundary conditions. The extension to other boundary conditions is analogous.



The variational formulation of problem $$(2.1)$$ reads as follows: find $$\lambda\in\mathbb{R}$$ such that there exists $$u\in V$$, with $$\lVert u\rVert_0 =1$$, satisfying
$$
a(u,v) = \lambda b(u,v) \quad \forall v\in V, \tag{2.2}
$$
where $$V=H^1_0(\Omega)$$, $$a(u,v)=\int_\Omega\nabla u\cdot\nabla v$$ and $$b(\cdot,\cdot)$$ denotes the $$L^2$$-inner product.



It is well known that the eigenvalues of problem $$(2.2)$$ form a positive increasing divergent sequence and that the corresponding eigenfunctions are an orthonormal basis of $$V$$ with respect to both the $$L^2$$-inner product and the scalar product associated with the bilinear form $$a(\cdot,\cdot)$$.



Due to regularity results [^(Agmon, 1965)], there exists a constant $$r>\frac{1}{2}$$ depending on $$\Omega$$, such that the solution $$u$$ belongs to the space $$H^{1+r}(\Omega)$$. It can be proved that $$r$$ is at least $$1$$ if $$\Omega$$ is a convex domain while $$r$$ is at least $$\frac{\pi}{\omega}-\varepsilon$$ for any $$\varepsilon>0$$ for a nonconvex domain, with $$\omega<2\pi$$ the maximum interior angle of $$\Omega$$.





[^(Agmon, 1965)]: *Lectures on Elliptic Boundary Value Problems*.

---

## How to treat the boundary condition in code

