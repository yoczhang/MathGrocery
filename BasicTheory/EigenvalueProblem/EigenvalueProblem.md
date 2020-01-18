# Eigenvalue Problems

## Basci theory

==Ref: 2018 (IMAJNA) Virtual Element Method for Second Order Elliptic Eigenvalue Problems.pdf==

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

## The relation of the source problem

==Ref: 2010 (Acta Numerica 120pages) Finite element approximation of eigenvalue problems.pdf==

An important tool for the analysis of (2.2) is the solution operator $T: H\rightarrow H$: (here $H$ is a more larger space than $V$, such as, $H = L^2(\Omega)$) given $f\in H$ our hypotheses guarantee the existence of a unique $Tf$ such that
$$
a(Tf, v) = (f, v)\quad \forall v\in V. \tag{2.3}
$$
Since we are interested in *compact* eigenvalue problems, we make the assumption that
$$
T: H\rightarrow H \quad \text{ is a compact operator},
$$
which is often a consequence of a compact embedding of $V$ into $H$. We have already observed that we consider $T$ to be self-adjoint.

On the one hand, considering the eigenvalue of the operator $T$
$$
Tf = \mu f.\tag{2.4}
$$
On the other hand, in the (2.3), we take $f = \lambda u$ and refer the (2.2), we have
$$
a(T(\lambda u),v) = \lambda(u,v) = a(u,v),
$$
so we infer that
$$
T(\lambda u) = \lambda T(u) = u, \quad \text{ i.e., } \quad Tu = \frac{1}{\lambda} u. \tag{2.5}
$$
Compare (2.4) and (2.5), we get
$$
\mu = \frac{1}{\lambda}.
$$


---

## How to treat the boundary condition in code

