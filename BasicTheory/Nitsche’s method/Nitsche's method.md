### The Nitsche's method

Here we introduce the Nitsche's method refering answer of StackExchange's question: [What is the general idea of Nitsche's method in numerical analysis?](https://scicomp.stackexchange.com/questions/19910/what-is-the-general-idea-of-nitsches-method-in-numerical-analysis)

Also, from the discussion of the answer, we find some other (maybe) useful references: in the MOFEM: [Periodic boundary conditions with Nitsche's method](http://mofem.eng.gla.ac.uk/mofem/html/nitsche_periodic.html)

---

The answer:

Nitsche's method is related to discontinuous Galerkin methods (indeed, as Wolfgang points out, it is a precursor to these methods), and can be  derived in a similar fashion. Let's consider the simplest problem,  Poisson's equation:
$$
\begin{equation}
\left\{\begin{array}{ll}
-\Delta u =f \quad &\text{on} \ \Omega,\\ 
u = g \quad &\text{on} \ \partial \Omega.
\end{array}\right. \tag{1}
\end{equation}
$$
We are now looking for a variational formulation that 

1. is satisfied by the (weak) solution $ u\in H^1(\Omega) $ (i.e., consistent),
2. is symmetric in $u$ and $v$,
3. admits a unique solution (which means that the bilinear form is coercive).

We start as usual by taking the strong form of the differential equation, multiplying by a test function $v\in H^1(\Omega)$ and integrating by parts. Starting with the right-hand side, we obtain
$$
\begin{align}
(f,v) = (-\Delta u,v) &= (\nabla u, \nabla v) - \int_{\partial \Omega}\partial_{\nu}uv\,ds \\
&= (\nabla u,\nabla v) - \int_{\partial \Omega}\partial_{\nu}uv\,ds - \int_{\partial \Omega}(u-g)\partial_{\nu}v\,ds
\end{align} \tag{2}
$$
where in the last equation we have added the productive zero $0=u-g$ on the boundary.

Rearranging the terms to separate linear and bilinear forms now gives a  variational equation for a symmetric bilinear form that is satisfied for the solution $u\in H^1(\Omega)$ of (1).

The bilinear form is however not coercive, since you cannot bound it from below for $u=v$ by $c\Vert v\Vert_{H^1}^2$ (as we don't have any boundary conditions for arbitrary $v\in H^1(\Omega)$, we cannot use Poincaré's inequality as usual -- this means we can make the $L^2$ part of the norm arbitrarily large without changing the bilinear form). So we need to add another (symmetric) term that vanishes for the true solution: $\eta \int_{\partial\Omega}(u-g)v\,ds$ for some $\eta>0$ large enough. This leads to the (symmetric, consistent, coercive) weak formulation: Find $u\in H^1(\Omega)$ such that
$$
\begin{align}
&(\nabla u,\nabla v) - \int_{\partial \Omega}\partial_{\nu}uv\,ds - \int_{\partial \Omega}u\partial_{\nu}v\,ds + \eta\int_{\partial \Omega}uv\,ds \\
&= \int_{\Omega}fv\,dx - \int_{\partial \Omega}g\partial_{\nu}v\,ds + \eta \int_{\partial \Omega}gv\,ds \qquad \text{for all}\quad v\in H^1(\Omega).
\end{align} \tag{3}
$$
Taking instead of $u,v\in H^1(\Omega)$ discrete approximations $u_h,v_h\in V_h\subset H^1(\Omega)$ yields the usual Galerkin approximation. Note that since it's non-conforming due to the boundary conditions (we are looking for the discrete solution in a space that is larger than the one we sought the continuous solution in $H^1_g(\Omega):=\{v\in H^1(\Omega)| v|_{\partial\Omega}=g \}$), one cannot deduce well-posedness of the discrete problem from that of the continuous problem. Nitsche now showed that if $\eta$ is chosen as $ch^{-1}$ for $c>0$ sufficiently large, the discrete problem is in fact stable (with respect to a suitable mesh-dependent norm).

(This is not Nitsche's original derivation, which predates discontinuous Galerkin methods and starts from an equivalent minimization problem. In fact, [his original paper](http://link.springer.com/article/10.1007%2FBF02995904) does not mention the corresponding bilinear form at all, but you can find it in, e.g., [Freund and Stenberg, *On weakly imposed boundary conditions for second-order problems*, Proceedings of the Ninth Int. Conf. Finite Elements in Fluids, Venice 1995. M. Morandi Cecchi et al., Eds. pp. 327-336](http://math.aalto.fi/~rstenber/Publications/Venice95.pdf).)

**Remark**: nonconformity: the discrete solution space $V_h$ is not a subspace of the continuous solution space $H^1_g(\Omega)$ - because the Dirichlet boundary conditions are enforced only in a weak sense. One can see the [2009 (MC) Nitsche's method for general boundary conditions](./2009 (MC) Nitsche's method for general boundary conditions.pdf).

