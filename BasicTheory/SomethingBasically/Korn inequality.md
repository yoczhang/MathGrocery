阅读下面这个提问与回答, [Why is Korn's inequality not trivial?](https://math.stackexchange.com/questions/248358/why-is-korns-inequality-not-trivial)

这个回答中虽然没有完全解释清楚, 提问者的问题, 但是仍然提供了很好的解释.



另外也可以参见 2016 (Volker John) Finite element methods for incompressible flow problems.pdf 

- Lemma 3.37 (Korn’s Inequality in $V=H^1_0(\Omega)$)
- Remark 3.38: It should be also noted that estimates of Korn’s type do not hold for all spaces, e.g., they do not hold for certain non-conforming finite element spaces.

---

#### The Korn's inequality

Let $\Omega\subset\subset \mathbb{R}^N$ have smooth boundary, $N\geq 2$ and
$$
 \mathcal{E}(v):=\int_\Omega\sum_{ij}\varepsilon_{ij}(v)\varepsilon_{ij}(v)=\int_\Omega\sum_{ij}(\frac{v_{i,j}+v_{j,i}}{2})^2 dx 
$$
be defined in $H^1(\Omega,\mathbb R^N)$. The ==Korn's inequality==:
$$
\Vert \nabla v \Vert^2_{L^2} \leq c \mathcal{E}(v). \tag{1}
$$
For example in 2D, let $v=(v_1,v_2)$,
$$
\begin{align}
\Vert\nabla v\Vert_{L^2}^2 &= \int_\Omega \left( \begin{array}{} \partial_{x}v_1 &\ \partial_{y}v_1 \\ 
\partial_{x}v_2 &\ \partial_{y}v_2 \end{array} \right) : \left( \begin{array}{} \partial_{x}v_1 &\ \partial_{y}v_1 \\ 
\partial_{x}v_2 &\ \partial_{y}v_2 \end{array} \right) \\
&=\int_\Omega (\partial_x v_1)^2 + (\partial_y v_1)^2 + (\partial_x v_2)^2 + (\partial_y v_2)^2. \tag{2}
\end{align}
$$
And 
$$
\begin{align}
\mathcal{E}(v) &= \int_\Omega \left( \begin{array}{} \partial_{x}v_1 &\ \frac{\partial_{y}v_1+\partial_{x}v_2}{2} \\ 
\frac{\partial_{y}v_1+\partial_{x}v_2}{2} &\ \partial_{y}v_2 \end{array} \right) : \left( \begin{array}{} \partial_{x}v_1 &\ \frac{\partial_{y}v_1+\partial_{x}v_2}{2} \\ 
\frac{\partial_{y}v_1+\partial_{x}v_2}{2} &\ \partial_{y}v_2 \end{array} \right) \\
&=\int_\Omega (\partial_x v_1)^2 + \frac{1}{2}(\partial_y v_1)^2 + \partial_y v_1\partial_x v_2 + \frac{1}{2}(\partial_x v_2)^2 + (\partial_y v_2)^2.  \tag{3}
\end{align}
$$
Clearly, using the Young inequality $\partial_y v_1\partial_x v_2 \leq \frac{1}{2}( (\partial_y v_1)^2+(\partial_x v_2)^2 )$, easy to get
$$
\mathcal{E}(v) \leq \Vert \nabla v \Vert^2_{L^2}.
$$


#### The explanation of Korn's inequality

**See the answer**.

Yes, it's "only" a matter of possible cancellation in the sum $v_{i,j}+v_{j,i}$. Like children, analysts can get excited about tiny little things of this kind.

But seriously, this is an amazing result. For example, if you want to bound the $H^1$ norm of a scalar function *𝑢*, you have to integrate the square of every single partial derivative: $\int\sum_{i=1}^n u_i^2$, or use another positive definite quadratic form of the derivatives. **No semidefinite form will do**. If $Q$ positive semidefinite and $Q(\xi,\xi) = 0$ for some vector $\xi\neq 0$, then there is a function $u$ such that $\nabla u$ is always parallel to $\xi$, and therefore $\int Q(\nabla u,\nabla u) = 0$ while $\Vert u\Vert_{H^1}$ can be huge (and $\Vert u\Vert_{L^2}$ will not control $\Vert u\Vert_{H^1}$ either).

In Korn's inequality we integrate the quadratic form $K(\xi,\xi) =\sum(\xi_{ij}+\xi_{ij})^2$. It is semidefinite with a huge kernel: there is an $n(n-1)/2$ dimensional subspace along which $K$ is zero (skew-symmetric matrices, to be precise). Recalling that in the scalar case even one-dimensional kernel killed the estimate, how can we expect that $\int K(\nabla v,\nabla v)$ will control $\Vert u\Vert_{H^1}$? **Yet it does**.



