阅读下面这个提问与回答, [Why is Korn's inequality not trivial?](https://math.stackexchange.com/questions/248358/why-is-korns-inequality-not-trivial)

这个回答中虽然没有完全解释清楚, 提问者的问题, 但是仍然提供了很好的解释.

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

