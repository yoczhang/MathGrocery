# Frechet and Gateaux derivatives

1. See wiki [Fréchet derivative](https://en.wikipedia.org/wiki/Fr%C3%A9chet_derivative)
2. See ./subfiles/different concepts of derivates.pdf

---

我们最初接触的导数是 $$X\rightarrow \mathbb R$$ 的一种情况, F 导数是 $$X\rightarrow Y$$ 的推广 ( $$X$$, $$Y$$ 是两个向量空间 (向量空间是无穷维的)), 而 G 导数是更进一步的看做是 F 导数上的方向导数, 所以我们大致会有如下的关系
$$
\text{ ordinary derivative } \subset \text{ F derivative } \subset \text{ G derivative },
$$
也就是说, G 可导但是不一定 F 可导.

---

In [mathematics](https://en.wikipedia.org/wiki/Mathematics), the **Fréchet derivative**is a [derivative](https://en.wikipedia.org/wiki/Derivative) defined on [Banach spaces](https://en.wikipedia.org/wiki/Banach_space). Named after [Maurice Fréchet](https://en.wikipedia.org/wiki/Maurice_René_Fréchet), it is commonly used to generalize the derivative of a [real-valued function](https://en.wikipedia.org/wiki/Real-valued_function)of a single real variable to the case of a [vector-valued function](https://en.wikipedia.org/wiki/Vector-valued_function) of multiple real variables, and to define the [functional derivative](https://en.wikipedia.org/wiki/Functional_derivative) used widely in the [calculus of variations](https://en.wikipedia.org/wiki/Calculus_of_variations).

Generally, it extends the idea of the derivative from real-valued [functions ](https://en.wikipedia.org/wiki/Function_(mathematics))of one real variable to functions on Banach spaces. The Fréchet derivative should be  contrasted to the more general [Gateaux derivative](https://en.wikipedia.org/wiki/Gateaux_derivative) which is a generalization of the classical [directional derivative](https://en.wikipedia.org/wiki/Directional_derivative).

The Fréchet derivative has applications to nonlinear problems throughout [mathematical analysis](https://en.wikipedia.org/wiki/Mathematical_analysis) and physical sciences, particularly to the calculus of variations and much of nonlinear analysis and [nonlinear functional analysis](https://en.wikipedia.org/wiki/Nonlinear_functional_analysis).



## Definition of Fréchet derivative

Let $$V$$ and $$W$$ be normed vector spaces, and $$U\subset V$$ be an open subset of $$V$$. A function $$f: U\rightarrow W$$ is called **Fréchet differentiable** at $$x\in U$$ if there exists a bounded linear operator $$A: V\rightarrow W$$ such that, 
$$
\lim_{\Vert h \Vert\rightarrow 0} \frac{\Vert f(x+h)-f(x)-Ah \Vert_W}{\Vert h\Vert_V} = 0.
$$
The limit here is meant in the usual sense of a limit of a function defined on a metric space, using $$V$$ and $$W$$ as the two metric spaces, and the above expression as the function of argument $$h$$ in $$V$$. As a consequence, it must exist for all sequences $$\{h_n\}_{n=1}^\infin$$ of non-zero elements of $$V$$ which converge to the zero vector, i.e. $$h_n\rightarrow 0$$. Equivalently, the first-order expansion holds, [Landau notation](https://en.wikipedia.org/wiki/Landau_notation) (i.e. $$O(\cdot)$$, $$o(\cdot)$$)
$$
f(x+h) = f(x) + Ah + o(h).
$$
If there exists such as an operator $$A$$, it is unique, so we write $$Df(x) = A$$ and call it the **Fréchet derivative** of $$f$$ at $$x$$. A function $$f$$ that is **Fréchet differentiable** for any point of $$U$$ is said to be $$C^1$$ if the function 
$$
Df: U \rightarrow B(V,W); \quad \text{and}\quad x\rightarrow Df(x)
$$
is continuous. Note that this is not the same as requiring that the map $$Df(x):V\rightarrow W$$ be continuous for each value of $$x$$ (which is assumed; bounded and continuous are equivalent). $$Df$$ and $$Df(x)$$ have the different meanings.

This notion of derivative is a generalization of the ordinary derivative of a function on the real numbers $$f: \mathbb R \rightarrow \mathbb R$$ since the linear maps from $$\mathbb R$$ to $$\mathbb R$$ are just multiplication by a real number. In this case, $$Df(x)$$ is the function $$t\rightarrow f'(x)t$$.



## Properties of Fréchet derivative

- A function differentiable at a point is continuous at that point.

- Differentiation is a linear operation in the following sense: if $$f$$ and $$g$$ are two maps $$V\rightarrow W$$ which are differentiable at $$x$$, and $$r$$ and $$s$$ are scalars (two real or complex numbers), then $$rf+sg$$ is differentiable at $$x$$ with $$D(rf+sg)(x)=rDf(x)+sDg(x)$$.

- The chain rule is also vaild in this context: if $$f:U\rightarrow Y$$ is differentiable at $$x$$ in $$U$$, and $$f:Y\rightarrow W$$ is differentiable at $$y=f(x)$$, then the composition $$g\circ f$$ is differentiable at $$x$$ and the derivative is the composition of the derivatives:
  $$
  D(g\circ f)(x) = Dg(f(x))\circ Df(x).
  $$



## Fréchet derivative in finite dimensions

The Fréchet derivative in finite-dimensional spaces is the usual (ordinary) derivative. In particular, it is represented in coordinates by the [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant). 

Supplse that $$f$$ is a map, $$f: U\subset \mathbb R^n \rightarrow \mathbb R^m$$ with $$U$$ an open set. If $$f$$ is Fréchet differentiable at a point $$a\in U$$, then its derivative is 