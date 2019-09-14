# Lipschitz continuous

Firstly, we can see the wiki [Lipschitz continuity](https://en.wikipedia.org/wiki/Lipschitz_continuity).

## Definitions

Given two metric spaces $$(X, d_X)$$ and $$(Y, d_Y)$$, where $$dX$$ denotes the metric on the set $$X$$ and $$dY$$ is the metric on set $$Y$$, a function $$f : X \rightarrow Y$$ is called **Lipschitz continuous** if there exists a real constant $$K \geq 0$$ such that, for all $$x_1$$ and $$x_2$$ in $$X$$,
$$
d_Y(f ( x_1 ),f ( x_2 )) \leq d_X(x_1, x_2).
$$
Any such K is referred to as a Lipschitz constant for the function f. The smallest constant is sometimes called **the (best) Lipschitz constant**; however, in most cases, the latter notion is less relevant. If $$K = 1$$ the function is called a short map, and if $$0 \leq K < 1$$ and $$f$$ maps a metric space to itself, the function is called a contraction.

In particular, a real-valued function $$f : \mathbb R \rightarrow \mathbb R$$ is called **Lipschitz continuous** if there exists a positive real constant $$K$$ such that, for all real $$x_1$$ and $$x_2$$,
$$
|f(x_{1})-f(x_{2})|\leq K|x_{1}-x_{2}|.
$$
In this case, $$Y$$ is the set of real numbers $$\mathbb R$$ with the standard metric $$d_Y(y_1, y_2) = |y_1 − y_2|$$, and $$X$$ is a subset of $$\mathbb R$$. 



More generally, a function $$f$$ defined on $$X$$ is said to be **Hölder continuous** or to satisfy a **Hölder condition** of order $$\alpha > 0$$ on $$X$$ if there exists a constant $$M > 0$$ such that

$$
d_{Y}(f(x),f(y))\leq Md_{X}(x,y)^{\alpha }
$$
for all $$x$$ and $$y$$ in $$X$$. Sometimes a Hölder condition of order $$\alpha$$ is also called a uniform Lipschitz condition of order $$\alpha>0$$. 



## Relationships

We have the following chain of strict inclusions for functions over a closed and bounded (i.e. compact) non-trivial interval of the real line:

Continuously differentiable $$\subset$$ Lipschitz continuous $$\subset$$ $$\alpha$$-Hölder continuous $$\subset$$ uniformly continuous = continuous, where $$0<\alpha\leq 1$$. 

We also have:

Lipschitz continuous $$\subset$$ absolutely continuous $$\subset$$ bounded variation $$\subset$$ differentiable almost everywhere.



## Properties

See the wiki for details.

One property is that, for a Lipschitz continuous function, there exists a double cone (white) whose origin can be moved along the graph so that the whole graph always stays outside the double cone:

![Lipschitz_Visualisierung](/Users/yczhang/Documents/MathGrocery/BasicTheory/SomethingBasically/subfiles/Lipschitz_Visualisierung.gif)



## Understanding Lipschitz Continuity

You can not pick $$K$$ sufficiently enough for a function to be Lipschitz continuous if they are not. That's the main point of that kind of continuity. If $$f$$ is not lipschitz continuous, and you say that $$K=10^6$$, I can find an pair of points $$x_1$$ and $$x_2$$ such that $$|f(x_1)-f(x_2)|\geq 10^6|x_1-x_2|$$. 

Think about the mean value theorem and Lipschitz continuity.

Mean value theorem says if $$f$$ is is continuous at $$[a,b]$$ and differentiable at $$(a,b)$$ then
$$
\exist\, c\in(a,b) \text{ such that } \frac{f(b)-f(a)}{b-a} = f'(c).
$$
Lipschitz says that 
$$
\exist\, K>0,\, \forall a,b\in D_f, \text{ such that } \frac{|f(b)-f(a)|}{|b-a|} \leq K.
$$
Then if the derivative of $$f$$ as a function is bounded, then $$f$$ will be Lipschitz.

Consider the case:

$$f(x) = \sqrt{x}$$ for $$x\in[0,1]$$, then $$f$$ is not Lipschitz, since $$\sup_{x\in[0,1]}f'(x) = \lim_{x\rightarrow 0} = +\infin$$.

Also, as an additional note if a function $$f$$ defined on $$S\subset \mathbb R$$ is Lipschitz continuous then $$f$$ is uniformly continuous on $$S$$.



