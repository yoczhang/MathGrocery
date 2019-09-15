# Lagrange multiplier

See wiki [Lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier)

---

In mathematical optimization, the method of Lagrange multiplier is a strategy for finding the local maxima and minima of a function subject to equality constraints (i.e., subject to the condition that one or more [equations](https://en.wikipedia.org/wiki/Equation) have to be satisfied exactly by the chosen values of the variables). The basic idea is to convert a constrained problem into a form such that the [derivative test](https://en.wikipedia.org/wiki/Derivative_test) of an unconstrained problem can still be applied. Once [stationary points](https://en.wikipedia.org/wiki/Stationary_point) have been identified from the first-order necessary conditions, the [definiteness](https://en.wikipedia.org/wiki/Definiteness_of_a_matrix) of the [bordered Hessian matrix](https://en.wikipedia.org/wiki/Bordered_Hessian) determines whether those points are maxima, minima, or [saddle points](https://en.wikipedia.org/wiki/Saddle_point).

The Lagrange multiplier theorem roughly states that at any [stationary point](https://en.wikipedia.org/wiki/Stationary_point) of the function that also satisfies the equality constraints, the [gradient](https://en.wikipedia.org/wiki/Gradient) of the function at that point can be expressed as a [linear combination](https://en.wikipedia.org/wiki/Linear_combination) of the gradients of the constraints at that point, with the Lagrange multipliers acting as [coefficients](https://en.wikipedia.org/wiki/Coefficient). The relationship between the gradient of the function and gradients of the constraints rather naturally leads to a reformulation of the original problem, known as the **Lagrangian function**.

The great advantage of this method is that it allows the optimization to be solved without explicit [parameterization](https://en.wikipedia.org/wiki/Parameterization) in terms of the constraints. As a result, the method of Lagrange multipliers is widely used to solve challenging constrained optimization problems. The method can be summarized as follows: in order to find the stationary points of a function $$f(x)$$ subject to the equality constraint $$g(x)=0$$, form the Lagrangian function
$$
\mathcal L(x,\lambda) = f(x) + \lambda g(x)
$$
and find the stationary points of $$\mathcal L$$ considered as a function of $$x$$ and the Lagrange multiplier $$\lambda$$. The solution corresponding to the original constrained optimization is always a saddle point of the Lagrangian function.

........ (TODO: see wiki)

