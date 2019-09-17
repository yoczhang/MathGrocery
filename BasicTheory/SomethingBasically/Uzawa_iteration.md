# Uzawa iteration

From wiki [Uzawa iteration](https://en.wikipedia.org/wiki/Uzawa_iteration)

---

In [numerical mathematics](https://en.wikipedia.org/wiki/Numerical_mathematics), the **Uzawa iteration** is an algorithm for solving [saddle point](https://en.wikipedia.org/wiki/Saddle_point) problems. It is named after [Hirofumi Uzawa](https://en.wikipedia.org/wiki/Hirofumi_Uzawa) and was originally introduced in the context of concave programming.



## Basic idea

We consider a saddle point problem of the form
$$
\begin{pmatrix}
A \quad B\\
B^* \quad 0
\end{pmatrix} \begin{pmatrix}
x_1 \\ x_2
\end{pmatrix} = \begin{pmatrix}
b_1 \\ b_2
\end{pmatrix},
$$
where $$A$$ is a symmetric [positive-definite matrix](https://en.wikipedia.org/wiki/Positive-definite_matrix). Multiplying the first row by $$B^*A^{-1}$$ and subtracting from the second row yields the upper-triangular system
$$
\begin{pmatrix}
A \quad B\\
0 \quad -S
\end{pmatrix} \begin{pmatrix}
x_1 \\ x_2
\end{pmatrix} = \begin{pmatrix}
b_1 \\ b_2- B^*A^{-1}b_1
\end{pmatrix},
$$
where $$S:= B^*A^{-1}B$$ denotes the [Schur complement](https://en.wikipedia.org/wiki/Schur_complement). Since $$S$$ is symmetric positive-definite, we can apply standard iterative methods like the [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) method or the [conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method) to
$$
Sx_2 = B^*A^{-1}b_1 - b_2
$$
in oder to compute $$x_2$$. The vector $$x_1$$ can be reconstructed by solving 
$$
A x_1 = b_1 - B x_2.
$$
It is possible to update $$x_1$$ alongside $$x_2$$ during the iteration for the Schur complement system and thus obtain an efficient algorithm.



## Implementation

We start the **conjugte gradient iteration** by computing the residual
$$
r_2:= B^*A^{-1}b_1 - b_2 - Sx_2 = B^*A^{-1}(b_1-Bx_2) -b_2 = B^*x_1 - b_2,
$$
of the Schur complement system, where $$x_1:= A^{-1}(b_1-Bx_2)$$ denotes the upper half of the solution vector matching the initial guess $$x_2$$ for its lower half. We complete the initialization by choosing the first search direction 
$$
p_2:= r_2.
$$
In each step, we compute 
$$
a_2:=Sp_2=B^*A^{-1}Bp_2 = B^*p_1
$$
