# Rate of convergence in iteration methods

See wiki, [Rate of convergence](https://en.wikipedia.org/wiki/Rate_of_convergence). And we copy it in the following:

---

## Rate of convergence

In [numerical analysis](https://en.wikipedia.org/wiki/Numerical_analysis), the speed at which a [convergent sequence](https://en.wikipedia.org/wiki/Limit_of_a_sequence) approaches its limit is called the **rate of convergence**. Although strictly speaking, a limit does not give information about any finite first part of the sequence, the concept of rate of convergence is of practical importance when working with a sequence of successive approximations for an [iterative method](https://en.wikipedia.org/wiki/Iterative_method), as then typically fewer iterations are needed to yield a useful approximation if the rate of convergence is higher. This may even make the difference between needing ten or a million iterations.

Similar concepts are used for [discretization](https://en.wikipedia.org/wiki/Discretization) methods. The solution of the discretized problem converges to the solution of the continuous problem as the grid size goes to zero, and the speed of convergence is one of the factors of the efficiency of the method. **However, the terminology in this case is different from the terminology for iterative methods**.

[Series acceleration](https://en.wikipedia.org/wiki/Series_acceleration) is a collection of techniques for improving the rate of convergence of a series discretization. Such acceleration is commonly accomplished with [sequence transformations](https://en.wikipedia.org/wiki/Sequence_transformation).



## Convergence speed for iterative methods

### Basic definition

Suppose that the [sequence](https://en.wikipedia.org/wiki/Sequence) $$\{x_k\}$$ to the number $$L$$.

- The sequence is said to **converge linearly** to $$L$$, if there exists a number $$\mu\in(0,1)$$ such that
  $$
  \lim_{k\rightarrow \infin} \frac{\Vert x_{k+1}-L\Vert}{\Vert x_k - L\Vert} = \mu,
  $$
  where $$\Vert\cdot\Vert$$ is some norm, and the number $$\mu$$ is called the **rate of convergence**.

- The sequence is said to **converge superlinearly** (i.e. faster than linearly) to $$L$$, if 
  $$
  \lim_{k\rightarrow \infin} \frac{\Vert x_{k+1}-L\Vert}{\Vert x_k - L\Vert} = 0.
  $$

- The sequence is said to **converge sublinearly** (i.e. slower than linearly) to $$L$$, if 
  $$
  \lim_{k\rightarrow \infin} \frac{\Vert x_{k+1}-L\Vert}{\Vert x_k - L\Vert} = 1.
  $$
  If the sequence **converges sublinearly** and additionally
  $$
  \lim_{k\rightarrow \infin} \frac{\Vert x_{k+2}- x_{k+1}\Vert}{\Vert x_{k+1} - x_{k}\Vert} = 1,
  $$
  then it is said that the sequence $$\{x_k\}$$ **converges logarithmically** to $$L$$.

- The next definition is used to distinguish superlinear rates of convergence. The sequence **converges with order** $$q$$ to $$L$$ for $$q>1$$ if 
  $$
  \lim_{k\rightarrow \infin} \frac{\Vert x_{k+1}-L\Vert}{\Vert x_k - L\Vert^q} < M
  $$
  for some positive constant $$M$$ (not necessarily less than 1). In particular, convergence with order 

  - $$q=2$$ is called **quadratic convergence**,
  - $$q=3$$ is called **cubic convergence**,
  - etc.

  This is sometimes called **Q-linear convergence**, **Q-quadratic convergence**, etc., to distinguish it from the definition below. The Q stands for "quotient", because the definition uses the quotient between two successive terms. A sequence that has a quadratic convergence implies that it has a superlinear rate of convergence. 

  A practical method to calculate the order of convergence for a sequence is to calculate the following sequence, which is converging to $$q$$ ($$q>1$$)
  $$
  q \approx \frac{\log\Vert\frac{ x_{k+1}- x_{k}}{ x_{k} - x_{k-1}}\Vert}{\log\Vert\frac{ x_{k}- x_{k-1}}{ x_{k-1} - x_{k-2}}\Vert}.
  $$
  

### Extended definition



The drawback of the above definitions is that these do not catch some sequences which still converge reasonably fast, but whose rate is variable, such as the sequence $$\{b_k\}$$ below. Therefore, the definition of rate of convergence is sometimes extended as follows. 

Under the new definition, the sequence $$\{x_k\}$$ converges with at least order $$q$$ if there exists a sequence $$\{\delta_k\}$$ such that 
$$
\Vert x_k-L\Vert \leq \delta_k \quad \text{ for all } k,
$$
and the sequence $$\{\delta_k\}$$ converges to zero with order $$q$$ according to the above Q-convergence definition. To distinguish it from that definition, this is sometimes called R-linear convergence, R-quadratic convergence, etc. (with the R standing for "root").

**Example**
$$
b_k = \left\{ \begin{align} &1 + \frac{1}{2^k} \quad k \text{ is even} \\
&1 \quad k \text{ is odd}
\end{align}
\right.
$$
note that the quotients in the definition of Q-linear convergence gets us in trouble. However,
$$
\Vert b_k - b_*\Vert \leq \frac{1}{2^k}=: \delta_k,
$$
and it is easy to check that $$\{\delta_k\}$$ converges Q-linearly with convergence rate $$\frac{1}{2}$$.



### Examples

See wiki [Rate of convergence](https://en.wikipedia.org/wiki/Rate_of_convergence). 



### Convergence speed for discretization methods

This is suitable for FEM, DG, ...

See wiki [Rate of convergence](https://en.wikipedia.org/wiki/Rate_of_convergence). 

