### The Nitsche's method

Here we introduce the Nitsche's method refering answer of StackExchange's question: [What is the general idea of Nitsche's method in numerical analysis?](https://scicomp.stackexchange.com/questions/19910/what-is-the-general-idea-of-nitsches-method-in-numerical-analysis)

Also, from the discussion of the answer, we find some other (maybe) useful references: in the MOFEM: [Periodic boundary conditions with Nitsche's method](http://mofem.eng.gla.ac.uk/mofem/html/nitsche_periodic.html)

---

The answer:

Nitsche's method is related to discontinuous Galerkin methods (indeed,  as Wolfgang points out, it is a precursor to these methods), and can be  derived in a similar fashion. Let's consider the simplest problem,  Poisson's equation:
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















