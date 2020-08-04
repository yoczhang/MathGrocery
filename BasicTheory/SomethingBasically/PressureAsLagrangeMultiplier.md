The original see [Pressure as a Lagrange Multiplier](https://scicomp.stackexchange.com/questions/7474/pressure-as-a-lagrange-multiplier)

---

####The question

In the incompressible Navier-Stokes equations,

$$ \rho\left(\mathbf{u}_t + (\mathbf{u} \cdot \nabla)\mathbf{u}\right) = - \nabla p + \mu\Delta\mathbf{u} + \mathbf{f}\\ \nabla\cdot\mathbf{u} = 0 $$

the pressure term is often referred to as a Lagrange multiplier enforcing the incompressibility condition. 

In what sense is this true? Is there a formulation of the incompressible Navier-Stokes equations as an optimization problem subject to the incompressiblity constraint? If so, is there a numerical analog in which the equations of incompressible fluid flow are solved within an optimization framework?

---

####The answer

This is most easily seen by considering the stationary Stokes equations 

$$ -\mu \Delta u + \nabla p = f \\  \nabla \cdot u = 0 $$ 

which is equivalent to the problem 

$$ \min_u \frac\mu 2 \|\nabla u\|^2 - (f,u) \\  \text{so that} \; \nabla\cdot u = 0. $$ 

If you write down the Lagrangian and then the optimality conditions of this optimization problems, you will find that indeed the pressure is the Lagrange multiplier.

This equivalence between problems is not exploited in any numerical scheme (that I know of) but it is an important tool in the analysis because it shows that the Stokes equations are essentially the Poisson equation on a linear subspace. The same holds true for the time-dependent Stokes equations (which corresponds to the heat equation on the subspace) and it can be extended to the Navier-Stokes equations.

(And the time dependent problem) Not as an optimization problem -- the solution of the heat equation does not minimize anything  (though it's the stationary point of a Lagrangian function). But you can formulate the Stokes equations as follows: Find $u \in H_\text{div}$ so that $(u_t,\varphi)+(\nabla u,\nabla\varphi)=(f,\varphi)$ for all  $\varphi\in \{v\in H_\text{div}: \nabla \cdot v = 0\}$ subject to the constraint that $\nabla \cdot u = 0$. Note that I have chosen the test  space smaller than the trial space and so the left and right hand side of the variational equation will not be equal. The difference is the pressure.