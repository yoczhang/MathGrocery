

# Distributive Gauss-Seidel for General Saddle Point Problem

The DGS for Stokes problem has been introduced in [DGS_for_Stokes](./DGS_for_Stokes.md), here we focus on the DGS method for the general saddle-point problem.

---

## Step 1 (Done)

Suppose we solve Stokes equations 
$$
\mathcal L U = F \tag{1}
$$
where 
$$
\begin{equation}
\mathcal L = \begin{pmatrix}
-\mu \Delta \quad \nabla \\
-\nabla\cdot \quad 0
\end{pmatrix}, \quad U = \begin{pmatrix}
\pmb u \\
p
\end{pmatrix}, \quad F = \begin{pmatrix}
\pmb f \\
g
\end{pmatrix}.
\end{equation}
$$
And we introduce the distributive matrix
$$
\mathcal M = \begin{pmatrix}
I \quad \nabla \\
0 \quad \mu\Delta_p
\end{pmatrix}.
$$
Since $$-\mu \Delta \nabla + \nabla \mu\Delta = (-\mu \Delta + \nabla \mu\nabla\cdot)\nabla = (\mu\nabla\times\nabla\times)\nabla = 0$$, therefore $$\mathcal L\mathcal M = \begin{pmatrix} -\mu\Delta \quad 0 \\ -\nabla\cdot \quad - \Delta_p \end{pmatrix}$$ is lower triangluar matrix.

---

## Step 2

Suppose we solve equations 
$$
\mathcal L U = F \tag{2}
$$
where 
$$
\begin{equation}\mathcal L = \begin{pmatrix}-\mu \Delta \quad \nabla \\-\nabla\cdot \quad I\end{pmatrix}, \quad U = \begin{pmatrix}\pmb u \\p\end{pmatrix}, \quad F = \begin{pmatrix}\pmb f \\g\end{pmatrix},\end{equation}
$$
with $$I$$ is the identity operator.

We need to find the distributive matrix $$\mathcal M$$ such that $$\mathcal L \mathcal M$$ is the lower triangluar matrix.