# Schur complement

See wiki [Schur complement](https://en.wikipedia.org/wiki/Schur_complement)

---

In [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra) and the theory of [matrices](https://en.wikipedia.org/wiki/Matrix_(mathematics)), the **Schur complement** of a [block matrix](https://en.wikipedia.org/wiki/Block_matrix) is defined as follows.

Suppose $$A, B, C, D$$ are respectively $$p\times p$$, $$p\times q$$, $$q\times p$$, and $$q\times q$$ matrices, and $$D$$ is invertible. Let
$$
M = \begin{bmatrix}
A \quad B\\
C \quad D
\end{bmatrix}
$$
so that $$M$$ is a $$(p+q)\times(p+q)$$ matrix.

Then the **Schur complement** of the block $$D$$ of the matrix $$M$$ is the $$p\times p$$ matrix defined by 
$$
M/D:= A - BD^{-1} C \quad \text{Quick rember: A, B order.}
$$
and, if $$A$$ is invertible, the Schur complement of the block $$A$$ of the matrix $$M$$ is the $$q\times q$$ matrix defined by 
$$
M/A:= D-CA^{-1}B. \quad \text{Quick rember: D, C order.}
$$
