

# Inequality

---

## Introduction

The following inequlities mostly from HHO papers.



## Presetting

et 

- $$\Omega$$ be the domain;

- $$\mathcal T_h$$ be the collection of mesh;

- $$\mathcal F_h$$ be the collection of faces (or edges);

- $$T\in \mathcal T_h$$ be the element (or cell);

- We will use the notation $$\mathring{W^{1,p}}(\Omega)$$ to denote the subset of $$W^{1,p}(\Omega)$$, consisting of functions whose trace on $$\partial \Omega$$ is zero, that is 
  $$
  \mathring{W^{1,p}}(\Omega) = \{ v\in W^{1,p}(\Omega): v|_{\partial \Omega} = 0 \text{ in } L^2(\partial \Omega) \}.
  $$
  Similaryly, we let $$\mathring{W^{k,p}}(\Omega)$$ denote the subset of $$W^{1,p}(\Omega)$$ consisting of functions whose derivatives of order $$k-1$$ are in $$\mathring{W^{1,p}}(\Omega)$$, i.e.
  $$
  \mathring{W^{k,p}}(\Omega) = \{ v\in W^{k,p}(\Omega): v^{(\alpha)}|_{\partial \Omega} = 0 \text{ in } L^2(\partial \Omega) \quad \forall |\alpha|<k \}.
  $$

- 



---

## Projection inequality

One can prove that there exists a real number $$ C_{app} $$ depending on $$ \varrho $$ and $$ l $$, but independent of $$ h $$, such that, for all  $$T\in\mathcal{T}_h$$ , the following holds: For all $$ s\in\{1,...,l+1\} $$ and all $$ v\in H^s(T) $$, 
$$
\begin{align}
	|v-\pi_T^l v|_{m,T} + h_T^{\frac{1}{2}}|v-\pi_T^l v|_{m,\partial T} \leq C_{app}h_T^{s-m}|v|_{s,T} \quad \forall m\in\{0,...(s-1)\}.
\end{align}
$$


## Poincare inequality

- Continuous Poincare inequality

  - From [^2007_1]

    If $$v\in W_0^{1,p}(\Omega)$$ (actually, there only needs on the part of $$\partial\Omega$$, the trace of $$v = 0$$).
    $$
    \Vert v \Vert_{W^{1,p}} \leq C_\Omega | v |_{W^{1,p}}.
    $$

  - From [^2007_2]

    **定理 2.3.2 **(Poincare 不等式) 如果 $$\Omega$$ 为连通且在一个方向上有界的区域, 则对每个非负整数 $$m$$, 存在 $$C_m=const > 0$$, 使得
    $$
    \Vert v \Vert_{m,\Omega} \leq C_m \vert v \vert_{m,\Omega}, \quad v \in H_0^m(\Omega).
    $$
    

  - From[^2016]

    ![2016 Poincare inequality](/Users/yczhang/Documents/MathGrocery/BasicTheory/Inequality/2016 Poincare inequality.png)

  [^2007_1]: 2007 The Mathematical Theory of Finite Element Methods Third Edition.pdf
  [^2007_2]: 2007 (王烈衡-许学军) 有限元方法的数学基础.pdf
  [^2016]: 2016 (Volker John) Finite element methods for incompressible flow problems.pdf

- Discontinuous Poincare inequality