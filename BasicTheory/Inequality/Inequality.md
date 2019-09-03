# Inequality

---

## Introduction

The following inequlities mostly from HHO papers.



## Presetting

let 

- $$\Omega$$ be the domain;
- $$\mathcal T_h$$ be the collection of mesh;
- $$\mathcal F_h$$ be the collection of faces (or edges);
- $$T\in \mathcal T_h$$ be the element (or cell);
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
- Discontinuous Poincare inequality