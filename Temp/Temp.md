# Temp

## 记录一下各种括号

> \newcommand*{\snorm}[1]{|#1|} % the seminorm norm: |..|
> \newcommand*{\dnorm}[1]{\lVert#1\rVert} % the double norm: ||..||
> \newcommand*{\tnorm}[1]{\interleave#1\interleave} % the triple norm: |||..|||
> \newcommand*{\anglebrackets}[1]{\langle#1\rangle} % the angle brackets: <...>



## 8_HHO_StokesDarcyFracture

1. 因为用的 symmetric gradient，参考 2015 (CMAME) A hybrid high-order locking-free method for linear elasticity on general meshes，我们在构造时可以采用 $k\geq 0$, 在误差分析时采用 $k\geq 1$.
2. 改到 (4.23) 式，将 $\mathcal A_F^\Gamma$ 改为 $\mathcal D_F$.



