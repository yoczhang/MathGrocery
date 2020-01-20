# Euler formulas

参考 [欧拉公式到底 (总共) 有多少种形式啊? 各怎样表达?](https://www.zybang.com/question/80e9155c7b257ef4e2067e1d4a921486.html)

在数学历史上有很多公式都是欧拉 (Leonhard Euler) 发现的, 它们都称为欧拉公式, 分散在各个数学分支之中. 如复变函数中的欧拉幅角公式 (将复数、指数函数与三角函数联系起来); 拓扑学中的欧拉多面体公式; 初等数论中的欧拉函数公式; 分式公式等等.

- 分式公式中的欧拉公式
  $$
  \begin{align}
  &\frac{a^r}{(a-b)(a-c)}+\frac{b^r}{(b-c)(b-a)}+\frac{c^r}{(c-a)(c-b)}\\
  =&\left\{ 
  \begin{array}
  0 0, \ r = 0,1,\\
  1, \ r = 2,\\
  a+b+c, \ r = 3.
  \end{array}
  \right.
  \end{align}
  $$
  对于 $r\geq 4$ 时, 可以参考知网中 =="欧拉(Euler)公式的一个推广"== 这篇文章.

  

- 复变函数论中的欧拉公式
  $$
  e^{ix} = \cos(x) + i \sin(x),
  $$
  $e$ 是自然对数的底, $i$ 是虚数单位.

  它将三角函数的定义域扩大到复数, 建立了三角函数和指数函数的关系, 它在复变函数论里占有非常重要的地位.

  将公式里的 $x$ 换成 $-x$, 得到
  $$
  e^{-ix} = \cos(x) - i \sin(x),
  $$
  则可以分别得到
  $$
  \sin(x) = (e^{ix}-e^{-ix})/(2i), \quad \cos(x) = (e^{ix}+e^{-ix})/(2).
  $$
  另外将 $e^{ix} = \cos(x) + i \sin(x)$ 中 $x$ 取成 $\pi$, 则有
  $$
  e^{i\pi}+1 = 0,
  $$
  这个恒等式也叫做欧拉公式,它是数学里最令人着迷的一个公式, 它将数学里最重要的几个数学联系到了一起: 

  两个超越数: 自然对数的底 $e$, 圆周率 $\pi$;

  两个单位: 虚数单位 $i$ 和自然数的单位 1;

  以及数学里常见的0.

  数学家们评价它是 "上帝创造的公式", 我们只能看它而不能理解它.

  

- 三角形中的欧拉公式

  设 $R$ 为三角形外接圆半径, $r$ 为内切圆半径, $d$ 为外心到内心的距离, 则:
  $$
  d^2 = R^2 - 2Rr.
  $$



- 拓扑学里的欧拉公式

  - 对于多面体 $P$, 有
    $$
    V+ F- E = X(P),
    $$
    其中 $V$ 是多面体 $P$ 的顶点个数, $F$ 是多面体 $P$ 的面数, $E$ 是多面体 $P$ 的棱的条数, $X(P)$ 是多面体 $P$ 的欧拉示性数.

    如果多面体 $P$ 可以同胚于一个球面 (可以通俗地理解为能吹胀而绷在一个球面上), 那么 $X(P)=2$, 如果 $P$ 同胚于一个接有 $h$ 个环柄的球面, 那么 $X(P) = 2 - 2h$. 

    $X(P)$ 叫做 $P$ 的欧拉示性数, 是拓扑不变量, 就是无论再怎么经过拓扑变形也不会改变的量, 是拓扑研究的范围.

  - 对于经常用到的网格剖分中 (参考 ==2017 (ANM) A discrete divergence free weak Galerkin finite element method for the Stokes equations==)

    Let $\mathcal{T}_h$ be a partition of the domain $\Omega􏰼$ consisting of a set of polyhedra satisfying a set of conditions specified in [16]. In addition, we assume that all the elements $T\in\mathcal{T}_h$ are convex. Denote by $\mathcal{F}_h$ the set of all edges in 2D or faces in 3D in $\mathcal{T}_h$, and let $\mathcal{F}_h^0=\mathcal{F}_h\backslash \partial\Omega$ be the set of all interior edges or faces.

    - For two dimensional space

      For a given partition $\mathcal{T}_h$, let $\mathcal{V}_h^0$ be the set of all interior vertices. Let $N_F=card(\mathcal{F}_h^0)$, $N_V=card(\mathcal{V}_h^0)$, and $N_K=card(\mathcal{T}_h)$. It is known based on the Euler formula that for a partition consisting of convex polygons, then
      $$
      N_F + 1 = N_V + N_K.
      $$
      For a mesh $\mathcal{T}_h$ with hanging nodes, the above relation is still true if we treat the hanging nodes as vertices.

    - For three dimensional space

      Let partition $\mathcal{T}_h$ be a partition of $\Omega\subset \mathbb{R}^3$ consisting of polyhedra without hanging nodes. Recall $N_F=card(\mathcal{F}_h^0)$, $N_V=card(\mathcal{V}_h^0)$, and $N_K=card(\mathcal{T}_h)$. Denote by $\mathcal{E}_h$ all the edges in $\mathcal{T}_h$ and let $\mathcal{E}_h^0=\mathcal{E}_h\backslash \partial\Omega$. Let $N_E=card(\mathcal{E}_h^0)$. 

      It is known based on the Euler formula that for a partition consisting of convex polygons, then
      $$
      N_V+N_F + 1 = N_E + N_K.
      $$
      

  

- 初等数论里的欧拉公式

  欧拉 $\varphi$ 函数: $\varphi(n)$ 是所有小于 $n$ ($n$ 是一个正整数) 的正整数里, 和 $n$ 互素的整数的个数.

  欧拉证明了下面的式子:

  如果 $n$ 的标准素因子分解式是 $p_1^{a_1}*p_2^{a_2}*\cdots *p_m^{a_m}$, 其中 $p_j^{a_j} (j=1,2,\cdots, m)$ 都是素数, 而且两两不等. 则有
  $$
  \varphi(n) = n(1-\frac{1}{p_1})(1-\frac{1}{p_2})\cdots (1-\frac{1}{p_m}).
  $$
  利用容斥原理可以证明它.