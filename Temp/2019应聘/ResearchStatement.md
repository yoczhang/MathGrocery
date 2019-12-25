### Research statement (研究计划)

本文研究的主要方向是自由流, 多孔介质流及其耦合问题的高阶间断有限元方法的数值模拟和理论分析. 其中自由流区域考虑三种流体方程:  Stokes方程, 带阻尼项的 Stokes 方程, 以及拟牛顿 Stokes 方程. 多孔介质流由 Darcy 方程给出.

This thesis studies the numerical simulation and theoretical analysis of free fluid, porous media fluid and their coupling problems by high order discontinuous finite element methods. Three kinds of fluid equations are considered in the free fluid region: the Stokes equation, the Stokes equation with damping term and the quasi-Newton Stokes equation. The porous media fluid is given by the Darcy equation.

由于自由流和多孔介质区域通常有复杂的几何形状, 使得耦合问题的交界面也具有不规则形状. 实践中得到在界面处匹配的的网格剖分是很困难的, 而多边形网格可以很好的解决这一问题, 所以本文采用了支持多边形网格的数值算法. 另外, 在保持数值格式和网格尺寸不变的情况下, 高阶元可以达到更高的精度, 可以更好的逼近所求问题. 因此支持多边形网格和可能的任意阶的离散化方法是本文的研究方向, 相关的研究在过去的几年中经历了蓬勃的发展. 并且在此过程中开发了很多新的设计和分析方法, 它们借鉴了其他数学分支的思想 (如拓扑学和几何学), 或者扩展了有限元或有限体积法的原始思想.

Due to the complex geometry of free flow region and porous media, the interface of the coupling problem usually has irregular shape. In practice, it is very difficult to get the matching mesh at the interface, the polygonal mesh can solve this problem well. In addition, under the condition of keeping the numerical scheme and mesh size unchanged, higher order polynomial can achieve higher precision and better approach to the problem.  Therefore, the main research of this thesis is the high order discontinuous finite element methods supporting the polygon mesh. The related research has experienced a vigorous development over the last decade. Novel approaches to the design and analysis have been developed or rediscovered borrowing ideas from other branches of mathematics such as topology and geometry, or expanding past their initial limits the original ideas underlying finite element or finite volume methods. 

本文主要采用间断 Galerkin (discontinuous Galerkin, DG) 和杂交高阶 (hybrid high-order method, HHO) 两种方法来数值离散上述流体方程. 首先我们从经典的内部加罚 (interior penalty, IP) DG 方法开始研究. 由于 DG 方法的基函数是采用的分片多项式, 所以其格式可以不受单纯形网格的限制, 经过近几年的发展, 支持任意多边形网格的 DG 方法在理论上逐渐完善. 在此基础上, 将 DG 格式应用到带有阻尼项的 Stokes 问题中, 在该格式中我们采用了 $ \mathbb P_k^d-\mathbb P_k $  等阶元来离散速度和压力. 我们统一分析了离散问题的对称, 非对称和不完全对称三种格式的适定性, 并给出了关于速度的最优 $ H^1 $ 误差估计, 以及关于压力的最优 $ L^2 $ 误差估计.

In this paper, discontinuous Galerkin (DG) and hybrid high-order method (HHO) are used to discretize the above-mentioned fluid equations. Firstly, we start from the classical interior penalty (IP) DG method. Since the basis functions of the DG method are piecewise polynomial, its scheme can be independent of the simplex mesh, and after recent years of development, the DG methods that support polygon mesh are gradually perfecting in theory. In the thesis, the DG scheme is applied to Stokes problem with damping term. In this scheme, we use PK equal order element to discretize velocity and pressure. We analyze the well posedness of symmetric, asymmetric and incomplete symmetric schemes of discrete problem, and give the optimal H1 error estimation of speed, as well as the optimal L2 error estimation of pressure.



接下来的一部分工作是应用 HHO 方法求解自由流和多孔介质流耦合问题. 此耦合问题在工程中有着广泛的应用背景, 例如石油开采, 地表水和地下水的交互, 工业过滤, 血液在血管和器官之间的流动等. 在研究中, 该耦合问题的数学模型由三部分构成, 分别是自由流区域方程, 多孔介质区域方程, 以及三个交界面条件. 其中, 为了简化模型, 自由流区域我们考虑不可压 Stokes 方程, 多孔介质区域由 Darcy 方程给出. 在自由流区域, 使用 Laplace 形式的 Stokes 方程; 在多孔介质区域, 速度和压力两个变量以混合一阶形式给出. 采用分片 $ (\mathbb P_k(\mathcal T_h)^d $, $ \mathbb P_k(\mathcal F_h)^d)\times \mathbb P_k(\mathcal T_h) $ $ (k\geq 0) $ 多项式离散整体的速度和压力, 交界面处速度的法向连续性被强制在离散空间中. HHO 方法中, 对梯度和散度算子需要进行重构, 利用重构算子得到最终的离散格式, 并利用鞍点理论证明了离散问题的适定性. 给出了能量范数下的误差分析, 并用数值实验验证了误差分析中的收敛阶以及 HHO 方法的优点. 目前在推导 Stokes-Darcy 耦合问题的速度的最优 $ L^2 $ 误差估计时仍存在一定的困难, 如何构造合适的插值算子来估计速度的 $ L^2 $ 误差将会是一个有意义的工作.

The next part of the work is to apply the HHO method to solve the problem of free flow and porous media flow coupling. This coupling problem has a wide range of applications in engineering, such as oil extraction, interaction between surface water and groundwater, industrial filtration, blood flow between blood vessels and organs, etc. In the thesis, the mathematical model of the coupling problem consists of three parts, namely, the free fluid region equation, the porous medium region equation, and the three crossinterface conditions. Among them, in order to simplify the model, we consider the incompressible Stokes equation in the free fluid region and the Darcy equation in the porous media region. The Stokes equation is used in Laplace form and the Darcy equation in the second-order elliptical form. The equal-order piecewise polynomials $ (\mathbb P_k(\mathcal T_h)^d ,\mathbb P_k(\mathcal F_h)^d)\times \mathbb P_k(\mathcal T_h) $ $ (k\geq 0) $  are used to approximate the velocity and pressure in the whole domain, the continuity of velocity at the interface is forced in discrete space. In the HHO method, the gradient and divergence operators need to be reconstructed, the final discrete scheme is obtained by reconstructing the operator, and the well-posedness of the discrete system is proved by the classical saddle point theory. The error analysis under the energy norm is given, the convergence order and the advantages of HHO method in the error analysis are verified by numerical experiments. However, in the current framework of error estimates, the $ L^2 $-error estimate of Stokes equation involves the energy-error estimation of Darcy equation, but we cannot obtain a higher order error estimation from this term, which makes us unable to obtain the optimal error estimation. How to construct appropriate interpolation operators to estimate the $ L^2 $-error of velocity will be a meaningful work.

以上模型的自由流区域考虑的都是牛顿流体 (即粘度是恒定的常数), 接下来我们尝试用 HHO 方法求解一类拟牛顿流体问题, 拟牛顿流体是对复杂流体 (非牛顿流体) 的一种简单近似. 在此模型中, Stokes 方程采用应力张量形式, 令 $ |\nabla_s\bm| $ 表示剪切速率, 粘度函数则由非线性函数 $ \nu(|\nabla_s \bm{u} |) $ 给出, 并且要求 $ \nu $ 满足一定的正则性条件. 此处考虑了两种经典的粘度函数: power law 粘度函数和 Carreau\textquoteright s law 粘度函数. 数值离散时利用 HHO 方法重构了梯度和散度算子, 以及采用了经典的 Picard 来处理非线性项. 针对离散系统, 给出了 HHO 方法的静力凝聚法, 并给出了速度和压力的在能量范数下的最优误差估计. 

The free fluid equation of the above models is only Newtonian fluid (i.e. viscosity is constant constant), and then we try to use HHO method to solve a class of quasi-Newtonian fluid problems, which is a simple approximation of complex fluids (non-Newtonian fluids). In this model, the Stokes equation reads in the stress tensor form, the viscosity function is given by the nonlinear function  $ \nu(|\nabla_s \bm{u} |) $ , where $ |\nabla_s\bm{u}| $ represents the shear rate, and requires that the $ \nu $ meets certain regular conditions. Two classic viscosity functions are considered here: the power law viscosity function and the Carreau's law viscosity function. The gradient and divergence operators are reconstructed by using the HHO method in the discrete scheme, moreover, the classical Picard is used to handle nonlinear terms. For discrete systems, the static condensation of the HHO method is given, and the optimal error estimation of velocity and pressure under the energy norm is given. 

---

1. Cockburn, B., Di Pietro, D. A., & Ern, A. (2016). Bridging the  hybrid high-order and hybridizable discontinuous Galerkin methods. *ESAIM: Mathematical Modelling and Numerical Analysis*, *50*(3), 635-650.
2. Cockburn, B., Gopalakrishnan, J., & Lazarov, R. (2009). Unified  hybridization of discontinuous Galerkin, mixed, and continuous Galerkin  methods for second order elliptic problems. *SIAM Journal on Numerical Analysis*, *47*(2), 1319-1365.
3. Girault, V., Kanschat, G., & Rivière, B. (2014). Error analysis for a monolithic discretization of coupled Darcy and Stokes problems. *Journal of Numerical Mathematics*, *22*(2), 109-142.
4. Cesmelioglu, A., Rhebergen, S., & Wells, G. N. (2020). An  embedded–hybridized discontinuous Galerkin method for the coupled  Stokes–Darcy system. *Journal of Computational and Applied Mathematics*, *367*, 112476.



---

将来的工作计划

1. 针对 Stokes-Darcy 耦合方程的速度, 推导 $L^2$ 范数下的误差估计.
2. 粘弹性和粘塑性问题.











我们的研究工作主要是围绕上述模型进行展开, 分别设计数值格式进行求解. 



在今后的工作中我们仍将主要研究 HHO 方法在上述模型中的应用, 并尝试给出上述模型方程的后验误差分析.







