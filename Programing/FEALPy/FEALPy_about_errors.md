### FEALPy 中计算误差的函数

这里记录一下 FEALPy 非协调 VEM (`NonConformingVirtualElementSpace2d`) 中, 计算 $H1$ 和 $L2$ 误差的函数.

在 `PoissonNCVEMModel` 中有 `L2_error()` 和 `H1_semi_error()` 两个函数, 这两个函数的主要思路是:

- $L2$: 将所取得的离散解 `uh` 投影到 smspace 中, 为 `S`, 然后计算 `u-S` 在单元中的积分.
- $H1$-semi: 将所取得的离散解 `uh` 投影到 smspace 中, 为 `S`, 然后计算 `grad u - grad S` 在单元中的积分.

其中 $L2$ 误差正常计算, $H1$-semi 误差并不是常见的, 因为对于 DG, VEM, HHO 这几种非协调元方法的话, 通常用的是 energy norm.

