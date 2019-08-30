# Optimization methods

This paper is from [最速下降法/steepest descent，牛顿法/newton，共轭方向法/conjugate direction，共轭梯度法/conjugate gradient 及其他](http://www.codelast.com/?p=2573).

最优化 (Optimization) 是应用数学的一个分支，它是研究在给定约束之下如何寻求某些因素(的量)，以使某一(或某些)指标达到最优的一些学科的总称。

我们设定

- The space: $$V = \{\bf x| \bf{x} \in \mathbb R^n \} $$ or $$V = H_0^1(\Omega)$$;

- The objective function: $$f(\mathbf x): V \rightarrow \mathbb R$$;

- [Line search](https://en.wikipedia.org/wiki/Line_search): In optimization, the line search strategy is one of two basic iterative approaches to find a local minimum $$\mathbf x^*$$ of an objective function $$f$$. The other approach is trust region.

  The line search approach first finds a descent direction along which the objective function $$f$$ will be reduced and then computes a step size that determines how far $$\mathbf x$$ should move along that direction. The descent direction can be computed by various methods, such as gradient descent, Newton's method and quasi-Newton method. The step size can be determined either exactly or inexactly. 

则我们要求的是 (目前我们先考虑无约束极值)
$$
\min_{\mathbf x\in V} f(\mathbf x).
$$

---

## 梯度下降法

在此方法中，我们关心的是让**目标函数值** $$f(\bf x)$$ 最快速"下降"，也就是说在 $$V$$ 中的一个点 $$\mathbf x_k$$ 移动到另一个点 $$\mathbf x_{k+1} = (\mathbf x_k + \alpha \mathbf d)$$ 之后 ($$\alpha,\ \mathbf d$$ 的含义在下面)，**目标函数值**的改变情况。

首先我们通过 $$f(\mathbf x_{k+1})$$ 在 $$\mathbf x_k$$ 处 Taylor 展开
$$
f(\mathbf x_k + \alpha \mathbf d) = f(\mathbf x_k) + \alpha \mathbf g_k^T \mathbf d + o(\alpha^2), \tag{1.1}
$$
where

- $$\mathbf x_k$$: stands for the $$k$$-th variable (a vector);
- $$\mathbf d$$: stands for the (a vector), i.e., $$|\mathbf d| = 1$$;
- $$\alpha$$: stands for the step (a scalar);
- $$\mathbf g_k = \nabla f(\mathbf x_k)$$: the gradient of $$f$$ at $$\mathbf x_k$$;
- $$o(\alpha^2)$$: the higher order infinitesimal of $$\alpha$$.

We omit the $$o(\alpha^2)$$, and if we want $(1.1)$ to get the minimum value, we just let $$\alpha \mathbf g_k \cdot \mathbf d$$ be minimum. Note that $$\mathbf g_k$$ and $$\mathbf d$$ are vectors, so $$\mathbf g_k^T \mathbf d = |\mathbf g_k||\mathbf d|\cos\theta =|\mathbf g_k|\cos\theta $$ ($$\theta$$ is the angle between $$\mathbf g_k$$ and $$\mathbf d$$). So when $$\theta = \pi$$, i.e., $$\mathbf d = \frac{\mathbf g_k}{|\mathbf g_k|}$$, $$\mathbf g_k^T \mathbf d$$ get the minimum value: $$-|\mathbf g_k|$$. Thus, $(1.1)$ get the minimum value.

**梯度下降法的基本性质**:

- 梯度下降法的收敛性：对一般的目标函数是整体收敛的 (所谓整体收敛，是指不会非要在某些点附近的范围内，才会有好的收敛性)。
- 梯度下降法的收敛速度：至少是线性收敛的。
- 判断最终迭代结束的条件可选为：$$|\nabla f(\mathbf x_{k+1})|< tol$$.



---

## 牛顿法

[梯度下降法](#梯度下降法)只用到了梯度信息，即目标函数的一阶导数信息，并且在推导[梯度下降法](#梯度下降法)方法时，是通过 $$f(\mathbf x_{k+1})$$ 在 $$\mathbf x_k$$ 处 Taylor 展开得到的。而[牛顿法](#牛顿法)首先会用到目标函数的二阶导数信息，并且在推导的过程中是首先通过 $$f(\mathbf x)$$ 在 $$\mathbf x_k$$ 处 Taylor 展开，然后再对  $$f(\mathbf x)$$ 展开后的两端求极小值问题，我们具体来看

首先通过 $$f(\mathbf x)$$ 在 $$\mathbf x_k$$ 处 Taylor 展开
$$
f_k(\mathbf x) = f(\mathbf x_k) + \mathbf g_k^T(\mathbf x-\mathbf x_k) + \frac{1}{2}(\mathbf x-\mathbf x_k)^T\mathbf G_k (\mathbf x-\mathbf x_k) + o(\mathbf x^3), \tag{1.2}
$$
where 

- $$f_k(\mathbf x)$$: stands for the Taylor expansion of $$f$$ at $$\mathbf x_k$$;
- $$\mathbf g_k=\nabla f(\mathbf x_k)$$: the gradient of $$f$$ at $$\mathbf x_k$$;
- $$\mathbf G_k = \nabla^2f(\mathbf x_k)$$: the Hessian matirx of $$f$$ at $$\mathbf x_k$$ (In mathematics, the Hessian matrix or Hessian is a square matrix of second-order partial derivatives of a scalar-valued function, or scalar field). We assume that, here, $$\mathbf G_k$$ is continuous.

由于极小值点必然是驻点，而驻点是一阶导数为 $$0$$ 的点，所以，对 $$f$$ 这个函数来说，要取到极小值，我们应该分析其一阶导数。对 $$\mathbf x$$ 求一阶导数，令其等于 $$0$$ (并舍弃高阶项)：
$$
\nabla f_k(\mathbf x) = \mathbf g_k + \mathbf G_k(\mathbf x-\mathbf x_k) = 0,
$$
当 $$\mathbf G_k$$ 的逆矩阵存在，也即 $$\mathbf G_k$$ 为非奇异矩阵的时候，将上式两边都左乘 $$\mathbf G_k$$ 的逆矩阵 $$\mathbf G_k^{-1}$$，得：
$$
\begin{align}
&\mathbf G_k^{-1}\mathbf g_k + (\mathbf x - \mathbf x_k) = 0, \\
\Rightarrow \ & \mathbf x = \mathbf x_k - \mathbf G_k^{-1}\mathbf g_k =: \mathbf x_k + \mathbf d,
\end{align}
$$


到了这一步，已经很明显了。这个式子表达了下一点的计算方法：$$\mathbf x_k$$ 在方向 $$\mathbf d$$ 上按步长 $$\alpha = 1$$ 移动到点 $$\mathbf x_{k+1}$$。所以我们知道方向 $$\mathbf d$$ 怎么求了：
$$
\mathbf d = - \mathbf G_k^{-1} \mathbf g_k \Leftrightarrow \mathbf G_k \mathbf d = -\mathbf g_k.
$$
这里我们用 $$\mathbf G_k \mathbf d = -\mathbf g_k$$ 表示，是因为在实际应用中，$$\mathbf d$$ 并不是通过 $$\mathbf G_k^{-1}$$ 与 $$-\mathbf g_k$$ 相乘来计算出的 (因为我们并不知道逆矩阵 $$\mathbf G_k^{-1}$$ 是什么)，而是通过解方程组 $$\mathbf G_k \mathbf d = -\mathbf g_k$$ 求出的。这个解方程组的过程，其实也就可能是一个求逆矩阵的过程。而此方程组可能无解的 ($$\mathbf G_k$$ 是奇异的)，在这种情况下，就需要用到其他的修正技术，来获取搜索方向了.

**牛顿法的基本步骤**：

- 每一步迭代过程中，通过解线性方程组得到搜索方向 $$\mathbf d$$;
- 将自变量 ($$\mathbf x_k$$) 移动到下一个点 ($$\mathbf x_{k+1}$$); 
- 计算 $$f(\mathbf x_{k+1})$$ 是否符合收敛条件，不符合的话就一直按这个策略  (解方程组 $$\rightarrow$$ 得到搜索方向 $$\rightarrow$$ 移动点 $$\rightarrow$$ 检验收敛条件) 继续下去。

**牛顿法的基本性质**：

- 牛顿法的收敛性：对一般问题都不是整体收敛的 (只有当初始点充分接近极小点时，才有很好的收敛性);
- 牛顿法的收敛速度：二阶收敛。因此，它比[梯度下降法](#梯度下降法)要快。



---

## 共轭方向法

[共轭方向法](#共轭方向法) 是介于最速下降法和牛顿法之间的一种存在——它的收敛速度 (二阶收敛) 比 [梯度下降法](梯度下降法) (线性收敛) 快，同时它的计算量又比牛顿法要小，因此它的存在是有意义的。

需要注意，共轭方向法可以不使用目标函数的一阶导数信息 (当然也可以使用)。所以，如果目标函数的一阶导数不容易求的话，[共轭方向法](#共轭方向法) 可能就可以派上用场了。

共轭方向法的显著特征就是：两次搜索方向之间是有关联的，这种关联就是 "共轭"。

1. **向量共轭**

   先解释一下向量共轭的含义，你就明白共轭方向法的两次搜索方向之间的 "共轭" 是怎么回事了。

   设 $$\mathbf G$$ 为对称正定矩阵，若 $$\mathbf d_m^T \mathbf G \mathbf d_n = 0$$，$$m\not= n$$ 则称 $$\mathbf d_m$$ 和 $$\mathbf d_m$$ 为 "$$\mathbf G$$ 共轭"，共轭方向是 "互不相关" 的方向。

2. **特性**

   > 当目标函数是二次函数 $$f(\mathbf x) = \frac{1}{2} \mathbf x^T \mathbf G \mathbf x + \mathbf b^T \mathbf x + c $$ 时，共轭方向法最多经过 $$n$$ 步（$$n$$ 为向量维数）迭代，就可以到达极小值点——这种特性叫作 **二次收敛性**（Quadratic Convergence）。
   > 假设沿着一系列的共轭方向做迭代（寻找极小值点），这些共轭方向组成的集合叫作 **共轭方向集**，则沿共轭方向集的每个方向顺序做 line search 的时候，在每个方向上都不需要做重复搜索——在任何一个方向上的移动，都不会影响到在另一个方向上已经找到的极小值。

   上面这段描述是什么意思呢？我们先不讨论这些共轭方向是怎么计算出来的，拿一个在水平面上走路的例子来做比喻：你在水平方向 A 上走了 10 米，然后再沿着与水平方向垂直的另一个方向 B 上又走了 10 米，那么，你在方向 A 上走动的时候，在方向 B 上的坐标是不变的；你在方向 B 上走动的时候，在方向 A 上的坐标也是不变的。因此，把方向 A 和方向 B 看作两个共轭方向，那么，你在这两个共轭方向中的任何一个方向上移动，都不会影响到另一个方向上已经走到的坐标（把它想像成在这个方向上的极小值）。

   但是世界哪有那么美好？目标函数不是二次函数的时候多得去了！这个时候，共轭方向法不就不能用了吗？理论与实践证明，将二次收敛算法用于非二次的目标函数，也有很好的效果。但是，这个时候，就不能保证 $$n$$ 步迭代到达极小值点了。大家需要记住的是，**很多函数都可以用二次函数很好地近似**，这种近似在工程上是很重要。

   有人一定会问，哪些函数可以用二次函数很好地近似呢？请原谅我没在书中看到这个总结，你只能自己去挖掘了。

3. **理论基础**

   共轭方向法有一个重要的理论基础，它是一个神奇的定理，有了它，可以推导出很多结论 (共轭梯度法的理论推导就依赖于此)。

   这里只把结论写上来，证明较长，不是本文关注的所以就不写了：

   > 在精确 line search 的情况下，当前迭代点的梯度 $$\mathbf g$$ 与前面所有的搜索方向 $$\mathbf d$$ 正交：
   > $$
   > \mathbf g_{k+1}^T \mathbf d_k = 0 \quad k = 0, 1, ..., k
   > $$

   在这个 $$\mathbf g_{k+1}^T \mathbf d_k = 0$$ 式子中，当 $$\mathbf g$$ 的下标是 $$k+1$$ 时，$$\mathbf d$$ 的下标可以是 $$0,1,...,k$$，例如，$$\mathbf g_{3}^T \mathbf d_2 = 0$$，$$\mathbf g_{3}^T \mathbf d_1 = 0$$，$$\mathbf g_{3}^T \mathbf d_0 = 0$$，这表明，当前迭代点的梯度 $$\mathbf g_3$$ 与前面所有的搜索方向 ($$\mathbf d_0, \mathbf d_1, \mathbf d_2$$) 正交。

   

   一般的书中可能会用一种更加抽象的语言来描述，例如

   > 共轭方向法在迭代过程中的每一个迭代点 $$\mathbf x_{i+1}$$ 都是目标函数 $$f(\mathbf x)$$ 在 $$\mathbf x_0$$ 和方向 $$\mathbf d_0, \mathbf d_1,...,\mathbf d_i$$ 所张成的线性流形
   > $$
   > \Big\{ \mathbf x | \mathbf x = \mathbf x_0 + \sum_{j=0}^{i} \alpha_j \mathbf d_j, \ \forall \alpha_j \Big\}
   > $$
   > 中的极小点。

   其实这个晦涩的描述，是 **line search 基础定理 (梯度与方向的点积为零)** 的另一种表述。我们用一个特例来说明：
   迭代点 $$\mathbf x_2$$ (此时 $$i=1$$) 是目标函数 $$f(\mathbf x)$$ 和方向 $$\mathbf d_0, \mathbf d_1$$ 所张成的线性流形 $$\{\mathbf x| \mathbf x = \mathbf x_0 + \alpha_0\mathbf d_0 + \alpha_1\mathbf d_1 \}$$ 的极小值点。而 $$\mathbf x_0 + \alpha_0\mathbf d_0 + \alpha_1\mathbf d_1 = \mathbf x_1 + \alpha_1\mathbf d_1 = \mathbf x_2$$，所以这就说明了 $$\mathbf x_1$$ 是在 $$\mathbf d_0$$ 方向上 line search 得到的极小值点， $$\mathbf x_2$$ 是在 $$\mathbf d_1$$ 方向上 line search 得到的极小值点。所以由基础定理可知，当前迭代点的梯度与前面所有方向的点积为零。

   

4. 





