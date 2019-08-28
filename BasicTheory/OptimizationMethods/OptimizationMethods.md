# Optimization methods

最优化 (Optimization) 是应用数学的一个分支，它是研究在给定约束之下如何寻求某些因素(的量)，以使某一(或某些)指标达到最优的一些学科的总称。

我们设定

- The space: $$V = \{\bf x| \bf{x} \in \mathbb R^n \} $$ or $$V = H_0^1(\Omega)$$;
- The objective function: $$f(\mathbf x): V \rightarrow \mathbb R$$;

---

## 梯度下降

在此方法中，我们关心的是让**目标函数值** $$f(\bf x)$$ 最快速"下降"，也就是说在 $$V$$ 中的一个点 $$\mathbf x_k$$ 移动到另一个点 $$\mathbf x_{k+1} = (\mathbf x_k + \alpha \mathbf d)$$ 之后 ($$\alpha,\ \mathbf d$$ 的含义在下面)，**目标函数值**的改变情况。

首先我们通过 $$f(\mathbf x_{k+1})$$ 在 $$\mathbf x_k$$ 处 Taylor 展开
$$
f(\mathbf x_k + \alpha \mathbf d) = f(\mathbf x_k) + \alpha \mathbf g_k \cdot \mathbf d + o(\alpha^2), \tag{1.1}
$$
where

- $$\mathbf x_k$$: stands for the $$k$$-th variable (a vector);
- $$\mathbf d$$: stands for the (a vector), i.e., $$|\mathbf d| = 1$$;
- $$\alpha$$: stands for the step (a scalar);
- $$\mathbf g_k = \nabla f(\mathbf x_k)$$: the gradient of $$f$$ at $$\mathbf x_k$$;
- $$o(\alpha^2)$$: the higher order infinitesimal of $$\alpha$$.

We omit the $$o(\alpha^2)$$, and if we want $(1.1)$ to get the minimum value, we just let $$\alpha \mathbf g_k \cdot \mathbf d$$ be minimum. Note that $$\mathbf g_k$$ and $$\mathbf d$$ are vectors, so $$\mathbf g_k \cdot \mathbf d = |\mathbf g_k||\mathbf d|\cos\theta =|\mathbf g_k|\cos\theta $$ ($$\theta$$ is the angle between $$\mathbf g_k$$ and $$\mathbf d$$). So when $$\theta = \pi$$, i.e., $$\mathbf d = \frac{\mathbf g_k}{|\mathbf g_k|}$$, $$\mathbf g_k \cdot \mathbf d$$ get the minimum value: $$-|\mathbf g_k|$$. Thus, $(1.1)$ get the minimum value.

---

## 牛顿法

[梯度下降](#梯度下降)只用到了梯度信息，即目标函数的一阶导数信息，并且在推导[梯度下降](#梯度下降)方法时，是通过 $$f(\mathbf x_{k+1})$$ 在 $$\mathbf x_k$$ 处 Taylor 展开得到的。而[牛顿法](#牛顿法)首先会用到目标函数的二阶导数信息，并且在推导的过程中是首先通过 $$f(\mathbf x)$$ 在 $$\mathbf x_k$$ 处 Taylor 展开，然后再对  $$f(\mathbf x)$$ 展开后的两端求极小值问题，我们具体来看

首先通过 $$f(\mathbf x)$$ 在 $$\mathbf x_k$$ 处 Taylor 展开
$$
f(\mathbf x) = f(\mathbf x_k) + \mathbf g_k^T(\mathbf x-\mathbf x_k) + \frac{1}{2}(\mathbf x-\mathbf x_k)^T\mathbf G_k (\mathbf x-\mathbf x_k), \tag{1.2}
$$














