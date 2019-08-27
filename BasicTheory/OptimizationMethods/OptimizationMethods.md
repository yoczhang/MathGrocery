# Optimization methods

最优化 (Optimization) 是应用数学的一个分支，它是研究在给定约束之下如何寻求某些因素(的量)，以使某一(或某些)指标达到最优的一些学科的总称。

我们设定

- The space: $$V = \{\bf x| \bf{x} \in \mathbb R^n \} $$ or $$V = H_0^1(\Omega)$$;
- The objective function: $$f(\mathbf x): V \rightarrow \mathbb R$$;

---

## 最速下降法

在此方法中，我们关心的是让**目标函数值** $$f(\bf x)$$ 最快速"下降"，也就是说在 $$V$$ 中的一个点 $$\mathbf x_k$$ 移动到另一个点 $$\mathbf x_{k+1} = (\mathbf x_k + \alpha \mathbf d)$$ 之后 ($$\alpha,\ \mathbf d$$ 的含义在下面)，**目标函数值**的改变情况。

首先我们通过 Taylor 展开
$$
f(\mathbf x_k + \alpha \mathbf d) = f(\mathbf x_k) + \alpha \mathbf g_k^T \mathbf d + o(\alpha^2), \tag{1.1}
$$
where

- $$\mathbf x_k$$: stands for the $$k$$-th variable (a vector);
- $$\mathbf d$$: stands for the  (a vector), i.e., $$|\mathbf d| = 1$$;
- $$\alpha$$: stands for the step (a scalar);
- $$\mathbf g_k = \nabla f(\mathbf x_k)$$: the gradient of $$f$$ at $$\mathbf x_k$$;
- 















