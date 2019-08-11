# Learning Python

---



---

## What is a[:2, :3] meaning?

```py
>>> import numpy as np
>>> a = np.array([[11, 12, 13, 14, 15],
...               [16, 17, 18, 19, 20],
...               [21, 22, 23, 24, 25],
...               [26, 27, 28 ,29, 30],
...               [31, 32, 33, 34, 35]])
>>> a[:2, :3]
array([[11, 12, 13],
       [16, 17, 18]])
```



---

## What is :: (double colon) in Python?

See: [What is :: (double colon) in Python?](https://stackoverflow.com/questions/3453085/what-is-double-colon-in-python)

`a[::y]` it prints every $$y^{th}$$ element from the list/array, 

i.e.,

```pyt
>>> a = [1,2,3,4,5,6,7,8,9]
>>> a[::3]
[1, 4, 7]
```

The additional syntax of `a[x::y]` means get every $$y^{th}$$ element starting at position `x`.

i.e.,

```pyth
>>> a[2::3]
[3, 6, 9]
```

---

## What does the Python Ellipsis (...) object do?

See:  

* [What does “three dots” in Python mean when indexing what looks like a number?](https://stackoverflow.com/questions/42190783/what-does-three-dots-in-python-mean-when-indexing-what-looks-like-a-number/42191121)

* [What does the Python Ellipsis object do?](https://stackoverflow.com/questions/772124/what-does-the-python-ellipsis-object-do)

但是简单来说就是在 NumPy 的数组切片时, 它是省略所有的冒号 (:) 来用 `...` 代替, 即 `a[..., 0]` 代替 `a[:,:, 0]`. 例如下面的例子：

```py
>>> a = np.array([[[1],[2],[3]], [[4],[5],[6]]])
>>> a.shape
(2, 3, 1)
>>> a
array([[[1],
        [2],
        [3]],

       [[4],
        [5],
        [6]]])
>>> a[..., 0]
array([[1, 2, 3],
       [4, 5, 6]])
>>> a[:, 0]
array([[1],
       [4]])
>>> a[:,:, 0]
array([[1, 2, 3],
       [4, 5, 6]])

```

---

## NumPy 数组中 None 参数

可以先看一个例子

```pyt
>>> a = np.array([[11, 12, 13, 14, 15],
...                [16, 17, 18, 19, 20],
...                [21, 22, 23, 24, 25],
...                [26, 27, 28 ,29, 30],
...                [31, 32, 33, 34, 35]])
>>> a.shape
(5, 5)
>>> b = a[:, None]
>>> b
array([[[11, 12, 13, 14, 15]],

       [[16, 17, 18, 19, 20]],

       [[21, 22, 23, 24, 25]],

       [[26, 27, 28, 29, 30]],

       [[31, 32, 33, 34, 35]]])
>>> b.shape
(5, 1, 5)
```

将列的参数改成 `None`，输出的 `shape` 都变了. 这里大家要知道，`None` 代表新增加一个维度，它有一个别称叫 `newaxis`，大家可以输出一下 `numpy.newaxis` 查看一下

```pyt
>>> print(np.newaxis)
None
```

那么为什么是 `5x1x5`，而不是 `5x5x1` 呢，那是因为在第二维上用了 `None`.

我们下面尝试在第三维上用 `None`:

```pyt
>>> c = a[..., None]
>>> c.shape
(5, 5, 1)
```

这个例子也说明了 `...` 的用法.