# FEALPy Basic

#### 1. 

p = int(sys.argv[2])  #  p 表示空间的分片多项式阶数，如 p=2 即采用 P2 元.

n = int(sys.argv[3]) #  n 表示网格初始化时加密的次数.



#### 2. 

in `class TriangleMesh(Mesh2d)`:

...
`N = node.shape[0]`, `node.shape` returns the numbers of rows and cols:	

```pyth
In [23]: import numpy as np
In [24]: node = np.array([
...:             (0, 0),
...:             (1, 0),
...:             (1, 1),
...:             (0, 1)], dtype=np.float)
...:             
In [25]: node.shape
Out[25]: (4, 2)
```





