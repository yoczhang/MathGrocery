# ch2

---

We consider the $$2D$$ case:
$$
\begin{bmatrix}
&x_1 \ \ &x_2 \ \ &x_3 \\
&y_1 \ \ &y_2 \ \ &y_3 \\
&1 \ \ &1 \ \ &1 
\end{bmatrix} 
\begin{bmatrix}
\lambda_1 \\
\lambda_2 \\
\lambda_3
\end{bmatrix} = 
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix},
$$
this gives us $$\lambda_1+\lambda_2+\lambda_3 = 1$$, and 
$$
\begin{align}
&x = \lambda_1 x_1 + \lambda_2 x_2 + (1-\lambda_1-\lambda_2) x_3, \\
&y = \lambda_1 y_1 + \lambda_2 y_2 + (1-\lambda_1-\lambda_2) y_3.
\end{align}
$$
Rearranging, this is 
$$
\begin{align}
& \lambda_1(x_1-x_3)+\lambda_2(x_2-x_3)+ x_3 - x = 0, \\
& \lambda_1(y_1-y_3)+\lambda_2(y_2-y_3)+ y_3 - y = 0.
\end{align}
$$
Let $$\mathbf r = [x, y]^T,\ \mathbf r_i = [x_i, y_i]^T (i = 1,2,3)$$, and $$\pmb \lambda = [\lambda_1, \lambda_2]^T$$. The above linear transformation may be written more succinctly as
$$
\mathbf T \pmb \lambda = \mathbf r - \mathbf r_3, \text{ where }\ \mathbf T = [\mathbf r_1 - \mathbf r_3, \ \mathbf r_2 - \mathbf r_3],
$$
thus $$\pmb \lambda = \mathbf T^{-1} (\mathbf r - \mathbf r_3)$$, and we have 
$$
\begin{align}
\lambda_1 &= \frac{\rm det([\mathbf r - \mathbf r_3, \ \mathbf r_2 - \mathbf r_3])}{\rm det(\mathbf T)} = \frac{(y_2 - y_3)(x-x_3) - (x_2-x_3)(y-y_3)}{(y_2 - y_3)(x_1-x_3) - (x_2-x_3)(y_1-y_3)}, \\
\lambda_2 &= \frac{\rm det([\mathbf r_1 - \mathbf r_3, \ \mathbf r - \mathbf r_3])}{\rm det(\mathbf T)} = \frac{(x_1-x_3)(y-y_3) - (y_1 - y_3)(x-x_3)}{(y_2 - y_3)(x_1-x_3) - (x_2-x_3)(y_1-y_3)}, \\
\lambda_3 &= 1 - \lambda_1 - \lambda_2.
\end{align}
$$


---

1. Todo:

   

2. In this exercise, we give explicit formula of the stiffness matrix. Let $$\tau$$ be a triangle with vertices $$\pmb x_1, \pmb x_2, \pmb x_3$$ and let $$\lambda_1, \lambda_2, \lambda_3$$ be corresponding barycentric coordinates. See the figure <img src="./figures/trangle1.png" alt="trangle1" style="zoom:10%;" />

   

   Google: barycentric coordinate in finite element method

   See: [wiki- barycentric system](https://en.wikipedia.org/wiki/Barycentric_coordinate_system)

   http://www.iue.tuwien.ac.at/phd/nentchev/node26.html

   - Let $$\pmb n_i$$ be the outward normal vector of edge $$e_i$$ and $$d_i$$ be the distance from $$\pmb x_i$$ to $$e_i$$. Prove that
     $$
     \nabla \lambda_i = -\frac{1}{d_i}\pmb n_i.
     $$
     Proof: 

   

3. 
4. 









