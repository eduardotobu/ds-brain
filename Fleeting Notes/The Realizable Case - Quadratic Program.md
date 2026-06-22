We can also solve the problem by actually solving the optimization problem.
## Support Vector Machine

* Support Vector Machine finds the maximum margin linear separator by solving the quadratic program that corresponds to $J(\theta, \theta_0)$
* In the realizable case, if we disallow any margin violations, the quadratic program we have to solve is

$$\begin{aligned}
&\text{Find } \theta, \theta_0 \text{ that} \\
&\quad \text{minimize } \frac{1}{2}\|\theta\|^2 \text{ subject to} \\
&\quad\quad y^{(i)}(\theta \cdot x^{(i)} + \theta_0) \geq 1, \quad i = 1, \dots, n
\end{aligned}$$
So, the problem is linearly separable and we don't allow any errors, so if we want all the trainign samples to have 0 hinge loss, they would have to have the agreement greater or equal to 1.

Subject to those constraints, we're trying to minimize the norm of the parameter vector, and maximize the margin subject to having the examples on the correct sides of the margin boundaries. What we get here is now a quadratic program (there are packages for trying to solve this). 

You can also generalize this to allowing these constraints to be violated and incorporating the actual loss value. It would be a slightly modified version of the quadratic programming problem.

The solution for this strict case is that you always have some subset of points that are exactly on the margin boundary, because we increase the margin until we cannot do it anymore, and we start hitting the training samples, but we cannot go further because in this simple case, we strictly try to enforce the margin constraints.
![[file-20260621191024992.jpg]]

