Briefly
For simplicity, let's show it with just one parameter ($\theta$) and see how the loss function value change in function of $\theta$. So I have some starting parameter value $\theta$ and I'm foinf to try to change the parameter value to get closer to the minimum of the function. 
![[file-20260621174154879.jpg]]
At any point, here at the starting point, I can compute the slope of that function, that slope is given by the derivative of the objective function with respect to $\theta$.

If that slope is positive, if I move in the same direction, im going to increase the function, if i move in the opposite direction of the sign of the slope I will decrease the function.

So my gradient descent update rule here is getting a new parameter value ($\theta$) in terms of the old one and moving in the negative direction of the derivative of the gradient of the function ($-\frac{\partial J(\theta)}{\partial \theta}$). We're also including a step size or learning rate parameter ($\eta$), the role of this is not to take too large of a step so that i would actually find myself somewhere where the value of the objective function might bethe same or even larger.

$$\theta \leftarrow \theta - \eta \frac{\partial J(\theta)}{\partial \theta}$$
So, if the function is nice so there is a little convex curvature at every point, meaning that is a strongly convex function, then there would be a constant learning rate small enough that i am guaranteed quickly to get to the minimum of that function.

In 3D is the same, with respect of each of the parameters, we're moving in the negative direction of the derivative of the function with respect of that particular parameter.
![[file-20260621175516593.jpg]]

### 1. Mathematical Form (Component-wise Update)
To update the parameters one by one, you take the partial derivative of the cost function with respect to each specific parameter $\theta_j$:

$$\theta_j \leftarrow \theta_j - \eta \frac{\partial J(\theta)}{\partial \theta_j}$$
### 2. Vector Form (Simultaneous Update)
Instead of updating each parameter individually, you can update all of them at once using the gradient vector $\nabla J(\theta)$. This is the most common and efficient way to compute gradient descent in machine learning:
$$\theta \leftarrow \theta - \eta \nabla J(\theta)$$
### 3. The Gradient Vector Definition
The image also defines what the gradient $\nabla J(\theta)$ actually is. It is simply a column vector containing all the partial derivatives for each parameter from $\theta_1$ up to $\theta_d$:
$$\nabla J(\theta) = \begin{bmatrix} \frac{\partial J(\theta)}{\partial \theta_1} \\ \vdots \\ \frac{\partial J(\theta)}{\partial \theta_d} \end{bmatrix}$$
### Variable Breakdown:
- **$\theta_j$**: A single specific parameter (weight) in your model.
- **$\theta$**: The entire vector of parameters.
- **$\eta$**: The learning rate (step size).
- **$\nabla J(\theta)$**: The gradient of the cost function, which points in the direction of the steepest ascent. We subtract it to move "down" the 3D slope shown in the graph!
