For minimizing empirical risk.
The good news about the [[Empirical Risk]] Function is that it's actually differentiable everywhere.
So what we are doing is what we have done in the past.
We're going to randomly select one example and then look at the direction of the gradient. Since we're trying to minimize, we will just slightly nudge the parameters in the right direction. (And the right direction will be determined by the gradient). The right direction is against the gradient (opposite to the positive value).

So what we are doing is to random pick up one example and try to compute the gradient.

### 1. The Gradient Derivation

The top part of the board calculates the gradient of the squared error loss for a single training example, denoted by the index $t$.

The loss for this single example is $\frac{1}{2}(y^{(t)} - \theta x^{(t)})^2$. Taking the gradient (derivative) with respect to $\theta$ yields:

$$\nabla_\theta \frac{(y^{(t)} - \theta x^{(t)})^2}{2} = (y^{(t)} - \theta x^{(t)}) \nabla_\theta (y^{(t)} - \theta x^{(t)})$$

Applying the chain rule, the derivative of the inner term $(y^{(t)} - \theta x^{(t)})$ with respect to $\theta$ is $-x^{(t)}$, which results in the final gradient:

$$= -(y^{(t)} - \theta x^{(t)}) \cdot x^{(t)}$$

### 2. The Algorithm (Alg)

- **Initialize:** Start with an empty weight vector.
$$\theta = 0$$
- **Iterate:** Randomly pick a single training example from the dataset.
$$\text{randomly pick } t = \{1 \dots n\}$$
- **Update:** Adjust the weight $\theta$. Note that the standard update rule is $\theta = \theta - \eta \cdot \text{gradient}$. Because the gradient we calculated above has a negative sign in front, subtracting it turns the operation into addition:
$$\theta = \theta + \eta (y^{(t)} - \theta x^{(t)}) \cdot x^{(t)}$$
(Here, $\eta$ represents the learning rate, which we previously called $\alpha$.)_

This algorithm in some ways is self-correcting.

Coming Next: [[Linear Regression Closed Form Solution]] 