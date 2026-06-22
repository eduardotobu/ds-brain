So, first let's erase the $\theta_0$ for ease the calculations and we can include the regularization factor inside the Sum, since it doesn't depend in the particular example, is the same if it's inside or outside the sum.
![[file-20260621182820683.jpg]]
And also, understand that the loss function is actually an average/expectation over the loss of individual terms.

So let's move on:

Stochastic simply means 'random' so the main difference between stochastic and standard gradient descent is : 

- Standard gradient descent uses the entire dataset to update a given parameter.
  $$\theta = \theta - \alpha \nabla_\theta J(\theta)$$
- Stochastic Gradient Descents selects examples at random and uses one at a time to update the parameter.
  $$\theta = \theta - \alpha \nabla_\theta J(\theta; x_i, y_i)$$
## The Robbins-Monro Conditions for Convergence

Because SGD relies on taking steps based on single, randomly chosen data points, it is inherently noisy and bounces around. The bottom three handwritten equations define the strict mathematical conditions the learning rate ($\eta_t$) must follow to guarantee that this bouncy path eventually settles down at the true minimum (convergence):

- **$\eta_t \to 0$**: As time goes on, the learning rate must shrink toward zero. The steps must get smaller so the model stops bouncing and settles.
- **$\sum_{t=1}^{\infty} \eta_t = \infty$**: The sum of all learning rates over time must be infinite. This ensures that no matter how far away the starting point is from the minimum, the algorithm can still travel the distance required to reach it.
- **$\sum_{t=1}^{\infty} \eta_t^2 < \infty$**: The sum of the _squares_ of the learning rates must be finite. This guarantees that the learning rate decreases fast enough to ensure the variance (the erratic bouncing) eventually drops to zero, forcing the algorithm to converge.

Explaining better the last two conditions:
- The first rule ($∑ηt​=∞$) says: **"Don't shrink the steps so fast that you get stuck."**
- The second rule ($∑ηt^2​<∞$) says: **"But do shrink them fast enough that you stop bouncing."**

**Does a number like this actually exist?** Yes! The classic example is using a learning rate of $η_t​=\frac{1}{t}$. At step 1, your rate is 1. At step 2, it is $\frac{1}{2}$​​. At step 3, it is $\frac{1}{3}$​.

- **Rule 1 check:** In calculus, the sum of $1+\frac{1}{2}+\frac{1}{3}+…$(the Harmonic Series) is well-known to equal **infinity**. So it satisfies the first rule.
- **Rule 2 check:** If we square those steps, the sum of $1+\frac{1}{4}​+\frac{1}{9}​+…$ is mathematically proven to equal a finite number (specifically, $\frac{\pi^2}{6}$)​. It satisfies the second rule.

## Stochastic Update Rule Formula
Instead of computing the gradient for the entire dataset at once (which is computationally expensive), SGD approximates the gradient by picking one training example at a time:

Select $i \in \{1, \dots, n\}$ at random

$$\theta \leftarrow \theta - \eta_t \nabla_\theta \left[ \text{Loss}_h(y^{(i)} \theta \cdot x^{(i)}) + \frac{\lambda}{2} \|\theta\|^2 \right]$$

### **Breakdown of the Variables**
- **$J(\theta)$**: The overall objective function being minimized.
- **$\theta$**: The parameters (or weights) of the model.
- **$n$**: The total number of examples in the training dataset.
- **$x^{(i)}$** and **$y^{(i)}$**: The feature vector and the true label for the $i$-th randomly selected training example.
- **$\text{Loss}_h$**: The loss function (e.g., Hinge loss for Support Vector Machines) calculating the error between the prediction and the actual label.
- **$\frac{\lambda}{2} \|\theta\|^2$**: The L2 Regularization term, where $\lambda$ is the regularization strength.
- **$\eta_t$**: The learning rate (or step size) at iteration $t$, which controls how much the parameters are adjusted.
- **$\nabla_\theta$**: The gradient (derivative) of the loss function with respect to the parameters $\theta$.
## Differences Between SGD and Perceptron
1. We are actually using a decreassing learning rate due to stochasticity.
2. We are actually performing the update, even if we are correctly classifying the example, because the regularization term will always yield an update, regardless of what the loss is. And the goal of that regularization term is to nudge the parameters a little bit backwards, so decrease the norm of the parameter. vector at every stamp, which corresponds to trying to maximize the margin.


### Why use SGD?
Here is a breakdown of why SGD (and its variants) powers almost all modern deep learning:

| Pros                                                                                                                                                                | Cons                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Speed:** It updates the model immediately, leading to much faster convergence on massive datasets.                                                                | **Erratic Path:** The updates are "noisy," meaning the loss will bounce around instead of decreasing smoothly.                                                              |
| **Memory Efficiency:** You only need to load a single data point into memory at a time, making it possible to train on datasets that exceed your RAM.               | **Doesn't Settle:** Because of the noise (randomness), SGD might keep bouncing around the very bottom of the valley rather than stopping at the exact mathematical minimum. |
| **Escapes Traps:** The erratic "bouncing" behavior actually helps the model escape "local minima" (shallow dips in the mountain) to find a better overall solution. | **Vectorization Loss**: Processing one item at a time loses the speed advantages of matrix multiplication optimized by modern GPUs.                                         |
**The Practical Middle Ground: Mini-Batch SGD** In reality, data scientists rarely use pure SGD (one example at a time) or pure GD (all examples at once). They use **Mini-batch SGD**. This takes a small chunk of data (e.g., 32, 64, or 256 examples) at a time. It strikes the perfect balance: it's fast, fits into GPU memory perfectly, and is smooth enough to converge well while retaining enough "noise" to escape local minima.

## Personal Intuition
Since the loss function is the expectation of each indidivual loss in the set, we select random examples and run gradient descent one at a time.