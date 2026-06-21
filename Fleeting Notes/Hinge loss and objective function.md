So now, our regularization goal here is to maximize the distance that the margin boundaries are from the decision boundaries.
$$\max(\frac{1}{\|\theta\|})$$
This will be our regularization type.

Now it's time to define the objective function for finding large margin decision boundaries.
As we've already seen, it has two components:
- A loss (The error function)
- A regularization (set of methods for reducing overfitting)

We have talked about both.

## Hinge Loss
 We already know that if $y^{(i)}(\theta \cdot x^{(i)} + \theta_0)$ is positive, that means the prediction agrees with the label. So we can call this argument **agreement**. lets denoted by $z$.
$$z=\text{agreement}= y^{(i)}(\theta \cdot x^{(i)} + \theta_0)$$
We also know that if the point lies exactly on the margin boundary, the agreement value is exactly 1. If it lies beyond the margin boundary is greater than 1.

So we can define the laws of how much this preference to keep the points outside the margin boundaries is violated by saying:
$$\text{Loss}_h(z) = \begin{cases} 0 & \text{if } z \ge 1 \\ 1 - z & \text{if } z < 1 \end{cases}$$
Meaning that if $z$ is beyond the margin boundary the loss is 0, and if it is before the margin boundary the loss is $1-z$.
So it starts penalizing for any example that penetrates into the margin boundaries.

## Regularization: towards max margin
Now, the regularization, we've already seen that we want to maximize the distance between the decision boundary and the margin boundaries.
$$max(\frac{1}{\|\theta\|})$$
Which is the sames as trying to minimize the norm of $\theta$
$$min(\|\theta\|)$$
The same of minimizing the squared norm of theta and just for prettiness, multiplying that by $\frac{1}{2}$, which will become clear later why we do that.
$$min(\frac{1}{2}\|\theta\|^2)$$
So we can regularize the solutions by trying to penalize large values of the norm of parameter vector $\theta^2$.
## The objective/cost/loss function

So now, we have an objective function that guides our selection of parameter vector $\theta$ and $\theta_0$.
$$J(\theta, \theta_0) = \frac{1}{n} \sum_{i=1}^n \text{Loss}_h(y^{(i)}(\theta \cdot x^{(i)} + \theta_0)) + \frac{\lambda}{2}\|\theta\|^2$$
This objective function has two parts.
- $\frac{1}{n} \sum_{i=1}^n \text{Loss}_h(y^{(i)}(\theta \cdot x^{(i)} + \theta_0))$: An average loss, where each loss term measures how much that example violates their margin boundary. Defined according to the hinge loss we just discussed.
- $\frac{\lambda}{2}\|\theta\|^2$: The regularization term that tries to push the margin boundaries further and further apart.

Now, our objective function is a balance between the two. We set that balance  by defining a new parameter that simply weights how these two terms should affect our solution: $\lambda$. This parameter is always greater than 0.

- For large values of $\lambda$: We favor large margin solutions but potentially at a cost of incurring some further loss as the margin boundaries push past the examples.
- For small values of $\lambda$: We favor, really, correctly putting the examples outside the margin boundarues, potentially at a cost of keeping the margin boundary as closer to the decision boundary.

**Optimal value of $\theta$ and $\theta_0$ is obtained by minimizing this objective function.** 

And now we've turned the learning problem into an optimization problem.

Let's look how the objective function guides the selection of $\theta$ and $\theta_0$.
![[file-20260620200157463.jpg]]
For a point that lies exactly on the margin boundary and is correctly labeled, the hinge loss will be $0$.
![[file-20260620200610501.jpg]]
For points that lie before the margin boundary the loss will be $1-z$, so if we have a point exactly in the decision boundary (labeled as positive or negative), the loss will be $1-0 = 1$. 
![[file-20260620200834385.jpg]]
A point that goes beyond the decision boundary, to the wrong side under the decision boundary. i.e. a negatively labeled point exactly on the positive margin boundary, the loss will be $1-(-1)=2$. If we go even further it will be greater than 2.
![[file-20260620201212681.jpg]]


Later we'll talk about the solution in terms of how it changes by changing the regularization parameter $\lambda$.
## References

**Smarter Optimization**
![[smarter_optimization.pdf]]

## Personal Intuition
![[file-20260620201508617.jpg]]
In general, learning problems are cast as optimization problems in terms of objective functions that guide the setting of the parameters that we are estimating.

The objective functions are decomposed in two parts: average loss over the training exam and the regularization. That guides the type of solution we are after. In this particular case, how large of a margin we can achieve for the training parts.

So, to this end we learned to define margin boundaries, how this relate to hinge loss  (penalizing values since they fall inside the boundaries) and how the regularization pushes the margin boundary as it apart (create larger boundaries, penalizing more values).
All of this together define, then, the objective function that guides how $\theta$ and $\theta_0$ are resolved as the minimizing values.


**Coming Next**: [[Review and The lambda Parameter]]