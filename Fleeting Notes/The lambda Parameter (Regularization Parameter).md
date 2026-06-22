
We're going to try to understand the solutions to the optimization problem and how to find those solutions.

Remember that the loss function for a large margin linear classifier (support vector machine) is composed by two parts: loss and regularization.
![[file-20260621154501914.jpg]]

So, the larger the value of $\lambda$ is, the more we try to push the margin boundaries apart. The smaller it is, the more emphasis we put on minimizing the average loss on the training example.

So **we can see the regularization part ($\frac{\lambda}{2}\|\theta\|^2$) as a balancing scale that forces the model to choose between being "perfect" on the training data and being "robust" for future data.** 


Let's see geometrically how this works.

#### Low $\lambda$ example
So here we have a set of training examples on a linear separable problem. We start with the regularization parameter being small: 0.1. So we're emphasizing correct classification of these examples. So the solution that i get is then that all the examples are on the correct side of the margin boundary.
![[file-20260621154957364.jpg]]

As we start increasing the value of $\lambda$, we put more emphasis on the regularization term, that is, trying to push the margin boundaries further and further apart.

In the previous case, they did not yet move because the additional emphasis on the regularization term is not enough to counterbalance losses that would incur if the margin boundaries past the actual examples.
#### High $\lambda$ example
So, if we increase $\lambda$ to 1000, we're pushing the margin boundaries further, getting a different solution to our optimization problem. AND also the margins are slightly rotated, guided more by where the bulk of the points between the margin boundary and the decision boundary.

> Important Note:**In support vector machines, any point that falls on or inside the margin boundary, becomes a support vector**.
![[file-20260621155827504.jpg]]

#### Not linealy separable with low $\lambda$ example
Some of the margin constraints are violated by default because the problem is not linearly separable so it will incur losses on some of those points already here.
![[file-20260621155919718.jpg]]

#### Not linealy separable with high $\lambda$ example
Equally to high $\lambda$ for linearly separable sets, the boundaries will rotate more guided by where the bulk of points are.
![[file-20260621160707127.jpg]]


## Personal Intuition
### High $\lambda$ (Low Margin)
When we use a higher regularization parameter ($\lambda$), we are not only pushing the boundaries further away. We're also rotating the boundaries so it minimizes the total penalty across the bulk of points fallinf inside the margin boundaries (Suports).

### Low $\lambda$ (Hard Margin)
So using a low $\lambda$ means that, **if the data is linearly separable**, no point is inside the margin boundary, it crosses to the nearest point from decision boundary and replicates the distance in the other margin boundary.


**Coming Next**: [[Regularization and Generalization]] 