We'll be talking about how to turn machine learning problem into optimization problems. **That is we are going to turn the problem on finding a linear classifier into an optimization problem, that can be solved in many ways.**

We'll be talking also about what linear large margin classification is and introduce notions such as margin loss and regularization.

In linear clasification, the perceptron could lead to many true solutions, since there are so many vectors that can divide a linear separable set.
The idea is to find the solution that is more 'in the middle' of both classifications, this is known as a **Large Margin Classifier**, since it leaves the most possible separation between the two classifications.
![[file-20260620162124132.jpg]]

Let's say the test examples that we see in the future are noisy versions of the training examples, in that sense, all the examples that we see are a cloud of points around each of the training examples. Si the classifier that hugs the training example very closely actually starts misclassifying those noisy versions of training examples very quickly.
![[file-20260620162328869.jpg]]

## Finding a large margin linear classifier as an optimization problem

We can divide this problem as stages.
First, we draw one decision boundary, with two adjoining boundaries that we call **margin boundaries**.![[file-20260620162829315.jpg]]
We have a margin boundary on the negative side and other on the positive side. These boundaries are equidistant from the actual decision boundary.
The goal here is now to use these margin boundaries essentially to define a fat decision boundary.

Now there are two aspects to this problem when we cast this into an optimization problem.
 - One is this favoring margin boundaries that are far apart from their decision boundaries. That's called a regularization term.
 - The other one is a counterweight that, as we push these margin boundaries apart, trying to still fit the decision boundary between the set of examples, we might start violating the preferenca that all the training examples are oustide this fat boundary. They may go within the fat boundary or even be misclassified. This counterbalance is quantified in terms of a loss function that we may incur on each of the examples that does not fit this ideal picture. 

These two points now define an objective function for selecting the parameters $\theta$ and $\theta_0$ that's a balance between the loss (how examples fit within this ideal notion) and regularization (our preference towards ñarge mounting solutions.)

So we will formalize the objective function and then find parameters $\theta$ and $\theta_0$ that optimize, minimize, this objective function.
![[file-20260620164113158.jpg]]
## Personal Intuition
So, previously we defined how to obtain a linear classifier using the perceptron algorithm, the thing here is that the solution could be any vector that linearly separates the two labels. Now we will add optimization to choose the best linear classificator for a linear separable set.



**Coming Next**: [[Margin Boundary]]