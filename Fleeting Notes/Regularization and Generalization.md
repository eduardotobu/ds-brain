Let's see how changing the values of the regularization parameter $\lambda$, changes the training and test average losses/errors.

As we emphasize the margin, we are less and less able to fit the training samples, so the losses would increase.

If we decrease the value of $\lambda$, we emphasize the training losses, and as a result, we would find solutions $\theta$ and $\theta_0$ where the training loss is smaller.

We can understand the previous two paragraphs as a function of $\lambda$, but I'll show it as a function of $\frac{1}{\lambda}$, that's a little bit easier conceptually. So let's multiply our loss function by $\frac{1}{\lambda}$:
$$\frac{1}{\lambda} J(\theta, \theta_0) = \frac{1}{\lambda n} \sum_{i=1}^{n} \text{Loss}_h \left( y^{(i)} (\theta \cdot x^{(i)} + \theta_0) \right) + \frac{1}{2} \|\theta\|^2$$
So the lambda in the regularization parameter goes away and lambda now affects the loss part of the function.
Let's define  $C=\frac{1}{\lambda n}$. As $C$ is small, we emphasize simpler solutions that have large margin. As $C$ is large, we emphasize losses.

So we plot how the average loss behaves as a function of $C$:
![[file-20260621165143205.jpg]]It is monotonically decreasing. It may not go exactly to 0, because the problem may not be linearly separable (so the average loss couldn't be 0).
So the previous curve is our training loss.

Now imagine that for any value of C we get the solution and we can evaluate also the test loss or test error corresponding to that $\theta$ and $\theta_0$, cannot really do that but we can imagine what their solution would be.
That curve as a function of $C$ would look something like this 'U-shaped' graph, tipically higher than the training loss.
![[file-20260621165828611.jpg]]

So we see that it exists an optimal value of $C$ that would minimize the test error.
This optimal value of $C$ divides the problem in two regimes: To the left of that point, the model is underfitting, to the right of that point the model is overfitting.
![[file-20260621170200249.jpg]]
So, if im underfitting and i make the method more complex (increase the value of $C$), i will actually improve my test performance. If I'm overfitting, then making the method even more complex will result in a poorer test loss or error.

So how do we get This optimal $C$ value ($C^*$). We can simulate evaluating test losses as follows.

We divide the training set in two parts, one which is the new training set and the remaining portion will be the validation set, which is now our pretend test set. How we use this?
1. We take the smaller training set and find $\theta$ and $\theta_0$ by optimizing the objective function only for those examples.
2. We take the corresponding values of $\theta$ and $\theta_0$ and use the validation examples to evaluate an approximate test error. Which now we're calling validation error.
3. Then we find the value of $C$ that optimizes the performance on those validation examples. So we don't get $C^*$ exactly, but we get some estimate approximate to it.

## References
Great Explanation of Train, Test and Validation sets.
https://mlu-explain.github.io/train-test-validation/

## Personal Intuition
Test set is were we define our optimal parameters $\theta$ and  $\theta_0$ , validation set is were we define our hyperparameters (in this case the regularization parameter $\lambda$). So we finally calculate the actual model error on our test set.

**If the training loss is low and the validation loss is high, the model might be overfitting.** This is because the model memorized the training set, but got less 'robust' for future examples that hasn't happened, so the model it's not general enough.

**If the training loss is high and the validation loss is high, the model might be underfitting.** So the model is bad for training set and also for test set.


## Gemini's Intuition on Sets
### 1. The Training Set (Learning Parameters)

This is the data your model actually "learns" from. During the training phase, the algorithm processes this data to find the optimal model parameters—specifically, your weights ($\theta$) and biases ($\theta_0$).

### 2. The Validation Set (Tuning Hyperparameters)

The validation set is used to evaluate the model while you are still building it. You use this set to tune **hyperparameters** (settings that dictate **_how_ the model learns**, rather than what it learns). This includes choosing your regularization parameter ($\lambda$ or $C$), deciding the depth of a decision tree, or adjusting the learning rate. It helps you prevent overfitting before the final evaluation.

### 3. The Test Set (Final Evaluation)

You were also correct about the final step: this is where we calculate the true model error. However, a golden rule in Machine Learning is that **the test set must never be touched during the training or tuning phases**. It acts as completely unseen, real-world data. If you used it to define $\theta$, $\theta_0$, or your regularization parameters, your model would essentially be "cheating" by peeking at the final exam, and your error metric would not reflect how the model will perform in the wild.


**Coming Next:** [[Optimization Algorithms]] 