## Training Error for linear classifier
If it goes through origin we just erase the offset $\theta_0$ from the formula.
$$\mathcal{E}_n(\theta, \theta_0) = \frac{1}{n} \sum_{i=1}^{n} [\![ y^{(i)}(\theta x^{(i)} + \theta_0) \leq 0 ]\!]$$
- **$\mathcal{E}_n(\theta, \theta_0)$**: The total training error over the dataset, given the model's learned parameters (the weight vector $\theta$ and the bias/offset $\theta_0$).

- **$\frac{1}{n} \sum_{i=1}^{n}$**: This averages the error across all $n$ examples in the training dataset.

- **$[\![ \dots ]\!]$**: This is the indicator function (sometimes called an Iverson bracket). It outputs a 1 if the condition inside is true (representing an error) and a 0 if the condition is false.

- **$y^{(i)}$**: The true target label for the $i$-th training example. In binary classification, this is typically either +1 or -1.

- **$\theta x^{(i)} + \theta_0$**: The raw predicted score of the linear classifier for the $i$-th feature vector $x^{(i)}$.

- **$y^{(i)}(\theta x^{(i)} + \theta_0) \leq 0$**: This evaluates whether the model made a mistake. If the actual label and the predicted score have different signs, their product will be negative (or zero), triggering the indicator function to count it as a misclassification.

Now we defined the set of linear classifiers, we understand how to parameterize it, and we have defined how to measure error for a set of linear classifiers. We can now turn to the problem of actually finding a linear classifier that agrees with the training examples to the extent possible.

### The Perceptron Algorithm
We're going to progressively define a learning algorithm that takes a learning set and finds an easily parameter vector $\theta$, so is a linear classifier through origin, that hopefully correctly classifies all the training examples.

This algorithm will succeed if there exist any linear classifiers through origin that currently classifies the training samples. Then the algorithm will find a solution to that problem.

#### Developing the Perceptron
The perceptron is one of the simplest and oldest machine learning algorithms for binary classification. Imagine you have a scatter plot of points belonging to two different categories (like "spam" and "not spam"). The perceptron’s goal is to draw a straight line (or a "hyperplane" in higher dimensions) that perfectly separates the two groups.

It does this by making a guess, checking if that guess is right, and making small corrections every time it makes a mistake. Over time, it nudges the line into the correct position.
![[file-20260614125354088.jpg]]

### Step-by-Step Algorithm Walkthrough

Here is exactly what the pseudocode in the image is doing line-by-line:
**1: procedure $PERCEPTRON({(x^{(i)}, y^{(i)}), i = 1,...,n}, T)$**
- This defines the function. It takes in your training data (which is a set of $n$ data points where each has features $x$ and a label $y$) and the number of epochs $T$.
**2: $\theta = 0$ (vector), $\theta_0 = 0$  (scalar)**  
- **Initialization:** The algorithm starts with a blank slate. It sets the weights ($\theta$) to a vector of all zeros and the offset ($\theta_0$) to exactly zero. At this point, the model has learned nothing.
**3: `for t = 1,...,T do`**
- **The Epoch Loop:** The algorithm will review the entire dataset $T$ times. Going through the data multiple times gives it multiple chances to fix its mistakes.
**4: `for i = 1,...,n do`**
- **The Data Point Loop:** Inside every epoch, the algorithm looks at every single data point $i$, one by one, from the first to the $n$-th.
**5: $\text{if } y^{(i)}(\theta \cdot x^{(i)} + \theta_0) \leq 0 \text{ then}$**
- **The Prediction and Mistake Check:** This is the core logic of the perceptron.
    - $\theta \cdot x^{(i)} + \theta_0$ is the model's "score" for the current point. If the score is positive, the model guesses class $+1$. If negative, it guesses class $-1$.
    - By multiplying the model's score by the true label $y^{(i)}$, we are checking for agreement. If both are positive or both are negative, the result of the multiplication is positive ($> 0$), meaning the model was right!
    - If the signs are different (e.g., the model guessed $-1$ but the truth was $+1$), the result is negative ($\leq 0$). **This `if` statement triggers only when the model makes a mistake.**
**6: $\theta = \theta + y^{(i)}x^{(i)}$** 
- **Weight Update:** Because the model made a mistake on point $i$, it needs to fix its line. It does this by adding the feature vector $x^{(i)}$ (scaled by the sign of the label $y^{(i)}$) to the current weight vector $\theta$. This mathematically "tilts" the decision boundary so that it is closer to classifying this specific point correctly next time.
**7: $\theta_0 = \theta_0 + y^{(i)}$**
- **Offset Update:** Similarly, it updates the offset by adding the label $y^{(i)}$ to it. This shifts the line horizontally/vertically to accommodate the point it got wrong.
**8: return $\theta$, $\theta_0$**
- After the algorithm has looped through all $n$ data points $T$ times, it stops. It outputs the final $\theta$ and $\theta_0$. These final parameters define the best line the perceptron was able to find to separate your data.


## Other References
**Notes on Perceptrons**:
![[Perceptron.pdf]]

****
**There's a bound to $k$ number of updates made by the perceptron algorithm** 
https://arxiv.org/pdf/1305.0208

**Geometry Intuition on Perceptrons (doesn't explain offset):**
https://www.youtube.com/watch?v=Fj7BgxI73TA


Next: [[Introduction to Hinge loss, Margin Boundaries and Regularization]]