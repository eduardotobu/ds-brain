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
So, we're starting with a 0 parameter vector. Well, there, our parameters are set to 0. 
$$\theta = 0 \text{ (vector)}$$
Then we go through training examples, take te $i$-th example and we check wether it is misclassified. 