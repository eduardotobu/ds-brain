**Linear Regression** is a fundamental supervised machine learning algorithm used for predicting a continuous numerical value. It assumes a linear relationship between the input variables (features) and the single output variable (target).

## The Goal of the Model

The goal of linear regression is to find the best-fitting straight line (or hyperplane in higher dimensions) that maps the input features to that continuous target variable.

## The Core Equation

Linear regression hypothesis function:

$$f(x; \theta, \theta_0) = \sum_{i=1}^d \theta_i x_i + \theta_0 = \theta x + \theta_0$$

Here is the breakdown of what each component means:

- **$f(x; \theta, \theta_0)$**: The prediction function. It takes an input vector $x$ and uses the parameters $\theta$ and $\theta_0$ to output a predicted value.
- **$x$**: The input feature vector. If you have $d$ features (e.g., predicting house price based on square footage and number of bedrooms, where $d=2$), $x_i$ represents the value of the $i$-th feature.
- **$\theta$**: The weight vector (also called coefficients). These dictate how much each feature $x_i$ influences the final prediction.
- **$\sum_{i=1}^d \theta_i x_i$**: The algebraic sum of each feature multiplied by its corresponding weight.
- **$\theta_0$**: The bias term (or y-intercept). This shifts the regression line up or down. It represents the predicted value of $y$ when all features in $x$ are exactly zero.
- **$\theta x$**: This is the vector notation (often written as the dot product $\theta^T x$) which is a cleaner, more compact way to write the summation.

## Special Case: $\theta_0 = 0$

If you set the bias term to zero, the equation simplifies to $f(x) = \theta x$.

- **Geometrically:** This forces the regression line (or plane) to pass exactly through the origin $(0,0)$.
- **Use Case:** You typically only do this if you have a strong domain-specific reason to believe that when all input features are zero, the target must strictly be zero. Otherwise, leaving $\theta_0$ in the model allows for a better fit.


## **What if our outputs don't follow a linear trend?**
If our dependencies are not linear, the model will not represent the real phenomena well, but we still can scale a non-linear space to a linear space so our feature vector looks linear in this new space and linear regression can be sufficient.


## Empirical Risk

There are many ways to define loss, here we are going to define the most common way for calculating loss in linear regression. 

What we want to do is kind of quantify how our predictions is making deviate from the known values of $y$ 

To find the empirical risk, we sum up the individual losses for every data point in our training set (from $i = 1$ to $n$) and divide by the total number of points ($n$) to get the average.

The Empirical Risk, often denoted as $R_{emp}$ or $J(\theta)$, is calculated as:

$$R_{emp}(\theta) = \frac{1}{n} \sum_{i=1}^{n} (f(x^{(i)}; \theta) - y^{(i)})^2$$

If we substitute the linear regression hypothesis $f(x) = \theta^T x$ (assuming the bias is folded into the vector), the equation looks like this:

$$R_{emp}(\theta) = \frac{1}{n} \sum_{i=1}^{n} (\theta^T x^{(i)} - y^{(i)})^2$$

_Note: In many textbooks and courses, you will see a $\frac{1}{2}$ added in front (e.g., $\frac{1}{2n}$). This is a mathematical convenience that makes the calculus cleaner later on when taking the derivative; it does not change where the minimum value is located.

***The ultimate goal of training a linear regression model is Empirical Risk Minimization (ERM).***

When an algorithm (like Gradient Descent or Ordinary Least Squares) "trains" a linear regression model, it is actively searching for the specific set of weights ($\theta$) that results in the absolute lowest possible empirical risk. Geometrically, it is tweaking the slope and intercept of the line until the sum of the squared distances between the data points and the line is minimized.

### 1. Risk on the First Subset (Training Set)

The first equation defines the empirical risk over the first $n$ examples in your dataset:

$$R_n(\theta) = \frac{1}{n} \sum_{t=1}^{n} \frac{(y^{(t)} - \theta x^{(t)})^2}{2}$$

- **$R_n(\theta)$**: The empirical risk evaluated over $n$ samples.
- **$\sum_{t=1}^{n}$**: The summation starts at the first data point ($t=1$) and goes up to the $n$-th data point. (Note: The board uses $t$ as the index variable here instead of $i$, which is just a stylistic choice; they mean the same thing).
- **The Division by 2 ($/2$)**: You'll notice the squared error term is divided by 2. This is a very common mathematical convenience in machine learning. When you later take the derivative of this loss function (using Calculus) to minimize it, the exponent $2$ drops down and perfectly cancels out the division by $2$, making the resulting math much cleaner. It does not change the location of the optimal weights ($\theta$).


### 2. Risk on the Second Subset (Test/Validation Set)

The second equation defines the empirical risk over a _new_ batch of $n'$ examples:

$$R_{n'}(\theta) = \frac{1}{n'} \sum_{t=n+1}^{n+n'} \frac{(y^{(t)} - \theta x^{(t)})^2}{2}$$

- **$R_{n'}(\theta)$**: The empirical risk evaluated over a different set of size $n'$.
- **$\sum_{t=n+1}^{n+n'}$**: This is the most important part of this equation. Notice the indices: the summation _starts_ at $n+1$ and ends at $n+n'$.
- **The "Why":** By starting at $n+1$, this ensures that the data points used in this calculation are strictly separate from the first $n$ points. If $R_n$ represents your training error, $R_{n'}$ represents your generalization error on completely unseen data. Evaluating the risk on this separate set tells you if your model has actually learned the underlying pattern or if it just memorized the training data (overfitting).
## 2 Types of mistakes when using linear regression
**Structural**: The mapping between training vectors and $y$'s may be highly nonlinear, so the model doesn't represent the real phenomena.
**Estimation:** Even knowing the mapping is linear but you have very limited training data you cannot estimate it correctly.

## The Bias-Variance Tradeoff

![[file-20260625224504085.jpg]]
