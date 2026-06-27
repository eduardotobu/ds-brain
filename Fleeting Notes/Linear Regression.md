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


Coming Next: [[Empirical Risk]]