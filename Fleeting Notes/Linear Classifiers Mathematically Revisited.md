Now, let's try to understand more deeply the linear classifiers.

Essentialy, **a linear classifier is a function that separates a space into two halves, linearly.**

At one side, the clasifier says that all examples are positive and on the other side, they're labeled negative.

The dividing line is also **called decision boundary**.

In 2D, that decision boundary is a line, in 1D is a point, in 3D is a plane.
In higher dimensions, it's called hyperplane that divides the space into two halves.

A linear decision boundary can be in any direction with any slope, like a common linear equation.

![[file-20260613195105654.jpg]]

So now we need to parameterize the linear classifiers, so that we can effectively search for the right one given the training set, let's see how we can index and parameterize these classifiers.

# Linear classifiers through origin.
Let's start with even a more restricted version of linear classifiers, the ones that go through origin. The equation for this line would be the set of all points that satisfy an equation for a line through origin.

To define that we're introducing 2 new parameters.
$\theta_1$ that multiplies $x_1$ coordinate ant $\theta_2$ that multiples $x_2$ coordinate and that is equal to 0. In other words, the dot product of 2 vectors equalling to 0, meaning the vector $\theta$ and $x$ are orthogonal. So the decision boundary is defined like:

$$\{x: \theta_1 x_1 + \theta_2 x_2 = 0\}$$

We can write this in the vector form by introducing $\theta$ and $x$ as vectors with two coordinates. So we can write down the decision boundary as all x such that the [[dot product]] between $\theta$ and $x$ is equal to 0 (meaning orthogonallity).

$$\{x:\theta \cdot x  = 0\}$$
Where the parameter vector $\boldsymbol{\theta}$ and the feature vector $\mathbf{x}$ are defined as:

$$\boldsymbol{\theta} = \begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

From this equation, yoy can see that the parameter vector of $\theta$ is orthogonal to all the points that lie on the decision boundary.

Ok, now let's look at  $\theta \cdot x$ as a linear function.

So this is a linear function. Parameter vector of $\theta$ is fixed, $x$ varies, so it's your linear function of $x$.

It is positive in one side and negative in the other. 

So to define which side of the classifier each vector is you calculate the dot product of $\theta$ and each $x$. Since dot product calculates how much two vectors are aligned.
- **If dot product is > 0** then both $\theta$ and $x$ are pointing to the s**ame direction** so $x$ in one side of the clasiffier.
- **If dot product is < 0** then both $\theta$ and $x$ are pointing to **opposite directions** so $x$ is in the other side of the clasiffier.

This now allows us to define the set of linear classifiers through origin, let's be a little more precise.

We have a classifier, now parameterized by the vector $\theta$, So each choice of $\theta$ defines one classifier, it's oriented differently, but it also goes through origin. So that classifier is now simply the sign of $\theta \cdot x$, the label that it returns is a sign of that linear function.

**So this is the set of all linear classifiers through origin:**
$$\left\{ \begin{array}{l} h(x; \theta) = \text{sign}(\theta \cdot x) \\ \qquad \qquad \theta \in \mathbb{R}^d \end{array} \right\}$$
![[file-20260613203758130.jpg]]

Note that this association between the classifier and the parameter vector theta is not unique, there are multiple parameter vectors theta that defined exactly the same classifier. But each linear classifier through origin has at least many parameter vectors theta that correspond to the same decision boundary, same decision. So what is that degree of freedom here?. We only care for the sign of the dot product, so we get the same result for every posible $\theta$ of the linear clasifiar, so the norm of the parameter vector of theta is not relevant in terms of the decision boundary.
We'll use these degrees of freedom later on.

So $\theta \cdot x$ is the degree to which we classify the example positively or negatively, that degree only changes when we move orthogonaly to the decision boundary. 

The further from the decision boundary, the strongest their classification is (it can be strongly positive or strongly negative).


# Linear classifiers not through origin (general form)

![[file-20260613211012689.jpg]]

We can use this and define the set of linear classifiers without the constraint that they have to go through origin. The only difference here is that we can now move the decision boundary to be anywhere. not just those that go through origin. Again, we have an orientation of the boundary as well as the location.
We add $\theta_0$ that acts as an offset, so it contols the location of the boundary (where it is in relation to origin). And $\theta$ controls the orientation of that boundary. Again $\theta$ as a vector is orthogonal to the decision boundary.

$$\{x:\theta \cdot x + \theta_0 = 0\}$$
So once again:
- If $\theta \cdot x + \theta_0 > 0$, then the vector is in one side of the classifier.
- If $\theta \cdot x + \theta_0 < 0$, then the vector is in the other side of the classifier.

So we can define the set of all linear classifiers now using two parameters ($\theta$,$\theta_0$) as:
$$h(x; \theta, \theta_0) = \text{sign}(\theta \cdot x + \theta_0)$$ Again the mapping is not unique, there are many $\theta$ and $\theta_0$ that define the same decision boundary.

So when we look for a linnear classifier, we try to find parameters $\theta$ and $\theta_0$ on the basis of the traning set, and we want to selec those class parameters in such a way that the classifier makes correct decisions.

### More intuition on $\theta_0$ (the offset)
$\theta_0$ acts as a translator, it pushes the hyperplane away from the origin along the axis of the normal vector.
In precise geometric terms, the perpendicular distance from the origin to the decision boundary is exactly:

$$d = \frac{-\theta_0}{||\theta||}$$

- If $\theta_0$ is negative, the boundary is pushed _in the direction_ that $\theta$ is pointing.
- If $\theta_0$ is positive, the boundary is pushed _away_ from the direction $\theta$ is pointing.
- The magnitude of $\theta_0$ (relative to the length of the weight vector $||\theta||$) dictates exactly how far the boundary shifts.

Next concept: [[Linear separation]]
