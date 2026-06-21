Let's formalize the objective function and then find parameters $\theta$ and $\theta_0$ that optimize, minimize, this objective function.

First of all, we must define what exactly the margin boundaries are and how we can control them, how far they are from the decision boundary.

As we've seen, here's the equation of the decision boundary $$\theta \cdot x + \theta_0 = 0$$
Meaning that the linear function $\theta \cdot x + \theta_0$ takes the value 0 at exactly on the decision boundary.

If we add a number to this linear function, it means we are pushing the decision boundary in $\theta$ direction so as we use $\theta_0$ to move away from the origin, we can use another constant to move away from the decision boundary to create the margin boundaries (parallel to the decision boundary).

So **we can define the positive margin boundary as the set of $x$'s where this linear function takes the value 1 exactly:**
$$\theta \cdot x + \theta_0 = 1$$

So this is a new boundary, because the linear function can't be 1 and 0 at the same time.

And **we also can define the negative margin boundary as the set of $x$'s where this linear function takes the value -1 exactly** so we are pushing the boundary the other side:
$$\theta \cdot x + \theta_0 = -1$$

So now, we have two parallel boundaries equidistant from the decision boundary. Now we can try to push these apart, because the margin boundaries themselves are also defined by the parameters that we wish to optimize ($\theta$,$\theta_0$).

Now, it remains to understand the distance the margin boundaries take from the decision boundary and how we can control it.

So, let's say we divide each parameter vector by it's norm (normalization):

$$\frac{\theta }{||\theta||}\cdot X + \frac{\theta_0}{||\theta||} = 0$$
It still defines the same decision boundary.
As a result there is the norm of theta, a free new parameter that we have not yet used. It is a new degree of freedom that gives the same decision boundary as a result.

## Using the norm to push the margin boundaries apart
We can use this norm to push the margin boundaries apart.
### Intuitively
As we move away from the decision boundary, then the value of the linear function increases at a rate that's related to the magnitude of the parameter vector of $\theta$. So the larger the magnitude of $\theta$ is, the faster that linear function reaches the value of the decision boundary (-1 or 1). As a result in terms of the distance from the decision boundary, the closer the margin boundary is to the decision boundary, the larger the value of norm of $\theta$ is. So the norm of $\theta$ does control the distance that the margin boundaries are drawn next to the decision boundary, and it is a degree of freedom that is available for us to use. 
So now when we select the parameters $\theta$ and $\theta_0$, it defines the orientation of the decision boundary as well as where the margin boundaries are drawn.

So now we have parameterized this large margin decision boundary. So we can now turn to defining what the result optimization problem would be.
![[file-20260620175100100.jpg]]

### Formally
The norm of $\theta$ is the regularization term that allows us to control how far the margin boundaries are from the decision boundary.
![[file-20260620175042869.jpg]]

So here we have a n