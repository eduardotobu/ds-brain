Let's undersant through examples how constrained a set of linear classifiers really is. So if we take a set of examples, the training set, and ask whether it's separable, wheter there 

**Linear separable points:**
Through origin:
![[file-20260613212305770.jpg]]
Not through origin:
![[file-20260613212337735.jpg]]

**Not linear separable:**
![[file-20260613212358515.jpg]]

Formally, the training examples are said to be linearly separable if there exists a linear classifier that correctly classifies those training points.

So if there is a linear classifier in the set that would correctly classify those training points, then the training points are said to be linearly separable.

And we've already seen cases where that linear separation does not succed, so the set of linear classifiers is inherently constrained.

**Formal definition:** Training examples $S_n = \{(x^{(i)}, y^{(i)})\}, i = 1, \dots, n$ are _linearly separable_ if there exists a parameter vector $\hat{\theta}$ and offset parameter $\hat{\theta}_0$ such that $y^{(i)}(\hat{\theta} \cdot x^{(i)} + \hat{\theta}_0) > 0$ for all $i = 1, \dots, n$.


# Gemini help on formal definition
### The Clever Math Trick

The core of the definition is this part: **$y^{(i)}(\hat{\theta} \cdot x^{(i)} + \hat{\theta}_0) > 0$**

This is a very clever mathematical shortcut to check if your line made a mistake. Here is how it works:

1. **The Prediction:** The part inside the parentheses $(\hat{\theta} \cdot x + \hat{\theta}_0)$ calculates which side of the line a dot is on. It will output a positive number for one side, and a negative number for the other side.
    
2. **The True Label:** The $y$ outside the parentheses is the actual color of the dot, represented as a **+1** (let's say, blue) or a **-1** (red).
    
3. **The Multiplication:** * If the formula guesses a positive number and the dot is actually a +1, multiplying them gives a positive result (**> 0**). _Correct!_
    
    - If the formula guesses a negative number and the dot is actually a -1, multiplying two negatives also gives a positive result (**> 0**). _Correct!_
        
    - If the formula guesses positive but the dot is actually a -1, multiplying them gives a negative result. _Mistake!_
        

### The "For All" Requirement

The very end says **"for all $i = 1, \dots, n$"**. This just means "for every single dot in the dataset."

So, putting it all together in plain English:

**"A dataset is linearly separable if you can find at least one straight line where the prediction and the true label agree (resulting in a number greater than zero) for every single example."**

Does the math trick using multiplication to check for "agreement" make sense, or would you like to explore what happens when data is _not_ linearly separable?



Next: [[The Perceptron Algorithm]]
