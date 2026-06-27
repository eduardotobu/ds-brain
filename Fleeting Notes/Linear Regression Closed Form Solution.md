Closed form solution for linear regression optimization.

For many algorithms in machine learning, you can't do closed form solution.

It happened to be because our loss function (empirical risk) actually happens to be a convex function, that's why we can solve it.


### Closed Form vs. Gradient-based solutions

|**Feature**|**Closed-Form (Normal Equation)**|**Gradient-Based (Gradient Descent)**|
|---|---|---|
|**Approach**|Mathematical formula|Iterative steps|
|**Time Complexity**|$O(d^3)$ (due to matrix inversion)|$O(k \cdot n \cdot d)$|
|**Handling Large Datasets ($n$)**|Slow and memory-heavy|Fast and memory-efficient (using batches)|
|**Handling Many Features ($d$)**|Very slow (impractical for $d > 10,000$)|Scales extremely well|
|**Need Feature Scaling?**|No|**Yes** (Crucial for speed)|
|**Need Hyperparameters?**|No|**Yes** (Learning rate, epochs, etc.)|
|**Use Case**|Small-to-medium datasets, linear regression|Massive datasets, Neural Networks, complex models|