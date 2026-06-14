- **Feature vectors ($x$)**
They are vectors who belong to $\mathbb{R}^d$ so they are $d$-dimensional vectors. $x \in \mathbb{R}^d$ 

- **Labels**
Targets or outputs. $y \in \{-1,1\}$

- **Training set**
The tasks in supervised machine learning are illustrated by a training set (examples) which are the list of pairs feature-vector. $S_n = \left\{ \left( x^{(i)}, y^{(i)} \right), i=1, \dots, n \right\}$ 

- **Classifier**
Is a mapping that takes as an input a feature vector and it outputs a label. It basically separates the space in n-parts so it labels vectors depending on which part they are. $h: \mathbb{R}^d \rightarrow \{-1, 1\}, \quad h(x) = 1, \quad \mathcal{X}^+ = \{ x \in \mathbb{R}^d : h(x) = 1 \}$ 

- **Training error**
To evaluate how good a classifier works with the training set. $\mathcal{E}_n(h) = \frac{1}{n} \sum_{i=1}^{n} \underbrace{\left[\!\left[ h\left(x^{(i)}\right) \neq y^{(i)} \right]\!\right]}_{= \begin{cases} 1 & \text{if error} \\ 0 & \text{o.w} \end{cases}}$ 
- **Test error**
To evaluate how good a classifier works with the test set. $\mathcal{E}(h)$

- **Set of classifiers**
We can effect generalization by limiting the choices that we have, the set of hypotheses or set of alternatives is also called the set of classifiers, so our classifier here belongs to a set ($\mathcal{H}$) . It's a limited set of options that we constrain ourselves to. $h \in \underline{\underline{\mathcal{H}}}$ 

Right now we are going to limit our choices of classifiers to linear classifiers, and we will just assume that if we do well in the training set, we will also generalize well.

Next: [[Linear Classifiers Mathematically Revisited]]
