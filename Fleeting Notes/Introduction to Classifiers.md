---
type: fleeting
created: 2026-06-12 00:33
tags: [fleeting]
---
# Quick capture — 2026-06-12

Ok, now lets look to the previous movie recommendator problem a little bit more grometrically.

We represent each movie as a point in space. To facilitate illustration, we asume the coordinates of those vectors are actually real numbers rather than binary numbers, so that these examples are a little more scattered.

![[file-20260613160917611.jpg]]

$x$ are feature vectors and $y$ are the labels associated with each one. Each $x,y$ pair is one training example.

The training set as a whole is now a set, which we will denote $S_n$ , where $n$ is 4, since we have 4 training examples.

Now, the training set can be described mathematically like this:

$$S_n = \left\{ \left( x^{(i)}, y^{(i)} \right), i=1, \dots, n \right\}$$

In Plain English: "_We have a training dataset named $S$ containing n examples. Each example consists of a pair of data points: an input $x$ (which can be made of more than one number (coordinate), depending the [[vectorial space]] $\mathbb{R}^n$) and a correct label $y$."

**This is the whole information that's provided to the learning algorithm.**

But the task that we really wish to solve is the one that corresponds to the rest of the tests examples.

![[file-20260613163122017.jpg]]

So the example here is also a pair of a vector, a future vector, associated with a label, but we don't know the label, the label is unknown until i actually see and view the movie, and then i know wether i would have wanted to see it or not. But the task is to label future movies rather than the training samples for which i already knew the labels.

So, on the basis of the training set, i need to map each new example to a corresponding label.

**What we need is a classifier that maps from points to corresponding labels.**
![[file-20260613163743406.jpg]]
So $h$ classifier here maps the space where the examples live ($\mathbb{R}^2$), to the corresponding labels ($\{-1,1\}$). So if i apply the classifier to any point outside the training set, it will give me one label.

What the classifier does is divide the space into escencially two halves, one that's labeled 1, and the other that's labeled -1 (for this example).


A few examples:
![[file-20260613164601729.jpg]]
Here's a linear classifier known as a [[linear classifier]]. because it divides the space linearly into two halves. One part of the space all the examples are mapped to +1 and in the other half map to -1.
Now, it does not agree well with the training examples that we have, because it missclassifies this vector as a - 1 and it misclassifies another vector as a +1.

So we need to evaluate how good the classifier is in relation to the **training examples** that we have. To this end we define something called [[training error]] ($\varepsilon_n$)

Training error:
$$\underset{\text{fraction of errors}}{\mathcal{E}_n(h)} = \frac{1}{n} \sum_{i=1}^{n} \underbrace{\left[\!\left[ h\left(x^{(i)}\right) \neq y^{(i)} \right]\!\right]}_{= \begin{cases} 1 & \text{if error} \\ 0 & \text{o.w} \end{cases}}$$


- **$\mathcal{E}_n(h)$**: This represents the total error ($\mathcal{E}$) for a specific classifier model (represented by $h$, which stands for "hypothesis"). The subscript $n$ indicates that this error is calculated over a dataset containing $n$ examples. As the handwritten note correctly states, this gives you the **"fraction of errors"**.
- **$\frac{1}{n} \sum_{i=1}^{n}$**: This is the standard mathematical way to calculate an average. The sum symbol ($\sum$) loops through every single example in your dataset—from the first item ($i=1$) to the last item ($n$)—adds up the results, and then divides by the total number of items ($n$).
- **$[\![ h(x^{(i)}) \neq y^{(i)} ]\!]$**: This double-bracket notation is called an **Iverson bracket** (or an indicator function). It evaluates the true/false statement inside it and outputs a simple number based on the result:
    - **$h(x^{(i)})$**: What your model _predicts_ for a given input $x$.
    - **$y^{(i)}$**: The actual _true_ label for that input.
    - **$\neq$**: The "not equal to" sign checks if your model made a mistake.

In plain English, it calculates the exact percentage of times your model guessed the wrong answer on a given dataset. Here is the breakdown of what each piece of the notation means:

**To define if a classifier is good, we look at the percentage of errors, if its higher than our defined thresholds its 'bad', if its lower is 'good'.**

So the previous example is a bad classifier because it missclasified 2 out of 4 vectors (50% error.)


Here is an example of what it seems a good classifier (0% error):
![[file-20260613170434136.jpg]]


So, linear classifiers are not the only way of matching lables to a testing set, we also have other kind of classifiers that may do better with some problems:
![[file-20260613170729822.jpg]]
![[file-20260613170748321.jpg]]
some classifiers with error 0.

There's an issue called **[[Generalization]]: How well the classifier that we train on the training set generalizes or applies correctly, similarly to the test examples, as well.**

This is at the heart of machine-learning problems, tha ability to generalize from the training set to the test set.

**Hypothesis space:** the set of possible classifiers
Each classifier represents a possible “hypothesis" about the data; thus, the set of possible classifiers can be seen as the space of possible hypothesis.

The more complex set of possible classifiers we consider, the less well we are likely to generalize. So we would wish to, in general, solve these problems by finding a small set of possibilities that work well on the training set, so as to generalize well on the test set.

In summary:
Training data can be graphically depicted on a (hyper)plane. **Classifiers** are **mappings** that take **feature vectors as input** and produce **labels as output**. A common kind of classifier is the **linear classifier**, which linearly divides space(the (hyper)plane where training data lies) into two. Given a point  in the space, the classifier  outputs  or , depending on where the point  exists in among the two linearly divided spaces.

Next: [[Different Kinds of Supervised Learning Classification vs. Regression]]






---
**Promoted to:** <!-- link the permanent note once created -->