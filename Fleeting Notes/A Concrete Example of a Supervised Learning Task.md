---
type: fleeting
created: 2026-06-12 00:09
tags: [fleeting]
---
# Quick capture — 2026-06-12

**Movie recommender Problem**
Now let's take a very concrete example of how to specify a machine learning task and how to solve it.

I have a set of movies i've already seen and i was to use the experience from those movies to make recommendations of wheter i would like to see tens of thousands of other movies.

Step 1: Ilustrate the task for the learning algorithm.
For example, if i like a movie ill give it +1, and if i dont like it ill give it -1
![[file-20260612001558963.jpg]]
Ok, now i've already got four annotated examples that illustrate the task that i was to solve (how to recommend movies to me)?
The real task, though, is to recommend, correctly, movies out of those tens of thousands of unseen movies.
while the computer can easily understand the +-1 labels that i have annotated the movies with, it doesn't really understand anything about the movies, so i need to construct a feature vector, put the movies into a form that computers can understand.

![[file-20260612001940465.jpg]]
So we will compute a feature vector for each movie systematically so that we can add that procedure, apply that procedure to each of the movies, in the training set (set of examples we've already seen). To compute a feature vector we can ask different questions, like the movie was comedy?, was action?, was directed by spielberg? and we will add a 1 to each 'yes' and a 0 to each 'no' in order to construct the feature vector.

So at this point, we have a training set of examples-- vectors with the associated labels. These are in the form that a computer can now understand. I**ts task is to learn a mapping from those vectors to the labels. This is called a classifier as we will see later on.**

![[file-20260612002304141.jpg]]

So i wish to learn the mapping from examples, x to labesl, plus minus one, on the basis of the training set, and hope and guarantee that that mapping, if applied now in the same way to the test examples, it would work well, that is the task that i wish to solve.

Train set: movies that are labeled.
Test set: movies that are not yet labeled.

Next, we are diving into [[Introduction to Classifiers]]


---
**Promoted to:** <!-- link the permanent note once created -->