---
title: 'Useful Algorithms'
tags: [kaggle, algorithm, levenshtein distance, dbscan]
layout: post
mathjax: true
categories: [Algorithms]
---

{% assign counter = 1 %}
{% assign link = "https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}

I find new and interesting algorithms and forget about them all the time! So from now on I'm going to save them all here.

<br>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
{% assign counter = counter | plus: 1 %} 
<br>

### DBSCAN 

[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN), aka Density-based spatial clustering of applications with Noise, is a [clustering algorithm](https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html) that identifies clusters by finding regions that are densely packed together, in other words, points that have many close neighbors.

DBSCAN is the [best](https://towardsdatascience.com/how-dbscan-works-and-why-should-i-use-it-443b4a191c80) clustering algorithm (better than k-means clustering or hierarchical clustering) for several reasons:

- It can determine the optimal number of clusters on its own.
- It can find clusters of abnormal shapes, not just circular ones.
- It’s robust enough to not be affected by outliers.

### Apriori Algorithm

The [Apriori](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/) Algorithm is an association rule algorithm and is most commonly used to determine groups of items that are most closely associated with each other in an itemset.

To give an example, suppose we had a database of customer purchases at a grocery store. The Apriori [Algorithm](https://www.educative.io/edpresso/what-is-the-apriori-algorithm) can be used to determine which pairs or groups of items are most frequently purchased together.

There are two main parameters: support and confidence. Support refers to the frequency that the item occurs, while the confidence represents the conditional probability that one item was purchased given that one or more other items were purchased.

###  Holt-Winters Exponential Smoothing

[Holt-Winters](https://towardsdatascience.com/holt-winters-exponential-smoothing-d703072c0572) [Exponential Smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing), aka Triple Exponential Smoothing, is a popular forecasting technique for time series data that exhibits both a trend and seasonality.

It’s called [triple exponential smoothing](https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/) because it takes into consideration the level of the data, the trend of the data, and the seasonality of the data.

The benefits of this method of forecasting over other methods, like ARIMA, are:

- It’s simple to understand and implement.
- It’s fairly accurate.
- And it’s computationally inexpensive and non-resource intensive.

### Matrix Factorization

[Matrix factorization](https://en.wikipedia.org/wiki/Matrix_factorization_%28recommender_systems%29) [algorithms](https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b) are a type of [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) algorithm commonly used to build recommendation systems.

The idea behind collaborative filtering is that is predicts the interests of a given user based on the interests of other similar users. This is known as a memory-based approach, but another approach is a model-based approach where machine learning algorithms are used to predict users’ ratings of unrated items.

### Levenshtein Distance

The [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) a simple algorithm used to determine the similarity between two strings.

Specifically, it is equal to the minimum number of single character edits (substitutions, additions, deletions) to change one word to another.

For example, the Levenshtein distance between “taco” and “eggs” is 4. The [Levenshtein distance](https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/) between “cross” and “crossword” is also 4. Intuitively, it’s odd that these pairs are ranked the same, which shows this algorithms limitations.

And so, two better string similarity algorithms that I recommend looking into are the Trigram and Jaro-Winkler algorithms.

### Epsilon-Greedy Algorithm

The Epsilon-Greedy Algorithm is a simple approach to the [multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit), which is a representation of the exploration vs exploitation dilemma.

The idea behind the problem is that there are k different alternatives that each return a different reward, but you don’t know the reward for any of the alternatives. And so, you start by exploring different alternatives, and as time goes by, there is a tradeoff between exploring more options and exploiting the highest paying option.

With the [Epsilon-Greedy](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/) Algorithm, a randomly alternative is chosen a fraction, $\varepsilon$, of the time. For the rest of the time $(1-\varepsilon)$, the alternative with the highest known payout (reward) is chosen. $\varepsilon$ is a parameter that you have to set.

Better solutions include the upper confidence bound solution and Bayesian Thompson sampling.

### Harris Corner Detector

The [Harris Corner Detector](https://medium.com/data-breach/introduction-to-harris-corner-detector-32a88850b3f6) is an operator that is used in computer vision algorithms to identify corners in an image. This is important for image processing and computer vision because corners are known to be important features in an image.

- In the flat region, there is no gradient change (color change) in any direction.
- In the edge region, there is no gradient change in the direction of the edge.
- Only in the corner region is there a gradient change in all directions.

Thus, this method is used throughout an image to determine where the corners in the image are.