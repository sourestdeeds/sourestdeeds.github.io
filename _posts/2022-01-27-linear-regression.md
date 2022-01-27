---
title: 'Linear Regression'
tags: [google course, machine learning glossary]
layout: post
mathjax: true
categories: [ML Crash Course]
permalink: /blog/:title/
---
{% assign counter = 1 %}
{% assign counter2 = 1 %}
{% assign link = "https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}

It has long been known that crickets (an insect species) chirp more frequently on hotter days than on cooler days. For decades, professional and amateur scientists have cataloged data on chirps-per-minute and temperature. As a birthday gift, your Aunt Ruth gives you her cricket database and asks you to learn a model to predict this relationship. Using this data, you want to explore this relationship.

First, examine your data by plotting it:

<br>
[![svg]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.svg#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.svg)
{% assign counter = counter | plus: 1 %} 
<br>

As expected, the plot shows the temperature rising with the number of chirps. Is this relationship between chirps and temperature linear? Yes, you could draw a single straight line like the following to approximate this relationship:

<br>
[![svg]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.svg#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.svg)
{% assign counter = counter | plus: 1 %} 
<br>

True, the line doesn't pass through every dot, but the line does clearly show the relationship between chirps and temperature. Using the equation for a line, you could write down this relationship as follows:

$$
y = mx + c
$$

where:

- $y$ is the temperature in Celsius—the value we're trying to predict.
- $m$ is the slope of the line.
- $x$ is the number of chirps per minute—the value of our input feature.
- $c$ is the y-intercept.

By convention in machine learning, you'll write the equation for a model slightly differently:

$$
y^{'} = b + w_1 x_1
$$

where:

- $y^{'}$ is the predicted [label](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology#labels) (a desired output).
- $b$ is the bias (the y-intercept), sometimes referred to as $w_0$.
- $w_1$ is the weight of feature $1$. Weight is the same concept as the "slope" $m$  in the traditional equation of a line.
- $x_1$ is a [feature](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology#features) (a known input).

To **infer** (predict) the temperature $y^{'}$ for a new chirps-per-minute value $x_1$, just substitute the $x_1$ value into this model.

Although this model uses only one feature, a more sophisticated model might rely on multiple features, each having a separate weight ($w_1$, $w_2$, etc.). For example, a model that relies on three features might look as follows:

$$
y^{'} = b + w_1 x_1 + w_2 x_2 + w_3 x_3
$$