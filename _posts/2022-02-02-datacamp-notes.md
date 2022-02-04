---
title: 'DataCamp Notes'
tags: [datacamp, data science]
layout: post
mathjax: true
categories: [DataCamp Notes]
permalink: /blog/:title/
---
{% assign counter = 1 %}
{% assign counter2 = 1 %}
{% assign link = "https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}

### Definitions

- **Statistical Inference**
    - To draw probabilistic conclusions about what we might expect if we collected the same data again.
    - To draw actionable conclusions from data.
    - To draw more general conclusions from relatively few data or observations.

### Cross Validation

```python
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
```


***
### EDA

#### Distributions

```python
pip install empiricaldist
```

- **Probability Mass Function** is the probability that you get exactly $x$. 
- **Cumulative Distribution Function** is the probability that you get a value $<=x$.

```python
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y
```

```python
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success
```

***
### Decorators

```python
from functools import wraps

def timer(func):
    """A decorator that prints how long a function took to run."""
    # Define the wrapper function to return.
    # Decorate wrapper() so that it keeps func()'s metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.    
        t_start = time.time()
        # Call the decorated function and store the result.    
        result = func(*args, **kwargs)
        # Get the total time it took to run, and print it.    
        t_total = time.time() - t_start    
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper

```

```python
def memoize(func):
    """Store the results of the decorated function for fast lookup."""
    # Store results in a dict that maps arguments to results  
    cache = {}
    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # If these arguments haven't been seen before,
        if (args, kwargs) notin cache:
            # Call func() and store the result.      
            cache[(args, kwargs)] = func(*args, **kwargs)
        return cache[(args, kwargs)]
    return wrapper
```

```python
def print_return_type(func):
    """Print out the type of the variable that gets returned from every call of any function it is decorating."""
    # Define wrapper(), the decorated function
    def wrapper(*args, **kwargs):
        # Call the function being decorated
        result = func(*args, **kwargs)
        print('{}() returned type {}'.format(
            func.__name__, type(result)
        ))
        return result
    # Return the decorated function
    return wrapper
```

```python
def counter(func):
    """Print out how many times a function was called."""
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        # Call the function being decorated and return the result
        return func
        wrapper.count = 0
    # Return the new decorated function
    return wrapper
```