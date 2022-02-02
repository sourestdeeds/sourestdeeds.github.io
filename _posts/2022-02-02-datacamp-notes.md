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