---
title: 'Permutation Importance'
tags: [kaggle, permutation importance, machine learning explainability]
layout: post
mathjax: true
categories: [Machine Learning Explainability]
published: true
permalink: /blog/:title/
---

One of the most basic questions we might ask of a model is: What features have the biggest impact on predictions?

This concept is called feature importance.

There are multiple ways to measure feature importance. Some approaches answer subtly different versions of the question above. Other approaches have documented shortcomings.

We'll focus on permutation importance, compared to most other approaches, permutation importance is:

- Fast to calculate.
- Widely used and understood.
- Consistent with properties we would want a feature importance measure to have.


### How It Works

Permutation importance uses models differently than anything you've seen so far, and many people find it confusing at first. So we'll start with an example to make it more concrete.

Consider data with the following format:

<br>
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-08-permutation-importance/1.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-08-permutation-importance/1.png)<br>

We want to predict a person's height when they become 20 years old, using data that is available at age 10.

Our data includes useful features (height at age 10), features with little predictive power (socks owned), as well as some other features we won't focus on in this explanation.

Permutation importance is calculated after a model has been fitted. So we won't change the model or change what predictions we'd get for a given value of height, sock-count, etc.

Instead we will ask the following question: If I randomly shuffle a single column of the validation data, leaving the target and all other columns in place, how would that affect the accuracy of predictions in that now-shuffled data?

<br>
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-08-permutation-importance/2.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-08-permutation-importance/2.png)<br>

Randomly re-ordering a single column should cause less accurate predictions, since the resulting data no longer corresponds to anything observed in the real world. Model accuracy especially suffers if we shuffle a column that the model relied on heavily for predictions. In this case, shuffling *height at age 10* would cause terrible predictions. If we shuffled *socks* owned instead, the resulting predictions wouldn't suffer nearly as much.

With this insight, the process is as follows:

- Get a trained model.
- Shuffle the values in a single column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
- Return the data to the original order (undoing the shuffle from step 2). Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.

### Code Example

Our example will use a model that predicts whether a soccer/football team will have the "Man of the Game" winner based on the team's statistics. The "Man of the Game" award is given to the best player in the game. Model-building isn't our current focus, so the cell below loads the data and builds a rudimentary model.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)
```

Here is how to calculate and show importances with the [eli5](https://eli5.readthedocs.io/en/latest/) library:

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

<table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>
        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1750
                    ± 0.0848 
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Goal Scored
            </td>
        </tr>
        <tr style="background-color: hsl(120, 100.00%, 91.68%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0500
                    ± 0.0637 
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Distance Covered (Kms)
            </td>
        </tr>
        <tr style="background-color: hsl(120, 100.00%, 92.42%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0437
                    ± 0.0637 
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Yellow Card
            </td>
        </tr>
        <tr style="background-color: hsl(120, 100.00%, 95.81%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0187 
                    ± 0.0500
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Off-Target
            </td>
        </tr>
        <tr style="background-color: hsl(120, 100.00%, 95.81%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0187
                    ± 0.0637
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Free Kicks
            </td>
        </tr>
        <tr style="background-color: hsl(120, 100.00%, 95.81%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0187  
                    ± 0.0637
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Fouls Committed
            </td>
        </tr>
        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0125
                    ± 0.0637
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Pass Accuracy %
            </td>
        </tr>
        <tr style="background-color: hsl(120, 100.00%, 96.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0125 
                    ± 0.0306 
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Blocked
            </td>
        </tr>
        <tr style="background-color: hsl(120, 100.00%, 98.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063  
                    ± 0.0612
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Saves
            </td>
        </tr>
        <tr style="background-color: hsl(120, 100.00%, 98.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0063
                    ± 0.0250
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Ball Possession %
            </td>
        </tr>
        <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0
                    ± 0.0000
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Red
            </td>
        </tr>
        <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0
                    ± 0.0000 
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Yellow &amp; Red
            </td>
        </tr>
        <tr style="background-color: hsl(0, 100.00%, 100.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0000
                    ± 0.0559 
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                On-Target
            </td>
        </tr>
        <tr style="background-color: hsl(0, 100.00%, 98.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0063
                    ± 0.0729
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Offsides
            </td>
        </tr>
        <tr style="background-color: hsl(0, 100.00%, 98.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0063
                    ± 0.0919
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Corners
            </td>
        </tr>
        <tr style="background-color: hsl(0, 100.00%, 98.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0063
                     ± 0.0250
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Goals in PSO
            </td>
        </tr>
        <tr style="background-color: hsl(0, 100.00%, 95.81%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0187
                    ± 0.0306
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Attempts
            </td>
        </tr>
        <tr style="background-color: hsl(0, 100.00%, 91.68%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0500
                    ± 0.0637
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                Passes
            </td>
        </tr>
    </tbody>
</table>

### Interpreting Permutation Importances

The values towards the top are the most important features, and those towards the bottom matter least.

The first number in each row shows how much model performance decreased with a random shuffling (in this case, using "accuracy" as the performance metric).

Like most things in data science, there is some randomness to the exact performance change from a shuffling a column. We measure the amount of randomness in our permutation importance calculation by repeating the process with multiple shuffles. The number after the \\( \pm \\) measures how performance varied from one-reshuffling to the next.

You'll occasionally see negative values for permutation importances. In those cases, the predictions on the shuffled (or noisy) data happened to be more accurate than the real data. This happens when the feature didn't matter (should have had an importance close to 0), but random chance caused the predictions on shuffled data to be more accurate. This is more common with small datasets, like the one in this example, because there is more room for luck/chance.

In our example, the most important feature was *Goals scored*. That seems sensible. Soccer fans may have some intuition about whether the orderings of other variables are surprising or not.

### Example

Calculate permutation importance with a sample of data from the [Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) competition.

```python
# Loading data, dividing, modeling and EDA below
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)

# Remove data with extreme outlier coordinates or negative fares
data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
                  'fare_amount > 0'
                  )

y = data.fare_amount

base_features = ['pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude',
                 'passenger_count']

X = data[base_features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)


# show data
print("Data sample:")
data.head()
```

```python
train_X.describe()
```

<div class="table-wrapper" markdown="block">

|   |                           key | fare_amount |         pickup_datetime | pickup_longitude | pickup_latitude | dropoff_longitude | dropoff_latitude | passenger_count |
|--:|------------------------------:|------------:|------------------------:|-----------------:|----------------:|------------------:|-----------------:|----------------:|
| 2 |  2011-08-18 00:35:00.00000049 |         5.7 | 2011-08-18 00:35:00 UTC |       -73.982738 |       40.761270 |        -73.991242 |        40.750562 |               2 |
| 3 |   2012-04-21 04:30:42.0000001 |         7.7 | 2012-04-21 04:30:42 UTC |       -73.987130 |       40.733143 |        -73.991567 |        40.758092 |               1 |
| 4 | 2010-03-09 07:51:00.000000135 |         5.3 | 2010-03-09 07:51:00 UTC |       -73.968095 |       40.768008 |        -73.956655 |        40.783762 |               1 |
| 6 |   2012-11-20 20:35:00.0000001 |         7.5 | 2012-11-20 20:35:00 UTC |       -73.980002 |       40.751662 |        -73.973802 |        40.764842 |               1 |
| 7 |  2012-01-04 17:22:00.00000081 |        16.5 | 2012-01-04 17:22:00 UTC |       -73.951300 |       40.774138 |        -73.990095 |        40.751048 |               1 |

</div>

```python
train_y.describe()
```

    count    23466.000000
    mean         8.472539
    std          4.609747
    min          0.010000
    25%          5.500000
    50%          7.500000
    75%         10.100000
    max        165.000000
    Name: fare_amount, dtype: float64

```python
import eli5
from eli5.sklearn import PermutationImportance

# Make a small change to the code below to use in this problem. 
perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)
```

<div class="table-wrapper" markdown="block">

|           Weight | Feature           |
|-----------------:|-------------------|
|  0.8426 ± 0.0168 | dropoff_latitude  |
|  0.8269 ± 0.0211 | pickup_latitude   |
|  0.5943 ± 0.0436 | pickup_longitude  |
|  0.5387 ± 0.0273 | dropoff_longitude |
| -0.0020 ± 0.0013 | passenger_count   |

</div>

Without detailed knowledge of New York City, it's difficult to rule out most hypotheses about why latitude features matter more than longitude.

A good next step is to disentangle the effect of being in certain parts of the city from the effect of total distance traveled.

The code below creates new features for longitudinal and latitudinal distance. It then builds a model that adds these new features to those you already had.


```python
# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
# Use a random_state of 1 for reproducible results that match the expected solution.
perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)
eli5.show_weights(perm2, feature_names = new_val_X.columns.tolist())
# show the weights for the permutation importance you just calculated
```

How would you interpret these importance scores? Distance traveled seems far more important than any location effects.

But the location still affects model predictions, and dropoff location now matters slightly more than pickup location. Do you have any hypotheses for why this might be?

A colleague observes that the values for *abs_lon_change* and *abs_lat_change* are pretty small (all values are between -0.1 and 0.1), whereas other variables have larger values. Do you think this could explain why those coordinates had larger permutation importance values in this case?

Consider an alternative where you created and used a feature that was 100X as large for these features, and used that larger feature for training and importance calculations. Would this change the outputted permutaiton importance values?

Why or why not?

The scale of features does not affect permutation importance per se. The only reason that rescaling a feature would affect PI is indirectly, if rescaling helped or hurt the ability of the particular learning method we're using to make use of that feature. That won't happen with tree based models, like the Random Forest used here. If you are familiar with Ridge Regression, you might be able to think of how that would be affected. That said, the absolute change features are have high importance because they capture total distance traveled, which is the primary determinant of taxi fares...It is not an artifact of the feature magnitude.

You've seen that the feature importance for latitudinal distance is greater than the importance of longitudinal distance. From this, can we conclude whether travelling a fixed latitudinal distance tends to be more expensive than traveling the same longitudinal distance?

We cannot tell from the permutation importance results whether traveling a fixed latitudinal distance is more or less expensive than traveling the same longitudinal distance. Possible reasons latitude feature are more important than longitude features 1. latitudinal distances in the dataset tend to be larger 2. it is more expensive to travel a fixed latitudinal distance 3. Both of the above If abs_lon_change values were very small, longitues could be less important to the model even if the cost per mile of travel in that direction were high.