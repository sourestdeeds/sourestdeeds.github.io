---
title: 'Mutual Information'
tags: [kaggle, feature engineering, mutual information]
layout: post
mathjax: true
categories: [Kaggle Notes]
published: true
---

First encountering a new dataset can sometimes feel overwhelming. You might be presented with hundreds or thousands of features without even a description to go by. Where do you even begin?

A great first step is to construct a ranking with a feature utility metric, a function measuring associations between a feature and the target. Then you can choose a smaller set of the most useful features to develop initially and have more confidence that your time will be well spent.

The metric we'll use is called "mutual information". Mutual information is a lot like correlation in that it measures a relationship between two quantities. The advantage of mutual information is that it can detect any kind of relationship, while correlation only detects linear relationships.

Mutual information is a great general-purpose metric and especially useful at the start of feature development when you might not know what model you'd like to use yet. It is:

- Easy to use and interpret.
- Computationally efficient.
- Theoretically well-founded.
- Resistant to overfitting.
- Able to detect any kind of relationship.

### What it Measures

Mutual information describes relationships in terms of uncertainty. The mutual information (MI) between two quantities is a measure of the extent to which knowledge of one quantity reduces uncertainty about the other. If you knew the value of a feature, how much more confident would you be about the target?

Here's an example from the Ames Housing data. The figure shows the relationship between the exterior quality of a house and the price it sold for. Each point represents a house.

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/1.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/1.png)
<center><b>Figure 1:</b> Knowing the exterior quality of a house reduces uncertainty about its sale price.</center><br> 

From the figure, we can see that knowing the value of *ExterQual* should make you more certain about the corresponding *SalePrice* -- each category of ExterQual tends to concentrate *SalePrice* to within a certain range. The mutual information that *ExterQual* has with *SalePrice* is the average reduction of uncertainty in *SalePrice* taken over the four values of ExterQual. Since Fair occurs less often than Typical, for instance, Fair gets less weight in the MI score.

(Technical note: What we're calling uncertainty is measured using a quantity from information theory known as "entropy". The entropy of a variable means roughly: "how many yes-or-no questions you would need to describe an occurance of that variable, on average." The more questions you have to ask, the more uncertain you must be about the variable. Mutual information is how many questions you expect the feature to answer about the target.)

### Interpreting Mutual Information Scores

The least possible mutual information between quantities is 0.0. When MI is zero, the quantities are independent: neither can tell you anything about the other. Conversely, in theory there's no upper bound to what MI can be. In practice though values above 2.0 or so are uncommon. (Mutual information is a logarithmic quantity, so it increases very slowly.)

The next figure will give you an idea of how MI values correspond to the kind and degree of association a feature has with the target.

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/1.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/1.png)
<center><b>Figure 2:</b> <b>Left</b>: Mutual information increases as the dependence between feature and target becomes tighter. <b>Right</b>: Mutual information can capture any kind of association (not just linear, like correlation).</center><br> 

Here are some things to remember when applying mutual information:

- MI can help you to understand the relative potential of a feature as a predictor of the target, considered by itself.
- It's possible for a feature to be very informative when interacting with other features, but not so informative all alone. MI can't detect interactions between features. It is a **univariate** metric.
- The actual usefulness of a feature depends on the model you use it with. A feature is only useful to the extent that its relationship with the target is one your model can learn. Just because a feature has a high MI score doesn't mean your model will be able to do anything with that information. You may need to transform the feature first to expose the association.

### Example - 1985 Automobiles

The [Automobile](https://www.kaggle.com/toramky/automobile-dataset) dataset consists of 193 cars from the 1985 model year. The goal for this dataset is to predict a car's price (the target) from 23 of the car's features, such as make, body_style, and horsepower. In this example, we'll rank the features with mutual information and investigate the results by data visualization.

This hidden cell imports some libraries and loads the dataset.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-whitegrid")

df = pd.read_csv("../input/fe-course-data/autos.csv")
df.head()
```



<div class="table-wrapper" markdown="block">

|   | symboling | make        | fuel_type | aspiration | num_of_doors | body_style  | drive_wheels | engine_location | wheel_base | length | ... | engine_size | fuel_system | bore | stroke | compression_ratio | horsepower | peak_rpm | city_mpg | highway_mpg | price |
|---|-----------|-------------|-----------|------------|--------------|-------------|--------------|-----------------|------------|--------|-----|-------------|-------------|------|--------|-------------------|------------|----------|----------|-------------|-------|
| 0 | 3         | alfa-romero | gas       | std        | 2            | convertible | rwd          | front           | 88.6       | 168.8  | ... | 130         | mpfi        | 3.47 | 2.68   | 9                 | 111        | 5000     | 21       | 27          | 13495 |
| 1 | 3         | alfa-romero | gas       | std        | 2            | convertible | rwd          | front           | 88.6       | 168.8  | ... | 130         | mpfi        | 3.47 | 2.68   | 9                 | 111        | 5000     | 21       | 27          | 16500 |
| 2 | 1         | alfa-romero | gas       | std        | 2            | hatchback   | rwd          | front           | 94.5       | 171.2  | ... | 152         | mpfi        | 2.68 | 3.47   | 9                 | 154        | 5000     | 19       | 26          | 16500 |
| 3 | 2         | audi        | gas       | std        | 4            | sedan       | fwd          | front           | 99.8       | 176.6  | ... | 109         | mpfi        | 3.19 | 3.40   | 10                | 102        | 5500     | 24       | 30          | 13950 |
| 4 | 2         | audi        | gas       | std        | 4            | sedan       | 4wd          | front           | 99.4       | 176.6  | ... | 136         | mpfi        | 3.19 | 3.40   | 8                 | 115        | 5500     | 18       | 22          | 17450 |

</div>

The scikit-learn algorithm for MI treats discrete features differently from continuous features. Consequently, you need to tell it which are which. As a rule of thumb, anything that must have a float dtype is not discrete. Categoricals (object or categorial dtype) can be treated as discrete by giving them a label encoding. 

```python
X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int
```

Scikit-learn has two mutual information metrics in its *feature_selection module*: one for real-valued targets (*mutual_info_regression*) and one for categorical targets (*mutual_info_classif*). Our target, *price*, is real-valued. The next cell computes the MI scores for our features and wraps them up in a nice dataframe.

```python
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores
```

    curb_weight          1.477695
    highway_mpg          0.953932
    length               0.621122
    fuel_system          0.484737
    stroke               0.383782
    num_of_cylinders     0.330589
    compression_ratio    0.133822
    fuel_type            0.048139
    Name: MI Scores, dtype: float64

And now a bar plot to make comparisions easier:

```python
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/3.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/3.png)
<center><b>Figure 3:</b> The high-scoring curb weight feature exhibits a strong relationship with price, the target.</center><br> 

Data visualization is a great follow-up to a utility ranking. Let's take a closer look at a couple of these.

As we might expect, the high-scoring *curb_weight* feature exhibits a strong relationship with *price*, the target.

```python
sns.relplot(x="curb_weight", y="price", data=df);
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/4.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/4.png)
<center><b>Figure 4:</b> Two price populations separates with different trends within the horsepower feature.</center><br>

The *fuel_type* feature has a fairly low MI score, but as we can see from the figure, it clearly separates two price populations with different trends within the *horsepower* feature. This indicates that *fuel_type* contributes an interaction effect and might not be unimportant after all. Before deciding a feature is unimportant from its MI score, it's good to investigate any possible interaction effects -- domain knowledge can offer a lot of guidance here.

```python
sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);5
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/5.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-01-mutual-information/5.png)
<center><b>Figure 5:</b> Domain knowledge for the fuel types of gas and diesel.</center><br>



This kind of target encoding is sometimes called a **mean encoding**. Applied to a binary target, it's also called **bin counting**. (Other names you might come across include: likelihood encoding, impact encoding, and leave-one-out encoding.)

### Smoothing

An encoding like this presents a couple of problems, however. First are unknown categories. Target encodings create a special risk of overfitting, which means they need to be trained on an independent "encoding" split. When you join the encoding to future splits, Pandas will fill in missing values for any categories not present in the encoding split. These missing values you would have to impute somehow.

Second are rare categories. When a category only occurs a few times in the dataset, any statistics calculated on its group are unlikely to be very accurate. In the Automobiles dataset, the mercurcy make only occurs once. The "mean" price we calculated is just the price of that one vehicle, which might not be very representative of any Mercuries we might see in the future. Target encoding rare categories can make overfitting more likely.

A solution to these problems is to add smoothing. The idea is to blend the in-category average with the overall average. Rare categories get less weight on their category average, while missing categories just get the overall average.

In pseudocode:

    encoding = weight * in_category + (1 - weight) * overall

where *weight* is a value between 0 and 1 calculated from the category frequency.

An easy way to determine the value for weight is to compute an **m-estimate**:

    weight = n / (n + m)

where *n* is the total number of times that category occurs in the data. The parameter m determines the "smoothing factor". Larger values of *m* put more weight on the overall estimate.

    

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-06-target-encoding/1.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-06-target-encoding/1.png)
<center><b>Figure 1:</b> M-Estimates comapred.</center><br>     

In the *Automobiles* dataset there are three cars with the make chevrolet. If you chose \\( m=2.0 \\), then the chevrolet category would be encoded with 60% of the average Chevrolet price plus 40% of the overall average price.    

    chevrolet = 0.6 * 6000.00 + 0.4 * 13285.03

When choosing a value for *m*, consider how noisy you expect the categories to be. Does the price of a vehicle vary a great deal within each make? Would you need a lot of data to get good estimates? If so, it could be better to choose a larger value for *m*; if the average price for each make were relatively stable, a smaller value could be okay.

> ### Use Cases for Target Encoding
> Target encoding is great for:
> - **High-cardinality features**: A feature with a large number of categories can be troublesome to encode: a one-hot encoding would generate too many features and alternatives, like a label encoding, might not be appropriate for that feature. A target encoding derives numbers for the categories using the feature's most important property: its relationship with the target.
> - **Domain-motivated features**: From prior experience, you might suspect that a categorical feature should be important even if it scored poorly with a feature metric. A target encoding can help reveal a feature's true informativeness.

### Example - MovieLens1M

The [MovieLens1M](https://www.kaggle.com/grouplens/movielens-20m-dataset) dataset contains one-million movie ratings by users of the MovieLens website, with features describing each user and movie. This hidden cell sets everything up:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
warnings.filterwarnings('ignore')


df = pd.read_csv("../input/fe-course-data/movielens1m.csv")
df = df.astype(np.uint8, errors='ignore') # reduce memory footprint
print("Number of Unique Zipcodes: {}".format(df["Zipcode"].nunique()))
```

    Number of Unique Zipcodes: 3439

With over 3000 categories, the *Zipcode* feature makes a good candidate for target encoding, and the size of this dataset (over one-million rows) means we can spare some data to create the encoding.

We'll start by creating a 25% split to train the target encoder.

```python
X = df.copy()
y = X.pop('Rating')

X_encode = X.sample(frac=0.25)
y_encode = y[X_encode.index]
X_pretrain = X.drop(X_encode.index)
y_train = y[X_pretrain.index]
```

The *category_encoders* package in scikit-learn-contrib implements an *m-estimate* encoder, which we'll use to encode our *Zipcode* feature.

```python
from category_encoders import MEstimateEncoder

# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)

# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)
```

Let's compare the encoded values to the target to see how informative our encoding might be.

```python
plt.figure(dpi=90)
ax = sns.distplot(y, kde=False, norm_hist=True)
ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
ax.set_xlabel("Rating")
ax.legend(labels=['Zipcode', 'Rating']);
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-06-target-encoding/2.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-06-target-encoding/2.png)
<center><b>Figure 2:</b> Movie-watchers differed enough in their ratings from zipcode to zipcode that our target encoding was able to capture useful information.</center><br>  