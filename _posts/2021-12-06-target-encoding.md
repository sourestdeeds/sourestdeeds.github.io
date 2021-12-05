---
title: 'Target Encoding'
tags: [kaggle, feature engineering, target encoding, encoding]
layout: post
mathjax: true
categories: [Kaggle Notes]
published: false
---

Most of the techniques we've seen in this course have been for numerical features. The technique we'll look at in this lesson, target encoding, is instead meant for categorical features. It's a method of encoding categories as numbers, like one-hot or label encoding, with the difference that it also uses the target to create the encoding. This makes it what we call a supervised feature engineering technique.


A **target encoding** is any kind of encoding that replaces a feature's categories with some number derived from the target.

A simple and effective version is to apply a group aggregation from Lesson 3, like the mean. Using the Automobiles dataset, this computes the average price of each vehicle's make:

```python
import pandas as pd

autos = pd.read_csv("../input/fe-course-data/autos.csv")
autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")

autos[["make", "price", "make_encoded"]].head(10)
```

<div class="table-wrapper" markdown="block">

|     | make        | price | make_encoded | horsepower |
|-----|-------------|-------|--------------|------------|
| 0   | alfa-romero | 13495 | 15498.333333 | 2756       |
| 1   | alfa-romero | 16500 | 15498.333333 | 2756       |
| 2   | alfa-romero | 16500 | 15498.333333 | 2800       |
| 3   | audi        | 13950 | 17859.166667 | 3950       |
| 4   | audi        | 17450 | 17859.166667 | 3139       |
| 5   | audi        | 15250 | 17859.166667 | ...        |
| 6   | audi        | 17710 | 17859.166667 | 3750       |
| 7   | audi        | 18920 | 17859.166667 | 3770       |
| 8   | audi        | 23875 | 17859.166667 | 3430       |
| 9   | bmw         | 16430 | 26118.750000 | 3485       |
| 143 | toyota      | wagon | 62           | 3110       |

</div>

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


Often, we can give names to these axes of variation. The longer axis we might call the "Size" component: small height and small diameter (lower left) contrasted with large height and large diameter (upper right). The shorter axis we might call the "Shape" component: small height and large diameter (flat shape) contrasted with large height and small diameter (round shape).

Notice that instead of describing abalones by their 'Height' and 'Diameter', we could just as well describe them by their 'Size' and 'Shape'. This, in fact, is the whole idea of PCA: instead of describing the data with the original features, we describe it with its axes of variation. The axes of variation become the new features.

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-05-principle-component-analysis/2.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-05-principle-component-analysis/2.png)
<center><b>Figure 2:</b> The principal components become the new features by a rotation of the dataset in the feature space.</center><br>     

The new features PCA constructs are actually just linear combinations (weighted sums) of the original features:

    df["Size"] = 0.707 * X["Height"] + 0.707 * X["Diameter"]
    df["Shape"] = 0.707 * X["Height"] - 0.707 * X["Diameter"]

These new features are called the principal components of the data. The weights themselves are called loadings. There will be as many principal components as there are features in the original dataset: if we had used ten features instead of two, we would have ended up with ten components.

A component's loadings tell us what variation it expresses through signs and magnitudes:


This table of loadings is telling us that in the *Size* component, *Height* and *Diameter* vary in the same direction (same sign), but in the Shape component they vary in opposite directions (opposite sign). In each component, the loadings are all of the same magnitude and so the features contribute equally in both.

PCA also tells us the *amount* of variation in each component. We can see from the figures that there is more variation in the data along the *Size* component than along the Shape component. PCA makes this precise through each component's **percent of explained variance**.


[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-05-principle-component-analysis/3.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-05-principle-component-analysis/3.png)
<center><b>Figure 3:</b> Size accounts for about 96% and the Shape for about 4% of the variance between Height and Diameter.</center><br>    



### PCA for Feature Engineering

There are two ways you could use PCA for feature engineering.

The first way is to use it as a descriptive technique. Since the components tell you about the variation, you could compute the MI scores for the components and see what kind of variation is most predictive of your target. That could give you ideas for kinds of features to create -- a product of 'Height' and 'Diameter' if 'Size' is important, say, or a ratio of 'Height' and 'Diameter' if Shape is important. You could even try clustering on one or more of the high-scoring components.

The second way is to use the components themselves as features. Because the components expose the variational structure of the data directly, they can often be more informative than the original features. Here are some use-cases:

- **Dimensionality reduction**: When your features are highly redundant (*multicollinear*, specifically), PCA will partition out the redundancy into one or more near-zero variance components, which you can then drop since they will contain little or no information.
- **Anomaly detection**: Unusual variation, not apparent from the original features, will often show up in the low-variance components. These components could be highly informative in an anomaly or outlier detection task.
- **Noise reduction**: A collection of sensor readings will often share some common background noise. PCA can sometimes collect the (informative) signal into a smaller number of features while leaving the noise alone, thus boosting the signal-to-noise ratio.
- **Decorrelation**: Some ML algorithms struggle with highly-correlated features. PCA transforms correlated features into uncorrelated components, which could be easier for your algorithm to work with.

PCA basically gives you direct access to the correlational structure of your data. You'll no doubt come up with applications of your own!

> PCA Best Practices
> There are a few things to keep in mind when applying PCA:
> - PCA only works with numeric features, like continuous quantities or counts.
> - PCA is sensitive to scale. It's good practice to standardize your data before applying PCA, unless you know you have good reason not to.
> - Consider removing or constraining outliers, since they can an have an undue influence on the results.


### Example - 1985 Automobiles

In this example, we'll return to our [Automobile](https://www.kaggle.com/toramky/automobile-dataset) dataset and apply PCA, using it as a descriptive technique to discover features. We'll look at other use-cases in the exercise.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression


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


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


df = pd.read_csv("../input/fe-course-data/autos.csv")
```

We've selected four features that cover a range of properties. Each of these features also has a high MI score with the target, price. We'll standardize the data since these features aren't naturally on the same scale.

```python
features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]

X = df.copy()
y = X.pop('price')
X = X.loc[:, features]

# Standardize
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

Now we can fit scikit-learn's *PCA* estimator and create the principal components. You can see here the first few rows of the transformed dataset.

```python
from sklearn.decomposition import PCA

# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.head()
```

<div class="table-wrapper" markdown="block">

|   | PC1       | PC2       | PC3       | PC4       |
|---|-----------|-----------|-----------|-----------|
| 0 | 0.382486  | -0.400222 | 0.124122  | 0.169539  |
| 1 | 0.382486  | -0.400222 | 0.124122  | 0.169539  |
| 2 | 1.550890  | -0.107175 | 0.598361  | -0.256081 |
| 3 | -0.408859 | -0.425947 | 0.243335  | 0.013920  |
| 4 | 1.132749  | -0.814565 | -0.202885 | 0.224138  |

</div>

After fitting, the *PCA* instance contains the loadings in its *components_ attribute*. (Terminology for PCA is inconsistent, unfortunately. We're following the convention that calls the transformed columns in *X_pca* the components, which otherwise don't have a name.) We'll wrap the loadings up in a dataframe.

```python
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
loadings
```

<div class="table-wrapper" markdown="block">

|             | PC1       | PC2      | PC3       | PC4       |
|-------------|-----------|----------|-----------|-----------|
| highway_mpg | -0.492347 | 0.770892 | 0.070142  | -0.397996 |
| engine_size | 0.503859  | 0.626709 | 0.019960  | 0.594107  |
| horsepower  | 0.500448  | 0.013788 | 0.731093  | -0.463534 |
| curb_weight | 0.503262  | 0.113008 | -0.678369 | -0.523232 |

</div>

Recall that the signs and magnitudes of a component's loadings tell us what kind of variation it's captured. The first component (PC1) shows a contrast between large, powerful vehicles with poor gas milage, and smaller, more economical vehicles with good gas milage. We might call this the "Luxury/Economy" axis. The next figure shows that our four chosen features mostly vary along the Luxury/Economy axis.

```python
# Look at explained variance
plot_variance(pca);
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-05-principle-component-analysis/4.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-05-principle-component-analysis/4.png)
<center><b>Figure 4:</b> The four chosen features mostly vary along the Luxury/Economy axis.</center><br>   

Let's also look at the MI scores of the components. Not surprisingly, PC1 is highly informative, though the remaining components, despite their small variance, still have a significant relationship with price. Examining those components could be worthwhile to find relationships not captured by the main Luxury/Economy axis.

```python
mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
mi_scores
```

    PC1    1.013800
    PC2    0.379440
    PC3    0.306502
    PC4    0.204447
    Name: MI Scores, dtype: float64

The third component shows a contrast between *horsepower* and *curb_weight* -- sports cars vs. wagons, it seems.

```python
# Show dataframe sorted by PC3
idx = X_pca["PC3"].sort_values(ascending=False).index
cols = ["make", "body_style", "horsepower", "curb_weight"]
df.loc[idx, cols]
```

<div class="table-wrapper" markdown="block">

|     | make          | body_style  | horsepower | curb_weight |
|-----|---------------|-------------|------------|-------------|
| 118 | porsche       | hardtop     | 207        | 2756        |
| 117 | porsche       | hardtop     | 207        | 2756        |
| 119 | porsche       | convertible | 207        | 2800        |
| 45  | jaguar        | sedan       | 262        | 3950        |
| 96  | nissan        | hatchback   | 200        | 3139        |
| ... | ...           | ...         | ...        | ...         |
| 59  | mercedes-benz | wagon       | 123        | 3750        |
| 61  | mercedes-benz | sedan       | 123        | 3770        |
| 101 | peugot        | wagon       | 95         | 3430        |
| 105 | peugot        | wagon       | 95         | 3485        |
| 143 | toyota        | wagon       | 62         | 3110        |

</div>

```python
df["sports_or_wagon"] = X.curb_weight / X.horsepower
sns.regplot(x="sports_or_wagon", y='price', data=df, order=2);
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-05-principle-component-analysis/5.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-05-principle-component-analysis/5.png)
<center><b>Figure 5:</b> The third component shows a contrast between sports cars or wagons.</center><br>   