---
title: 'SHAP Values'
tags: [kaggle, SHAP, SHAP values]
layout: post
mathjax: true
categories: [Machine Learning Explainability]
published: false
---

You've seen (and used) techniques to extract general insights from a machine learning model. But what if you want to break down how the model works for an individual prediction?

SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the impact of each feature. Where could you use this?

- A model says a bank shouldn't loan someone money, and the bank is legally required to explain the basis for each loan rejection
- A healthcare provider wants to identify what factors are driving each patient's risk of some disease so they can directly address those risk factors with targeted health interventions



### How They Work

SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.

In these tutorials, we predicted whether a team would have a player win the Man of the Match award.

We could ask:

- How much was a prediction driven by the fact that the team scored 3 goals?

But it's easier to give a concrete, numeric answer if we restate this as:

- How much was a prediction driven by the fact that the team scored 3 goals, **instead of some baseline number of goals**.

Of course, each team has many features. So if we answer this question for *number of goals*, we could repeat the process for all other features.

SHAP values do this in a way that guarantees a nice property. Specifically, you decompose a prediction with the following equation:

```python
sum(SHAP values for all features) = pred_for_team - pred_for_baseline_values
```

That is, the SHAP values of all features sum up to explain why my prediction was different from the baseline. This allows us to decompose a prediction in a graph like this:


[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/1.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/1.png)<br>



How do you interpret this?

We predicted 0.7, whereas the base_value is 0.4979. Feature values causing increased predictions are in pink, and their visual size shows the magnitude of the feature's effect. Feature values decreasing the prediction are in blue. The biggest impact comes from *Goal Scored* being 2. Though the ball possession value has a meaningful effect decreasing the prediction.

If you subtract the length of the blue bars from the length of the pink bars, it equals the distance from the base value to the output.

There is some complexity to the technique, to ensure that the baseline plus the sum of individual effects adds up to the prediction (which isn't as straightforward as it sounds). We won't go into that detail here, since it isn't critical for using the technique. This [blog post](https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d) has a longer theoretical explanation.

### Code to Calculate SHAP Values

We calculate SHAP values using the wonderful [Shap](https://github.com/slundberg/shap) library.

For this example, we'll reuse the model you've already seen with the Soccer data.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
```

We will look at SHAP values for a single row of the dataset (we arbitrarily chose row 5). For context, we'll look at the raw predictions before looking at the SHAP values.

```python
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


my_model.predict_proba(data_for_prediction_array)
```

    array([[0.29, 0.71]])

The team is 70% likely to have a player win the award.

Now, we'll move onto the code to get SHAP values for that single prediction.

```python
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
```

The *shap_values* object above is a list with two arrays. The first array is the SHAP values for a negative outcome (don't win the award), and the second array is the list of SHAP values for the positive outcome (wins the award). We typically think about predictions in terms of the prediction of a positive outcome, so we'll pull out SHAP values for positive outcomes (pulling out *shap_values[1]*).

It's cumbersome to review raw arrays, but the shap package has a nice way to visualize the results.

```python
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
```

<svg data-reactroot="" style="user-select: none; display: block; font-family: arial; height: 150px; width: 234px;"><style>
          .force-bar-axis path {
            fill: none;
            opacity: 0.4;
          }
          .force-bar-axis paths {
            display: none;
          }
          .tick line {
            stroke: #000;
            stroke-width: 1px;
            opacity: 0.4;
          }
          .tick text {
            fill: #000;
            opacity: 0.5;
            font-size: 12px;
            padding: 0px;
          }</style><g><g transform="translate(0,50)" class="force-bar-axis" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><path class="domain" stroke="#000" d="M0.5,0V0.5H234.5V0"></path><g class="tick" opacity="1" transform="translate(9.554841356374371,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.1013</text></g><g class="tick" opacity="1" transform="translate(36.416131017280776,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.2013</text></g><g class="tick" opacity="1" transform="translate(63.27742067818719,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.3013</text></g><g class="tick" opacity="1" transform="translate(90.1387103390936,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.4013</text></g><g class="tick" opacity="1" transform="translate(117,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.5013</text></g><g class="tick" opacity="1" transform="translate(143.8612896609064,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.6013</text></g><g class="tick" opacity="1" transform="translate(170.72257932181282,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.7013</text></g><g class="tick" opacity="1" transform="translate(197.58386898271922,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.8013</text></g><g class="tick" opacity="1" transform="translate(224.44515864362563,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.9013</text></g></g><path class="force-bar-blocks" d="M95.07294216714212,56L95.18595521058761,56L99.18595521058761,64.5L95.18595521058761,73L95.07294216714212,73L99.07294216714212,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M95.18595521058761,56L95.40785531999975,56L99.40785531999975,64.5L95.40785531999975,73L95.18595521058761,73L99.18595521058761,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M95.40785531999975,56L97.0239715221681,56L101.0239715221681,64.5L97.0239715221681,73L95.40785531999975,73L99.40785531999975,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M97.0239715221681,56L98.75043042729867,56L102.75043042729867,64.5L98.75043042729867,73L97.0239715221681,73L101.0239715221681,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M98.75043042729865,56L100.87843902202705,56L104.87843902202705,64.5L100.87843902202705,73L98.75043042729865,73L102.75043042729865,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M100.87843902202705,56L103.64978415297165,56L107.64978415297165,64.5L103.64978415297165,73L100.87843902202705,73L104.87843902202705,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M103.64978415297166,56L107.03085330228731,56L111.03085330228731,64.5L107.03085330228731,73L103.64978415297166,73L107.64978415297166,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M107.0308533022873,56L110.48526728194261,56L114.48526728194261,64.5L110.48526728194261,73L107.0308533022873,73L111.0308533022873,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M110.48526728194261,56L116.60420788098106,56L120.60420788098106,64.5L116.60420788098106,73L110.48526728194261,73L114.48526728194261,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M116.60420788098105,56L124.42239225659452,56L128.42239225659452,64.5L124.42239225659452,73L116.60420788098105,73L120.60420788098105,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M124.42239225659453,56L132.43570430625272,56L136.43570430625272,64.5L132.43570430625272,73L124.42239225659453,73L128.42239225659455,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M132.43570430625272,56L145.45391781974917,56L149.45391781974917,64.5L145.45391781974917,73L132.43570430625272,73L136.43570430625272,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M145.45391781974917,56L173.07294216714212,56L173.07294216714212,64.5L173.07294216714212,73L145.45391781974917,73L149.45391781974917,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M173.07294216714212,56L185.80644866267002,56L181.80644866267002,64.5L185.80644866267002,73L173.07294216714212,73L173.07294216714212,64.5" fill="rgb(30, 136, 229)"></path><path class="force-bar-blocks" d="M185.80644866267002,56L191.34283259424308,56L187.34283259424308,64.5L191.34283259424308,73L185.80644866267002,73L181.80644866267002,64.5" fill="rgb(30, 136, 229)"></path><path class="force-bar-blocks" d="M191.34283259424308,56L193.40278733612794,56L189.40278733612794,64.5L193.40278733612794,73L191.34283259424308,73L187.34283259424308,64.5" fill="rgb(30, 136, 229)"></path><path class="force-bar-blocks" d="M193.40278733612791,56L194.74594801591843,56L190.74594801591843,64.5L194.74594801591843,73L193.40278733612791,73L189.40278733612791,64.5" fill="rgb(30, 136, 229)"></path><path class="force-bar-blocks" d="M194.7459480159185,56L195.00000000000003,56L191.00000000000003,64.5L195.00000000000003,73L194.7459480159185,73L190.7459480159185,64.5" fill="rgb(30, 136, 229)"></path><path class="force-bar-labelBacking" stroke="none" opacity="0.2" d="M145.45391781974917,73L76.01328793130227,83L76.01328793130227,104L-51.70688113608054,104L-51.70688113608054,83L132.43570430625272,73" fill="url(#linear-backgrad-0)"></path><path class="force-bar-labelBacking" stroke="none" opacity="0.2" d="M173.07294216714212,73L173.07294216714212,83L173.07294216714212,104L76.01328793130227,104L76.01328793130227,83L145.45391781974917,73" fill="url(#linear-backgrad-0)"></path><path class="force-bar-labelBacking" stroke="none" opacity="0.2" d="M185.80644866267002,73L308.81157436929055,83L308.81157436929055,104L173.07294216714212,104L173.07294216714212,83L173.07294216714212,73" fill="url(#linear-backgrad-1)"></path><rect class="force-bar-labelDividers" height="21px" width="1px" y="83" x="75.51328793130227" fill="url(#linear-grad-0)"></rect><rect class="force-bar-labelDividers" height="21px" width="1px" y="83" x="172.57294216714212" fill="url(#linear-grad-0)"></rect><line class="force-bar-labelLinks" y1="73" y2="83" stroke-opacity="0.5" stroke-width="1" x1="145.45391781974917" x2="76.01328793130227" stroke="rgb(255, 13, 87)"></line><line class="force-bar-labelLinks" y1="73" y2="83" stroke-opacity="0.5" stroke-width="1" x1="173.07294216714212" x2="173.07294216714212" stroke="rgb(255, 13, 87)"></line><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M95.18595521058761,56L99.18595521058761,64.5L95.18595521058761,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M95.40785531999975,56L99.40785531999975,64.5L95.40785531999975,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M97.0239715221681,56L101.0239715221681,64.5L97.0239715221681,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M98.75043042729865,56L102.75043042729865,64.5L98.75043042729865,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M100.87843902202705,56L104.87843902202705,64.5L100.87843902202705,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M103.64978415297166,56L107.64978415297166,64.5L103.64978415297166,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M107.0308533022873,56L111.0308533022873,64.5L107.0308533022873,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M110.48526728194261,56L114.48526728194261,64.5L110.48526728194261,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M116.60420788098105,56L120.60420788098105,64.5L116.60420788098105,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M124.42239225659452,56L128.42239225659452,64.5L124.42239225659452,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M132.43570430625272,56L136.43570430625272,64.5L132.43570430625272,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M145.45391781974917,56L149.45391781974917,64.5L145.45391781974917,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M173.07294216714212,56L177.07294216714212,64.5L173.07294216714212,73" stroke="#rgba(0,0,0,0)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M185.80644866267002,56L181.80644866267002,64.5L185.80644866267002,73" stroke="rgb(209, 230, 250)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M191.34283259424308,56L187.34283259424308,64.5L191.34283259424308,73" stroke="rgb(209, 230, 250)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M193.40278733612791,56L189.40278733612791,64.5L193.40278733612791,73" stroke="rgb(209, 230, 250)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M194.74594801591846,56L190.74594801591846,64.5L194.74594801591846,73" stroke="rgb(209, 230, 250)"></path></g><g><text class="force-bar-labels" font-size="12px" y="98" fill="rgb(255, 13, 87)" stroke="none" x="12.153203397610866" text-anchor="middle">Fouls Committed = 25</text><text class="force-bar-labels" font-size="12px" y="98" fill="rgb(255, 13, 87)" stroke="none" x="124.5431150492222" text-anchor="middle">Goal Scored = 2</text><text class="force-bar-labels" font-size="12px" y="98" fill="rgb(30, 136, 229)" stroke="none" x="240.94225826821634" text-anchor="middle">Ball Possession % = 38</text></g><text x="117" y="28" text-anchor="middle" font-size="12" fill="#000" opacity="0.5">base value</text><line x1="173.07294216714212" x2="173.07294216714212" y1="50" y2="56" stroke="#F2F2F2" stroke-width="1" opacity="1"></line><text x="173.07294216714212" y="45" color="#fff" text-anchor="middle" font-weight="bold" stroke="#fff" stroke-width="6" opacity="1">0.71</text><text x="173.07294216714212" y="45" text-anchor="middle" font-weight="bold" fill="#000" opacity="1">0.71</text><text x="157.07294216714212" y="12" text-anchor="end" font-size="13" fill="rgb(255, 13, 87)" opacity="1">higher</text><text x="180.07294216714212" y="8" text-anchor="end" font-size="13" fill="rgb(255, 13, 87)" opacity="1">→</text><text x="173.07294216714212" y="28" text-anchor="middle" font-size="12" fill="#000" opacity="0.5">f(x)</text><text x="166.07294216714212" y="14" text-anchor="start" font-size="13" fill="rgb(30, 136, 229)" opacity="1">←</text><text x="189.07294216714212" y="12" text-anchor="start" font-size="13" fill="rgb(30, 136, 229)" opacity="1">lower</text><text x="10" y="20" text-anchor="middle" font-size="12" stroke="#fff" fill="#fff" stroke-width="4" stroke-linejoin="round"></text><text x="10" y="20" text-anchor="middle" font-size="12" fill="#0f0"></text><linearGradient id="linear-grad-0" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="rgb(255, 13, 87)" stop-opacity="0.6"></stop><stop offset="100%" stop-color="rgb(255, 13, 87)" stop-opacity="0"></stop></linearGradient><linearGradient id="linear-backgrad-0" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="rgb(255, 13, 87)" stop-opacity="0.5"></stop><stop offset="100%" stop-color="rgb(255, 13, 87)" stop-opacity="0"></stop></linearGradient><linearGradient id="linear-grad-1" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="rgb(30, 136, 229)" stop-opacity="0.6"></stop><stop offset="100%" stop-color="rgb(30, 136, 229)" stop-opacity="0"></stop></linearGradient><linearGradient id="linear-backgrad-1" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="rgb(30, 136, 229)" stop-opacity="0.5"></stop><stop offset="100%" stop-color="rgb(30, 136, 229)" stop-opacity="0"></stop></linearGradient></svg>



If you look carefully at the code where we created the SHAP values, you'll notice we reference Trees in `shap.TreeExplainer(my_model)`{:.language-python .highlight}. But the SHAP package has explainers for every type of model.

- `shap.DeepExplainer`{:.language-python .highlight} works with Deep Learning models.
- `shap.KernelExplainer`{:.language-python .highlight} works with all models, though it is slower than other Explainers and it offers an approximation rather than exact Shap values.
Here is an example using `KernelExplainer`{:.language-python .highlight} to get similar results. The results aren't identical because KernelExplainer gives an approximate result. But the results tell the same story.

```python
# use Kernel SHAP to explain test set predictions
k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
```

<svg data-reactroot="" style="user-select: none; display: block; font-family: arial; height: 150px; width: 234px;"><style>
          .force-bar-axis path {
            fill: none;
            opacity: 0.4;
          }
          .force-bar-axis paths {
            display: none;
          }
          .tick line {
            stroke: #000;
            stroke-width: 1px;
            opacity: 0.4;
          }
          .tick text {
            fill: #000;
            opacity: 0.5;
            font-size: 12px;
            padding: 0px;
          }</style><g><g transform="translate(0,50)" class="force-bar-axis" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><path class="domain" stroke="#000" d="M0.5,0V0.5H234.5V0"></path><g class="tick" opacity="1" transform="translate(10.719032236391035,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.09333</text></g><g class="tick" opacity="1" transform="translate(37.289274177293265,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.1933</text></g><g class="tick" opacity="1" transform="translate(63.85951611819552,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.2933</text></g><g class="tick" opacity="1" transform="translate(90.42975805909776,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.3933</text></g><g class="tick" opacity="1" transform="translate(117,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.4933</text></g><g class="tick" opacity="1" transform="translate(143.57024194090224,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.5933</text></g><g class="tick" opacity="1" transform="translate(170.1404838818045,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.6933</text></g><g class="tick" opacity="1" transform="translate(196.71072582270673,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.7933</text></g><g class="tick" opacity="1" transform="translate(223.28096776360897,0)"><line stroke="#000" y2="4" x1="0.5" x2="0.5"></line><text fill="#000" y="-14" x="0.5" dy="0.71em">0.8933</text></g></g><path class="force-bar-blocks" d="M96.56885753862156,56L97.65525185720489,56L101.65525185720489,64.5L97.65525185720489,73L96.56885753862156,73L100.56885753862156,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M97.65525185720489,56L99.15079267898479,56L103.15079267898479,64.5L99.15079267898479,73L97.65525185720489,73L101.65525185720489,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M99.15079267898479,56L100.65554515851197,56L104.65554515851197,64.5L100.65554515851197,73L99.15079267898479,73L103.15079267898479,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M100.65554515851197,56L102.44087681148554,56L106.44087681148554,64.5L102.44087681148554,73L100.65554515851197,73L104.65554515851197,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M102.44087681148554,56L104.85923340682508,56L108.85923340682508,64.5L104.85923340682508,73L102.44087681148554,73L106.44087681148554,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M104.85923340682508,56L107.65065555277971,56L111.65065555277971,64.5L107.65065555277971,73L104.85923340682508,73L108.85923340682508,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M107.65065555277971,56L111.10932463704695,56L115.10932463704695,64.5L111.10932463704695,73L107.65065555277971,73L111.65065555277971,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M111.10932463704695,56L118.57405618908973,56L122.57405618908973,64.5L118.57405618908973,73L111.10932463704695,73L115.10932463704695,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M118.57405618908973,56L126.13760361536539,56L130.13760361536538,64.5L126.13760361536539,73L118.57405618908973,73L122.57405618908973,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M126.13760361536539,56L134.06172643272328,56L138.06172643272328,64.5L134.06172643272328,73L126.13760361536539,73L130.13760361536538,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M134.06172643272328,56L146.71576026998864,56L150.71576026998864,64.5L146.71576026998864,73L134.06172643272328,73L138.06172643272328,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M146.71576026998864,56L174.56885753862156,56L174.56885753862156,64.5L174.56885753862156,73L146.71576026998864,73L150.71576026998864,64.5" fill="rgb(255, 13, 87)"></path><path class="force-bar-blocks" d="M174.56885753862156,56L186.98103804192962,56L182.98103804192962,64.5L186.98103804192962,73L174.56885753862156,73L174.56885753862156,64.5" fill="rgb(30, 136, 229)"></path><path class="force-bar-blocks" d="M186.98103804192962,56L192.472996878057,56L188.472996878057,64.5L192.472996878057,73L186.98103804192962,73L182.98103804192962,64.5" fill="rgb(30, 136, 229)"></path><path class="force-bar-blocks" d="M192.47299687805702,56L194.27279065727473,56L190.27279065727473,64.5L194.27279065727473,73L192.47299687805702,73L188.47299687805702,64.5" fill="rgb(30, 136, 229)"></path><path class="force-bar-blocks" d="M194.2727906572747,56L195,56L191,64.5L195,73L194.2727906572747,73L190.2727906572747,64.5" fill="rgb(30, 136, 229)"></path><path class="force-bar-labelBacking" stroke="none" opacity="0.2" d="M146.7157602699886,73L77.50920330278171,83L77.50920330278171,104L-50.2109657646011,104L-50.2109657646011,83L134.06172643272328,73" fill="url(#linear-backgrad-0)"></path><path class="force-bar-labelBacking" stroke="none" opacity="0.2" d="M174.56885753862156,73L174.56885753862156,83L174.56885753862156,104L77.50920330278171,104L77.50920330278171,83L146.71576026998864,73" fill="url(#linear-backgrad-0)"></path><path class="force-bar-labelBacking" stroke="none" opacity="0.2" d="M186.98103804192962,73L310.30748974077,83L310.30748974077,104L174.56885753862156,104L174.56885753862156,83L174.56885753862156,73" fill="url(#linear-backgrad-1)"></path><rect class="force-bar-labelDividers" height="21px" width="1px" y="83" x="77.00920330278171" fill="url(#linear-grad-0)"></rect><rect class="force-bar-labelDividers" height="21px" width="1px" y="83" x="174.06885753862156" fill="url(#linear-grad-0)"></rect><line class="force-bar-labelLinks" y1="73" y2="83" stroke-opacity="0.5" stroke-width="1" x1="146.7157602699886" x2="77.50920330278171" stroke="rgb(255, 13, 87)"></line><line class="force-bar-labelLinks" y1="73" y2="83" stroke-opacity="0.5" stroke-width="1" x1="174.56885753862156" x2="174.56885753862156" stroke="rgb(255, 13, 87)"></line><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M97.65525185720489,56L101.65525185720489,64.5L97.65525185720489,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M99.15079267898479,56L103.15079267898479,64.5L99.15079267898479,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M100.65554515851197,56L104.65554515851197,64.5L100.65554515851197,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M102.44087681148554,56L106.44087681148554,64.5L102.44087681148554,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M104.85923340682508,56L108.85923340682508,64.5L104.85923340682508,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M107.65065555277971,56L111.65065555277971,64.5L107.65065555277971,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M111.10932463704695,56L115.10932463704695,64.5L111.10932463704695,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M118.57405618908973,56L122.57405618908973,64.5L118.57405618908973,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M126.13760361536539,56L130.13760361536538,64.5L126.13760361536539,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M134.06172643272328,56L138.06172643272328,64.5L134.06172643272328,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M146.7157602699886,56L150.7157602699886,64.5L146.7157602699886,73" stroke="rgb(255, 195, 213)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M174.56885753862156,56L178.56885753862156,64.5L174.56885753862156,73" stroke="#rgba(0,0,0,0)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M186.98103804192962,56L182.98103804192962,64.5L186.98103804192962,73" stroke="rgb(209, 230, 250)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M192.47299687805702,56L188.47299687805702,64.5L192.47299687805702,73" stroke="rgb(209, 230, 250)"></path><path class="force-bar-blockDividers" stroke-width="2" fill="none" d="M194.2727906572747,56L190.2727906572747,64.5L194.2727906572747,73" stroke="rgb(209, 230, 250)"></path></g><g><text class="force-bar-labels" font-size="12px" y="98" fill="rgb(255, 13, 87)" stroke="none" x="13.649118769090308" text-anchor="middle">Fouls Committed = 25</text><text class="force-bar-labels" font-size="12px" y="98" fill="rgb(255, 13, 87)" stroke="none" x="126.03903042070164" text-anchor="middle">Goal Scored = 2</text><text class="force-bar-labels" font-size="12px" y="98" fill="rgb(30, 136, 229)" stroke="none" x="242.43817363969578" text-anchor="middle">Ball Possession % = 38</text></g><text x="117" y="28" text-anchor="middle" font-size="12" fill="#000" opacity="0.5">base value</text><line x1="174.56885753862156" x2="174.56885753862156" y1="50" y2="56" stroke="#F2F2F2" stroke-width="1" opacity="1"></line><text x="174.56885753862156" y="45" color="#fff" text-anchor="middle" font-weight="bold" stroke="#fff" stroke-width="6" opacity="1">0.71</text><text x="174.56885753862156" y="45" text-anchor="middle" font-weight="bold" fill="#000" opacity="1">0.71</text><text x="158.56885753862156" y="12" text-anchor="end" font-size="13" fill="rgb(255, 13, 87)" opacity="1">higher</text><text x="181.56885753862156" y="8" text-anchor="end" font-size="13" fill="rgb(255, 13, 87)" opacity="1">→</text><text x="174.56885753862156" y="28" text-anchor="middle" font-size="12" fill="#000" opacity="0.5">f(x)</text><text x="167.56885753862156" y="14" text-anchor="start" font-size="13" fill="rgb(30, 136, 229)" opacity="1">←</text><text x="190.56885753862156" y="12" text-anchor="start" font-size="13" fill="rgb(30, 136, 229)" opacity="1">lower</text><text x="10" y="20" text-anchor="middle" font-size="12" stroke="#fff" fill="#fff" stroke-width="4" stroke-linejoin="round" opacity="0"></text><text x="10" y="20" text-anchor="middle" font-size="12" fill="#0f0" opacity="0"></text><linearGradient id="linear-grad-0" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="rgb(255, 13, 87)" stop-opacity="0.6"></stop><stop offset="100%" stop-color="rgb(255, 13, 87)" stop-opacity="0"></stop></linearGradient><linearGradient id="linear-backgrad-0" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="rgb(255, 13, 87)" stop-opacity="0.5"></stop><stop offset="100%" stop-color="rgb(255, 13, 87)" stop-opacity="0"></stop></linearGradient><linearGradient id="linear-grad-1" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="rgb(30, 136, 229)" stop-opacity="0.6"></stop><stop offset="100%" stop-color="rgb(30, 136, 229)" stop-opacity="0"></stop></linearGradient><linearGradient id="linear-backgrad-1" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="rgb(30, 136, 229)" stop-opacity="0.5"></stop><stop offset="100%" stop-color="rgb(30, 136, 229)" stop-opacity="0"></stop></linearGradient></svg>


### Example Scenario

A hospital has struggled with "readmissions," where they release a patient before the patient has recovered enough, and the patient returns with health complications.

The hospital wants your help identifying patients at highest risk of being readmitted. Doctors (rather than your model) will make the final decision about when to release each patient; but they hope your model will highlight issues the doctors should consider when releasing a patient.

The hospital has given you relevant patient medical information. Here is a list of columns in the data:

```python
import pandas as pd
data = pd.read_csv('../input/hospital-readmissions/train.csv')
data.columns
```
```
Index(['time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses', 'race_Caucasian',
       'race_AfricanAmerican', 'gender_Female', 'age_[70-80)', 'age_[60-70)',
       'age_[50-60)', 'age_[80-90)', 'age_[40-50)', 'payer_code_?',
       'payer_code_MC', 'payer_code_HM', 'payer_code_SP', 'payer_code_BC',
       'medical_specialty_?', 'medical_specialty_InternalMedicine',
       'medical_specialty_Emergency/Trauma',
       'medical_specialty_Family/GeneralPractice',
       'medical_specialty_Cardiology', 'diag_1_428', 'diag_1_414',
       'diag_1_786', 'diag_2_276', 'diag_2_428', 'diag_2_250', 'diag_2_427',
       'diag_3_250', 'diag_3_401', 'diag_3_276', 'diag_3_428',
       'max_glu_serum_None', 'A1Cresult_None', 'metformin_No',
       'repaglinide_No', 'nateglinide_No', 'chlorpropamide_No',
       'glimepiride_No', 'acetohexamide_No', 'glipizide_No', 'glyburide_No',
       'tolbutamide_No', 'pioglitazone_No', 'rosiglitazone_No', 'acarbose_No',
       'miglitol_No', 'troglitazone_No', 'tolazamide_No', 'examide_No',
       'citoglipton_No', 'insulin_No', 'glyburide-metformin_No',
       'glipizide-metformin_No', 'glimepiride-pioglitazone_No',
       'metformin-rosiglitazone_No', 'metformin-pioglitazone_No', 'change_No',
       'diabetesMed_Yes', 'readmitted'],
      dtype='object')
```

Here are some quick hints at interpreting the field names:

- Your prediction target is *readmitted*
- Columns with the word diag indicate the diagnostic code of the illness or illnesses the patient was admitted with. For example, *diag_1_428* means the doctor said their first illness diagnosis is number "428". What illness does 428 correspond to? You could look it up in a codebook, but without more medical background it wouldn't mean anything to you anyway.
- A column names like *glimepiride_No* mean the patient did not have the medicine *glimepiride*. If this feature had a value of \\( \texttt{False} \\), then the patient did take the drug *glimepiride*
Features whose names begin with medical_specialty describe the specialty of the doctor seeing the patient. The values in these fields are all \\( \texttt{True} \\) or \\( \texttt{False} \\).


You have built a simple model, but the doctors say they don't know how to evaluate a model, and they'd like you to show them some evidence the model is doing something in line with their medical intuition. Create any graphics or tables that will show them a quick overview of what the model is doing?

They are very busy. So they want you to condense your model overview into just 1 or 2 graphics, rather than a long string of graphics.

We'll start after the point where you've built a basic model. Just run the following cell to build the model called my_model.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/hospital-readmissions/train.csv')

y = data.readmitted

base_features = [c for c in data.columns if c != "readmitted"]

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
```

Now use the following cell to create the materials for the doctors.

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

<div class="table-wrapper" markdown="block">

|          Weight | Feature                                  |
|----------------:|------------------------------------------|
| 0.0451 ± 0.0068 | number_inpatient                         |
| 0.0087 ± 0.0046 | number_emergency                         |
| 0.0062 ± 0.0053 | number_outpatient                        |
| 0.0033 ± 0.0016 | payer_code_MC                            |
| 0.0020 ± 0.0016 | diag_3_401                               |
| 0.0016 ± 0.0031 | medical_specialty_Emergency/Trauma       |
| 0.0014 ± 0.0024 | A1Cresult_None                           |
| 0.0014 ± 0.0021 | medical_specialty_Family/GeneralPractice |
| 0.0013 ± 0.0010 | diag_2_427                               |
| 0.0013 ± 0.0011 | diag_2_276                               |
| 0.0011 ± 0.0022 | age_[50-60)                              |
| 0.0010 ± 0.0022 | age_[80-90)                              |
| 0.0007 ± 0.0006 | repaglinide_No                           |
| 0.0006 ± 0.0010 | diag_1_428                               |
| 0.0006 ± 0.0022 | payer_code_SP                            |
| 0.0005 ± 0.0030 | insulin_No                               |
| 0.0004 ± 0.0028 | diabetesMed_Yes                          |
| 0.0004 ± 0.0021 | diag_3_250                               |
| 0.0003 ± 0.0018 | diag_2_250                               |
| 0.0003 ± 0.0015 | glipizide_No                             |
|   … 44 more …   |                                          |

</div>

It appears number_inpatient is a really important feature. The doctors would like to know more about that. Create a graph for them that shows how num_inpatient affects the model's predictions.

```python
# PDP for number_inpatient feature

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_name = 'number_inpatient'
# Create the data that we will plot
my_pdp = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns, feature=feature_name)

# plot it
pdp.pdp_plot(my_pdp, feature_name)
plt.show()
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/2.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/2.png)

```python
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_name = 'time_in_hospital'
# Create the data that we will plot
my_pdp = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns, feature=feature_name)

# plot it
pdp.pdp_plot(my_pdp, feature_name)
plt.show()
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/3.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/3.png)


Woah! It seems like *time_in_hospital* doesn't matter at all. The difference between the lowest value on the partial dependence plot and the highest value is about 5%.

If that is what your model concluded, the doctors will believe it. But it seems so low. Could the data be wrong, or is your model doing something more complex than they expect?

They'd like you to show them the raw readmission rate for each value of *time_in_hospital* to see how it compares to the partial dependence plot.

Make that plot.
Are the results similar or different?

```python
# Do concat to keep validation data separate, rather than using all original data
all_train = pd.concat([train_X, train_y], axis=1)

all_train.groupby(['time_in_hospital']).mean().readmitted.plot()
plt.show()
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/4.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/4.png)

Now the doctors are convinced you have the right data, and the model overview looked reasonable. It's time to turn this into a finished product they can use. Specifically, the hospital wants you to create a function *patient_risk_factors* that does the following

- Takes a single row with patient data (of the same format you as your raw data)
- Creates a visualization showing what features of that patient increased their risk of readmission, what features decreased it, and how much those features mattered.

It's not important to show every feature with every miniscule impact on the readmission risk. It's fine to focus on only the most important features for that patient.

```python
import shap  # package used to calculate Shap values

sample_data_for_prediction = val_X.iloc[0].astype(float)  # to test function

def patient_risk_factors(model, patient_data):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)
patient_risk_factors(my_model, sample_data_for_prediction)
```

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/5.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-09-shap-values/5.png)