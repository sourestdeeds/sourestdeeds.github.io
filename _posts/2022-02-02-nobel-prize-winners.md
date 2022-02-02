---
title: 'Nobel Prize Winners'
tags: [datacamp, data science]
layout: post
mathjax: true
categories: [DataCamp Projects]
permalink: /blog/:title/
---
{% assign counter = 1 %}
{% assign counter2 = 1 %}
{% assign link = "https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}



## 1. The most Nobel of Prizes
<p><img style="float: right;margin:5px 20px 5px 1px; max-width:250px" src="https://assets.datacamp.com/production/project_441/img/Nobel_Prize.png"></p>
<p>The Nobel Prize is perhaps the world's most well known scientific award. Except for the honor, prestige and substantial prize money the recipient also gets a gold medal showing Alfred Nobel (1833 - 1896) who established the prize. Every year it's given to scientists and scholars in the categories chemistry, literature, physics, physiology or medicine, economics, and peace. The first Nobel Prize was handed out in 1901, and at that time the Prize was very Eurocentric and male-focused, but nowadays it's not biased in any way whatsoever. Surely. Right?</p>
<p>Well, we're going to find out! The Nobel Foundation has made a dataset available of all prize winners from the start of the prize, in 1901, to 2016. Let's load it in and take a look.</p>


```python
# Loading in required libraries
import pandas as pd
import seaborn as sns
import numpy as np

# Reading in the Nobel Prize data
nobel = pd.read_csv('datasets/nobel.csv')

# Taking a look at the first several winners
nobel.head(n=6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>category</th>
      <th>prize</th>
      <th>motivation</th>
      <th>prize_share</th>
      <th>laureate_id</th>
      <th>laureate_type</th>
      <th>full_name</th>
      <th>birth_date</th>
      <th>birth_city</th>
      <th>birth_country</th>
      <th>sex</th>
      <th>organization_name</th>
      <th>organization_city</th>
      <th>organization_country</th>
      <th>death_date</th>
      <th>death_city</th>
      <th>death_country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1901</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 1901</td>
      <td>"in recognition of the extraordinary services ...</td>
      <td>1/1</td>
      <td>160</td>
      <td>Individual</td>
      <td>Jacobus Henricus van 't Hoff</td>
      <td>1852-08-30</td>
      <td>Rotterdam</td>
      <td>Netherlands</td>
      <td>Male</td>
      <td>Berlin University</td>
      <td>Berlin</td>
      <td>Germany</td>
      <td>1911-03-01</td>
      <td>Berlin</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1901</td>
      <td>Literature</td>
      <td>The Nobel Prize in Literature 1901</td>
      <td>"in special recognition of his poetic composit...</td>
      <td>1/1</td>
      <td>569</td>
      <td>Individual</td>
      <td>Sully Prudhomme</td>
      <td>1839-03-16</td>
      <td>Paris</td>
      <td>France</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1907-09-07</td>
      <td>Châtenay</td>
      <td>France</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1901</td>
      <td>Medicine</td>
      <td>The Nobel Prize in Physiology or Medicine 1901</td>
      <td>"for his work on serum therapy, especially its...</td>
      <td>1/1</td>
      <td>293</td>
      <td>Individual</td>
      <td>Emil Adolf von Behring</td>
      <td>1854-03-15</td>
      <td>Hansdorf (Lawice)</td>
      <td>Prussia (Poland)</td>
      <td>Male</td>
      <td>Marburg University</td>
      <td>Marburg</td>
      <td>Germany</td>
      <td>1917-03-31</td>
      <td>Marburg</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1901</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1901</td>
      <td>NaN</td>
      <td>1/2</td>
      <td>462</td>
      <td>Individual</td>
      <td>Jean Henry Dunant</td>
      <td>1828-05-08</td>
      <td>Geneva</td>
      <td>Switzerland</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1910-10-30</td>
      <td>Heiden</td>
      <td>Switzerland</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1901</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1901</td>
      <td>NaN</td>
      <td>1/2</td>
      <td>463</td>
      <td>Individual</td>
      <td>Frédéric Passy</td>
      <td>1822-05-20</td>
      <td>Paris</td>
      <td>France</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1912-06-12</td>
      <td>Paris</td>
      <td>France</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1901</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1901</td>
      <td>"in recognition of the extraordinary services ...</td>
      <td>1/1</td>
      <td>1</td>
      <td>Individual</td>
      <td>Wilhelm Conrad Röntgen</td>
      <td>1845-03-27</td>
      <td>Lennep (Remscheid)</td>
      <td>Prussia (Germany)</td>
      <td>Male</td>
      <td>Munich University</td>
      <td>Munich</td>
      <td>Germany</td>
      <td>1923-02-10</td>
      <td>Munich</td>
      <td>Germany</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%nose

last_value = _
    
def test_pandas_loaded():
    assert pd.__name__ == 'pandas', \
        "pandas should be imported as pd"
    
def test_seaborn_loaded():
    assert sns.__name__ == 'seaborn', \
        "seaborn should be imported as sns"

def test_numpy_loaded():
    assert np.__name__ == 'numpy', \
        "numpy should be imported as np"

import pandas as pd
        
def test_nobel_correctly_loaded():
    correct_nobel = pd.read_csv('datasets/nobel.csv')
    assert correct_nobel.equals(nobel), \
        "The variable nobel should contain the data in 'datasets/nobel.csv'"

def test_Wilhelm_was_selected():
    assert "Wilhelm Conrad" in last_value.to_string(), \
        "Hmm, it seems you have not displayed at least the first six entries of nobel. A fellow named Wilhelm Conrad Röntgen should be displayed."
```






    5/5 tests passed




## 2. So, who gets the Nobel Prize?
<p>Just looking at the first couple of prize winners, or Nobel laureates as they are also called, we already see a celebrity: Wilhelm Conrad Röntgen, the guy who discovered X-rays. And actually, we see that all of the winners in 1901 were guys that came from Europe. But that was back in 1901, looking at all winners in the dataset, from 1901 to 2016, which sex and which country is the most commonly represented? </p>
<p>(For <em>country</em>, we will use the birth_country of the winner, as the organization_country is NaN for all shared Nobel Prizes.)</p>


```python
# Display the number of (possibly shared) Nobel Prizes handed
# out between 1901 and 2016
display(len(nobel))

# Display the number of prizes won by male and female recipients.
display(nobel['sex'].value_counts())

# Display the number of prizes won by the top 10 nationalities.
nobel['birth_country'].value_counts().head(10)
```


    911



    Male      836
    Female     49
    Name: sex, dtype: int64





    United States of America    259
    United Kingdom               85
    Germany                      61
    France                       51
    Sweden                       29
    Japan                        24
    Canada                       18
    Netherlands                  18
    Italy                        17
    Russia                       17
    Name: birth_country, dtype: int64




```python
nobel.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>category</th>
      <th>prize</th>
      <th>motivation</th>
      <th>prize_share</th>
      <th>laureate_id</th>
      <th>laureate_type</th>
      <th>full_name</th>
      <th>birth_date</th>
      <th>birth_city</th>
      <th>birth_country</th>
      <th>sex</th>
      <th>organization_name</th>
      <th>organization_city</th>
      <th>organization_country</th>
      <th>death_date</th>
      <th>death_city</th>
      <th>death_country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1901</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 1901</td>
      <td>"in recognition of the extraordinary services ...</td>
      <td>1/1</td>
      <td>160</td>
      <td>Individual</td>
      <td>Jacobus Henricus van 't Hoff</td>
      <td>1852-08-30</td>
      <td>Rotterdam</td>
      <td>Netherlands</td>
      <td>Male</td>
      <td>Berlin University</td>
      <td>Berlin</td>
      <td>Germany</td>
      <td>1911-03-01</td>
      <td>Berlin</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1901</td>
      <td>Literature</td>
      <td>The Nobel Prize in Literature 1901</td>
      <td>"in special recognition of his poetic composit...</td>
      <td>1/1</td>
      <td>569</td>
      <td>Individual</td>
      <td>Sully Prudhomme</td>
      <td>1839-03-16</td>
      <td>Paris</td>
      <td>France</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1907-09-07</td>
      <td>Châtenay</td>
      <td>France</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1901</td>
      <td>Medicine</td>
      <td>The Nobel Prize in Physiology or Medicine 1901</td>
      <td>"for his work on serum therapy, especially its...</td>
      <td>1/1</td>
      <td>293</td>
      <td>Individual</td>
      <td>Emil Adolf von Behring</td>
      <td>1854-03-15</td>
      <td>Hansdorf (Lawice)</td>
      <td>Prussia (Poland)</td>
      <td>Male</td>
      <td>Marburg University</td>
      <td>Marburg</td>
      <td>Germany</td>
      <td>1917-03-31</td>
      <td>Marburg</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1901</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1901</td>
      <td>NaN</td>
      <td>1/2</td>
      <td>462</td>
      <td>Individual</td>
      <td>Jean Henry Dunant</td>
      <td>1828-05-08</td>
      <td>Geneva</td>
      <td>Switzerland</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1910-10-30</td>
      <td>Heiden</td>
      <td>Switzerland</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1901</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1901</td>
      <td>NaN</td>
      <td>1/2</td>
      <td>463</td>
      <td>Individual</td>
      <td>Frédéric Passy</td>
      <td>1822-05-20</td>
      <td>Paris</td>
      <td>France</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1912-06-12</td>
      <td>Paris</td>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%nose
last_value = _

correct_value = nobel['birth_country'].value_counts().head(10)

def test_last_value_correct():
    assert last_value.equals(correct_value), \
        "The number of prizes won by the top 10 nationalities doesn't seem correct... Maybe check the hint?"
```






    1/1 tests passed;

    




## 3. USA dominance
<p>Not so surprising perhaps: the most common Nobel laureate between 1901 and 2016 was a man born in the United States of America. But in 1901 all the winners were European. When did the USA start to dominate the Nobel Prize charts?</p>


```python
# Calculating the proportion of USA born winners per decade
nobel['usa_born_winner'] = nobel['birth_country']=="United States of America"
nobel['decade'] = (np.floor(nobel['year'] / 10) * 10).astype(int)
prop_usa_winners = nobel.groupby('decade', as_index=False)['usa_born_winner'].mean()

# Display the proportions of USA born winners per decade
prop_usa_winners.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>decade</th>
      <th>usa_born_winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1900</td>
      <td>0.017544</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1910</td>
      <td>0.075000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1920</td>
      <td>0.074074</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1930</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1940</td>
      <td>0.302326</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%nose

def test_decade_int():
    assert nobel['decade'].dtype == "int64", \
    "Hmm, it looks like the decade column isn't calculated correctly. Did you forget to convert it to an integer?"

def test_correct_prop_usa_winners():
    correct_prop_usa_winners = nobel.groupby('decade', as_index=False)['usa_born_winner'].mean()
    assert correct_prop_usa_winners.equals(prop_usa_winners), \
        "prop_usa_winners should contain the proportion of usa_born_winner by decade. Don't forget to set as_index=False in the groupby() method."
```






    2/2 tests passed




## 4. USA dominance, visualized
<p>A table is OK, but to <em>see</em> when the USA started to dominate the Nobel charts we need a plot!</p>


```python
# Setting the plotting theme
sns.set()
# and setting the size of all plots.
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [11, 7]

# Plotting USA born winners 
ax = sns.lineplot(data=prop_usa_winners, x='decade', y='usa_born_winner')

# Adding %-formatting to the y-axis
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
```


<br>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
{% assign counter = counter | plus: 1 %} 
<br>


```python
%%nose

def test_y_axis():
    assert all(ax.get_lines()[0].get_ydata() == prop_usa_winners.usa_born_winner), \
    'The plot should be assigned to ax and have usa_born_winner on the y-axis'
    
def test_x_axis():
    assert all(ax.get_lines()[0].get_xdata() == prop_usa_winners.decade), \
    'The plot should be assigned to ax and have decade on the x-axis'
```






    2/2 tests passed




## 5. What is the gender of a typical Nobel Prize winner?
<p>So the USA became the dominating winner of the Nobel Prize first in the 1930s and had kept the leading position ever since. But one group that was in the lead from the start, and never seems to let go, are <em>men</em>. Maybe it shouldn't come as a shock that there is some imbalance between how many male and female prize winners there are, but how significant is this imbalance? And is it better or worse within specific prize categories like physics, medicine, literature, etc.?</p>


```python
# Calculating the proportion of female laureates per decade
nobel['female_winner'] = nobel['sex']=="Female"
prop_female_winners = nobel.groupby(['decade', 'category'], as_index=False)['female_winner'].mean()

# Plotting USA born winners with % winners on the y-axis
# Plotting USA born winners 
ax = sns.lineplot(data=prop_female_winners, x='decade', y='female_winner', hue='category')

# Adding %-formatting to the y-axis
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
```


<br>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
{% assign counter = counter | plus: 1 %} 
<br>


```python
%%nose
    

def test_correct_prop_usa_winners():
    correct_prop_female_winners = nobel.groupby(['decade', 'category'], as_index=False)['female_winner'].mean()
    assert correct_prop_female_winners.equals(prop_female_winners), \
        "prop_female_winners should contain the proportion of female_winner by decade. Don't forget to set as_index=False in the groupby() method."

def test_y_axis():
    assert all(pd.Series(ax.get_lines()[0].get_ydata()).isin(prop_female_winners.female_winner)), \
    'The plot should be assigned to ax and have female_winner on the y-axis'
    
def test_x_axis():
    assert all(pd.Series(ax.get_lines()[0].get_xdata()).isin(prop_female_winners.decade)), \
    'The plot should be assigned to ax and have decade on the x-axis'
```






    3/3 tests passed




## 6. The first woman to win the Nobel Prize
<p>The plot above is a bit messy as the lines are overplotting. But it does show some interesting trends and patterns. Overall the imbalance is pretty large with physics, economics, and chemistry having the largest imbalance. Medicine has a somewhat positive trend, and since the 1990s the literature prize is also now more balanced. The big outlier is the peace prize during the 2010s, but keep in mind that this just covers the years 2010 to 2016.</p>
<p>Given this imbalance, who was the first woman to receive a Nobel Prize? And in what category?</p>


```python
# Picking out the first woman to win a Nobel Prize
nobel[nobel['sex']=="Female"].nsmallest(1, 'year')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>category</th>
      <th>prize</th>
      <th>motivation</th>
      <th>prize_share</th>
      <th>laureate_id</th>
      <th>laureate_type</th>
      <th>full_name</th>
      <th>birth_date</th>
      <th>birth_city</th>
      <th>...</th>
      <th>sex</th>
      <th>organization_name</th>
      <th>organization_city</th>
      <th>organization_country</th>
      <th>death_date</th>
      <th>death_city</th>
      <th>death_country</th>
      <th>usa_born_winner</th>
      <th>decade</th>
      <th>female_winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>1903</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1903</td>
      <td>"in recognition of the extraordinary services ...</td>
      <td>1/4</td>
      <td>6</td>
      <td>Individual</td>
      <td>Marie Curie, née Sklodowska</td>
      <td>1867-11-07</td>
      <td>Warsaw</td>
      <td>...</td>
      <td>Female</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1934-07-04</td>
      <td>Sallanches</td>
      <td>France</td>
      <td>False</td>
      <td>1900</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
%%nose

last_value = _
    
def test_Marie_was_selected():
    assert "Marie Curie" in last_value.to_string(), \
        "Hmm, it seems you have not displayed the row of the first woman to win a Nobel Prize, her first name should be Marie."
```






    1/1 tests passed




## 7. Repeat laureates
<p>For most scientists/writers/activists a Nobel Prize would be the crowning achievement of a long career. But for some people, one is just not enough, and few have gotten it more than once. Who are these lucky few? (Having won no Nobel Prize myself, I'll assume it's just about luck.)</p>


```python
# Selecting the laureates that have received 2 or more prizes.
nobel.groupby('full_name').filter(lambda x: len(x) >= 2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>category</th>
      <th>prize</th>
      <th>motivation</th>
      <th>prize_share</th>
      <th>laureate_id</th>
      <th>laureate_type</th>
      <th>full_name</th>
      <th>birth_date</th>
      <th>birth_city</th>
      <th>...</th>
      <th>sex</th>
      <th>organization_name</th>
      <th>organization_city</th>
      <th>organization_country</th>
      <th>death_date</th>
      <th>death_city</th>
      <th>death_country</th>
      <th>usa_born_winner</th>
      <th>decade</th>
      <th>female_winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>1903</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1903</td>
      <td>"in recognition of the extraordinary services ...</td>
      <td>1/4</td>
      <td>6</td>
      <td>Individual</td>
      <td>Marie Curie, née Sklodowska</td>
      <td>1867-11-07</td>
      <td>Warsaw</td>
      <td>...</td>
      <td>Female</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1934-07-04</td>
      <td>Sallanches</td>
      <td>France</td>
      <td>False</td>
      <td>1900</td>
      <td>True</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1911</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 1911</td>
      <td>"in recognition of her services to the advance...</td>
      <td>1/1</td>
      <td>6</td>
      <td>Individual</td>
      <td>Marie Curie, née Sklodowska</td>
      <td>1867-11-07</td>
      <td>Warsaw</td>
      <td>...</td>
      <td>Female</td>
      <td>Sorbonne University</td>
      <td>Paris</td>
      <td>France</td>
      <td>1934-07-04</td>
      <td>Sallanches</td>
      <td>France</td>
      <td>False</td>
      <td>1910</td>
      <td>True</td>
    </tr>
    <tr>
      <th>89</th>
      <td>1917</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1917</td>
      <td>NaN</td>
      <td>1/1</td>
      <td>482</td>
      <td>Organization</td>
      <td>Comité international de la Croix Rouge (Intern...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1910</td>
      <td>False</td>
    </tr>
    <tr>
      <th>215</th>
      <td>1944</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1944</td>
      <td>NaN</td>
      <td>1/1</td>
      <td>482</td>
      <td>Organization</td>
      <td>Comité international de la Croix Rouge (Intern...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1940</td>
      <td>False</td>
    </tr>
    <tr>
      <th>278</th>
      <td>1954</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 1954</td>
      <td>"for his research into the nature of the chemi...</td>
      <td>1/1</td>
      <td>217</td>
      <td>Individual</td>
      <td>Linus Carl Pauling</td>
      <td>1901-02-28</td>
      <td>Portland, OR</td>
      <td>...</td>
      <td>Male</td>
      <td>California Institute of Technology (Caltech)</td>
      <td>Pasadena, CA</td>
      <td>United States of America</td>
      <td>1994-08-19</td>
      <td>Big Sur, CA</td>
      <td>United States of America</td>
      <td>True</td>
      <td>1950</td>
      <td>False</td>
    </tr>
    <tr>
      <th>283</th>
      <td>1954</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1954</td>
      <td>NaN</td>
      <td>1/1</td>
      <td>515</td>
      <td>Organization</td>
      <td>Office of the United Nations High Commissioner...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1950</td>
      <td>False</td>
    </tr>
    <tr>
      <th>298</th>
      <td>1956</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1956</td>
      <td>"for their researches on semiconductors and th...</td>
      <td>1/3</td>
      <td>66</td>
      <td>Individual</td>
      <td>John Bardeen</td>
      <td>1908-05-23</td>
      <td>Madison, WI</td>
      <td>...</td>
      <td>Male</td>
      <td>University of Illinois</td>
      <td>Urbana, IL</td>
      <td>United States of America</td>
      <td>1991-01-30</td>
      <td>Boston, MA</td>
      <td>United States of America</td>
      <td>True</td>
      <td>1950</td>
      <td>False</td>
    </tr>
    <tr>
      <th>306</th>
      <td>1958</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 1958</td>
      <td>"for his work on the structure of proteins, es...</td>
      <td>1/1</td>
      <td>222</td>
      <td>Individual</td>
      <td>Frederick Sanger</td>
      <td>1918-08-13</td>
      <td>Rendcombe</td>
      <td>...</td>
      <td>Male</td>
      <td>University of Cambridge</td>
      <td>Cambridge</td>
      <td>United Kingdom</td>
      <td>2013-11-19</td>
      <td>Cambridge</td>
      <td>United Kingdom</td>
      <td>False</td>
      <td>1950</td>
      <td>False</td>
    </tr>
    <tr>
      <th>340</th>
      <td>1962</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1962</td>
      <td>NaN</td>
      <td>1/1</td>
      <td>217</td>
      <td>Individual</td>
      <td>Linus Carl Pauling</td>
      <td>1901-02-28</td>
      <td>Portland, OR</td>
      <td>...</td>
      <td>Male</td>
      <td>California Institute of Technology (Caltech)</td>
      <td>Pasadena, CA</td>
      <td>United States of America</td>
      <td>1994-08-19</td>
      <td>Big Sur, CA</td>
      <td>United States of America</td>
      <td>True</td>
      <td>1960</td>
      <td>False</td>
    </tr>
    <tr>
      <th>348</th>
      <td>1963</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1963</td>
      <td>NaN</td>
      <td>1/2</td>
      <td>482</td>
      <td>Organization</td>
      <td>Comité international de la Croix Rouge (Intern...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1960</td>
      <td>False</td>
    </tr>
    <tr>
      <th>424</th>
      <td>1972</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1972</td>
      <td>"for their jointly developed theory of superco...</td>
      <td>1/3</td>
      <td>66</td>
      <td>Individual</td>
      <td>John Bardeen</td>
      <td>1908-05-23</td>
      <td>Madison, WI</td>
      <td>...</td>
      <td>Male</td>
      <td>University of Illinois</td>
      <td>Urbana, IL</td>
      <td>United States of America</td>
      <td>1991-01-30</td>
      <td>Boston, MA</td>
      <td>United States of America</td>
      <td>True</td>
      <td>1970</td>
      <td>False</td>
    </tr>
    <tr>
      <th>505</th>
      <td>1980</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 1980</td>
      <td>"for their contributions concerning the determ...</td>
      <td>1/4</td>
      <td>222</td>
      <td>Individual</td>
      <td>Frederick Sanger</td>
      <td>1918-08-13</td>
      <td>Rendcombe</td>
      <td>...</td>
      <td>Male</td>
      <td>MRC Laboratory of Molecular Biology</td>
      <td>Cambridge</td>
      <td>United Kingdom</td>
      <td>2013-11-19</td>
      <td>Cambridge</td>
      <td>United Kingdom</td>
      <td>False</td>
      <td>1980</td>
      <td>False</td>
    </tr>
    <tr>
      <th>523</th>
      <td>1981</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1981</td>
      <td>NaN</td>
      <td>1/1</td>
      <td>515</td>
      <td>Organization</td>
      <td>Office of the United Nations High Commissioner...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1980</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>13 rows × 21 columns</p>
</div>




```python
%%nose

last_value = _
    
def test_something():
    correct_last_value = nobel.groupby('full_name').filter(lambda group: len(group) >= 2)
    assert correct_last_value.equals(last_value), \
        "Did you use groupby followed by the filter method? Did you filter to keep only those with >= 2 prises?"
```






    1/1 tests passed




## 8. How old are you when you get the prize?
<p>The list of repeat winners contains some illustrious names! We again meet Marie Curie, who got the prize in physics for discovering radiation and in chemistry for isolating radium and polonium. John Bardeen got it twice in physics for transistors and superconductivity, Frederick Sanger got it twice in chemistry, and Linus Carl Pauling got it first in chemistry and later in peace for his work in promoting nuclear disarmament. We also learn that organizations also get the prize as both the Red Cross and the UNHCR have gotten it twice.</p>
<p>But how old are you generally when you get the prize?</p>


```python
# Converting birth_date from String to datetime
nobel['birth_date'] = pd.to_datetime(nobel['birth_date'])

# Calculating the age of Nobel Prize winners
nobel['age'] = nobel['year'] - nobel['birth_date'].dt.year

# Plotting the age of Nobel Prize winners
sns.lmplot(x='year', y='age', data=nobel, lowess=True, 
           aspect=2, line_kws={'color' : 'black'})
```




    <seaborn.axisgrid.FacetGrid at 0x7f8ef3496898>



<br>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
{% assign counter = counter | plus: 1 %} 
<br>


```python
%%nose

ax = _
    
def test_birth_date():
    assert pd.to_datetime(nobel['birth_date']).equals(nobel['birth_date']), \
        "Have you converted nobel['birth_date'] using to_datetime?"

    
def test_year():
    assert (nobel['year'] - nobel['birth_date'].dt.year).equals(nobel['age']), \
        "Have you caluclated nobel['year'] correctly?"

def test_plot_data():
    assert list(ax.data)[0] in ["age", "year"] and list(ax.data)[1] in ["age", "year"], \
    'The plot should show year on the x-axis and age on the y-axis'
    
# Why not this testing code?
# def test_plot_data():
#     assert list(ax.data)[0] == "age" and list(ax.data)[1] == "year", \
#     'The plot should show year on the x-axis and age on the y-axis'
```






    3/3 tests passed




## 9. Age differences between prize categories
<p>The plot above shows us a lot! We see that people use to be around 55 when they received the price, but nowadays the average is closer to 65. But there is a large spread in the laureates' ages, and while most are 50+, some are very young.</p>
<p>We also see that the density of points is much high nowadays than in the early 1900s -- nowadays many more of the prizes are shared, and so there are many more winners. We also see that there was a disruption in awarded prizes around the Second World War (1939 - 1945). </p>
<p>Let's look at age trends within different prize categories.</p>


```python
# Same plot as above, but separate plots for each type of Nobel Prize
sns.lmplot(data=nobel, x= 'year', y='age', row='category')
```




    <seaborn.axisgrid.FacetGrid at 0x7f8ef31f87f0>



<br>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
{% assign counter = counter | plus: 1 %} 
<br>


```python
%%nose

ax = _
    
def test_plot_data():
    assert list(ax.data)[0] in ["age", "year", "category"] and \
           list(ax.data)[1] in ["age", "year", "category"] and \
           list(ax.data)[2] in ["age", "year", "category"], \
    'The plot should show year on the x-axis and age on the y-axis, with one plot row for each category.'
```






    1/1 tests passed




## 10. Oldest and youngest winners
<p>More plots with lots of exciting stuff going on! We see that both winners of the chemistry, medicine, and physics prize have gotten older over time. The trend is strongest for physics: the average age used to be below 50, and now it's almost 70. Literature and economics are more stable. We also see that economics is a newer category. But peace shows an opposite trend where winners are getting younger! </p>
<p>In the peace category we also a winner around 2010 that seems exceptionally young. This begs the questions, who are the oldest and youngest people ever to have won a Nobel Prize?</p>


```python
# The oldest winner of a Nobel Prize as of 2016
display(nobel.nlargest(1, 'age'))

# The youngest winner of a Nobel Prize as of 2016
nobel.nsmallest(1, 'age')
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>category</th>
      <th>prize</th>
      <th>motivation</th>
      <th>prize_share</th>
      <th>laureate_id</th>
      <th>laureate_type</th>
      <th>full_name</th>
      <th>birth_date</th>
      <th>birth_city</th>
      <th>...</th>
      <th>organization_name</th>
      <th>organization_city</th>
      <th>organization_country</th>
      <th>death_date</th>
      <th>death_city</th>
      <th>death_country</th>
      <th>usa_born_winner</th>
      <th>decade</th>
      <th>female_winner</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>793</th>
      <td>2007</td>
      <td>Economics</td>
      <td>The Sveriges Riksbank Prize in Economic Scienc...</td>
      <td>"for having laid the foundations of mechanism ...</td>
      <td>1/3</td>
      <td>820</td>
      <td>Individual</td>
      <td>Leonid Hurwicz</td>
      <td>1917-08-21</td>
      <td>Moscow</td>
      <td>...</td>
      <td>University of Minnesota</td>
      <td>Minneapolis, MN</td>
      <td>United States of America</td>
      <td>2008-06-24</td>
      <td>Minneapolis, MN</td>
      <td>United States of America</td>
      <td>False</td>
      <td>2000</td>
      <td>False</td>
      <td>90.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 22 columns</p>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>category</th>
      <th>prize</th>
      <th>motivation</th>
      <th>prize_share</th>
      <th>laureate_id</th>
      <th>laureate_type</th>
      <th>full_name</th>
      <th>birth_date</th>
      <th>birth_city</th>
      <th>...</th>
      <th>organization_name</th>
      <th>organization_city</th>
      <th>organization_country</th>
      <th>death_date</th>
      <th>death_city</th>
      <th>death_country</th>
      <th>usa_born_winner</th>
      <th>decade</th>
      <th>female_winner</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>885</th>
      <td>2014</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 2014</td>
      <td>"for their struggle against the suppression of...</td>
      <td>1/2</td>
      <td>914</td>
      <td>Individual</td>
      <td>Malala Yousafzai</td>
      <td>1997-07-12</td>
      <td>Mingora</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2010</td>
      <td>True</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 22 columns</p>
</div>




```python
%%nose
    
last_value = _
    
def test_oldest_or_youngest():
    assert 'Hurwicz' in last_value.to_string() or 'Yousafzai' in last_value.to_string(), \
        "Have you displayed the row of the oldest winner and the row of the youngest winner?"
```






    1/1 tests passed




## 11. You get a prize!
<p><img style="float: right;margin:20px 20px 20px 20px; max-width:200px" src="https://assets.datacamp.com/production/project_441/img/paint_nobel_prize.png"></p>
<p>Hey! You get a prize for making it to the very end of this notebook! It might not be a Nobel Prize, but I made it myself in paint so it should count for something. But don't despair, Leonid Hurwicz was 90 years old when he got his prize, so it might not be too late for you. Who knows.</p>
<p>Before you leave, what was again the name of the youngest winner ever who in 2014 got the prize for "[her] struggle against the suppression of children and young people and for the right of all children to education"?</p>


```python
# The name of the youngest winner of the Nobel Prize as of 2016
youngest_winner = 'Malala Yousafzai'
```


```python
%%nose

import re
    
def test_right_name():
    assert re.match("(malala|yousafzai)", youngest_winner.lower()), \
        "youngest_winner should be a string. Try writing only the first / given name."
```






    1/1 tests passed



