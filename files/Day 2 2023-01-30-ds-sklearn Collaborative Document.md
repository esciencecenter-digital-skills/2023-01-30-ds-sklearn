![](https://i.imgur.com/iywjz8s.png)


# Day 2 2023-01-30-ds-sklearn Collaborative Document

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


Collaborative Document day 1: https://tinyurl.com/sklearn-day1

Collaborative Document day 2: https://tinyurl.com/sklearn-day2

Collaborative Document day 3: https://tinyurl.com/sklearn-day3

Collaborative Document day 4: https://tinyurl.com/sklearn-day4

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## 🎓 Certificate of attendance

If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, raise your hand in zoom. Click on the icon labeled "Reactions" in the toolbar on the bottom center of your screen,
then click the button 'Raise Hand ✋'. For urgent questions, just unmute and speak up!

You can also ask questions or type 'I need help' in the chat window and helpers will try to help you.
Please note it is not necessary to monitor the chat - the helpers will make sure that relevant questions are addressed in a plenary way.
(By the way, off-topic questions will still be answered in the chat)


## 🖥 Workshop website

https://esciencecenter-digital-skills.github.io/2023-01-30-ds-sklearn/

🛠 Setup

https://github.com/INRIA/scikit-learn-mooc/blob/main/local-install-instructions.md


## 👩‍🏫👩‍💻🎓 Instructors

Sven van der Burg, Cunliang Geng, Olga Lyashevska, Dani Bodor
## 🧑‍🙋 Helpers

Barbara Vreede



## :icecream: Icebreaker

Bring something blue back to your screen! & something square. Not the same thing!


## 🗓️ Agenda
| 09:00 | Welcome and icebreaker                         |
|-------|------------------------------------------------|
| 09:15 | Fitting a scikit-learn model on numerical data |
| 10:15 | Coffee break                                   |
| 10:30 | Handling categorical data                      |
| 11:30 | Coffee break                                   |
| 11:45 | Handling categorical data                      |
| 12:45 | Wrap-up                                        |
| 13:00 | END                                            |
## 🔧 Exercises
### Exercise (in breakout rooms): Compare with simple baselines
The goal of this exercise is to compare the performance of our classifier in the previous notebook (roughly 81% accuracy with LogisticRegression) to some simple baseline classifiers. The simplest baseline classifier is one that always predicts the same class, irrespective of the input data.

What would be the score of a model that always predicts ' >50K'?

What would be the score of a model that always predicts ' <=50K'?

Is 81% or 82% accuracy a good score for this problem?

Use a DummyClassifier such that the resulting classifier will always predict the class ' >50K'. What is the accuracy score on the test set? Repeat the experiment by always predicting the class ' <=50K'.

Hint: you can set the strategy parameter of the DummyClassifier to achieve the desired behavior.

You can import DummyClassifier like this:
```python
from sklearn.dummy import DummyClassifier
```



### Exercise: Recap fitting a scikit-learn model on numerical data
#### 1. Why do we need two sets: a train set and a test set?

a) to train the model faster
b) to validate the model on unseen data
c) to improve the accuracy of the model

Select all answers that apply

#### 2. The generalization performance of a scikit-learn model can be evaluated by:

a) calling fit to train the model on the training set, predict on the test set to get the predictions, and compute the score by passing the predictions and the true target values to some metric function
b) calling fit to train the model on the training set and score to compute the score on the test set
c) calling cross_validate by passing the model, the data and the target
d) calling fit_transform on the data and then score to compute the score on the test set

Select all answers that apply

#### 3. When calling `cross_validate(estimator, X, y, cv=5)`, the following happens:

a) X and y are internally split five times with non-overlapping test sets
b) estimator.fit is called 5 times on the full X and y
c) estimator.fit is called 5 times, each time on a different training set
d) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the train sets
e) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the test sets

Select all answers that apply

#### 4. (optional) Scaling
We define a 2-dimensional dataset represented graphically as follows:
![](https://i.imgur.com/muvSbI6.png)

Question

If we process the dataset using a StandardScaler with the default parameters, which of the following results do you expect:

![](https://i.imgur.com/t5mTlVG.png)


a) Preprocessing A
b) Preprocessing B
c) Preprocessing C
d) Preprocessing D

Select a single answer

#### 5. (optional) Cross-validation allows us to:

a) train the model faster
b) measure the generalization performance of the model
c) reach better generalization performance
d) estimate the variability of the generalization score

Select all answers that apply



### Exercise: The impact of using integer encoding for with logistic regression (breakout rooms of 3-4 people):
first load the data:
```python=
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```

Q1: Use `sklearn.compose.make_column_selector` to automatically select columns containing strings
that correspond to categorical features in our dataset.

Q2: Define a scikit-learn pipeline composed of an `OrdinalEncoder` and a `LogisticRegression` classifier and
evaluate it using cross-validation.
*Note*: Because `OrdinalEncoder` can raise errors if it sees an unknown category at prediction time,  you can set the `handle_unknown="use_encoded_value"` and `unknown_value=-1` parameters. You can refer to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
for more details regarding these parameters.

Q3: Now, compare the generalization performance of our previous
model with a new model where instead of using an `OrdinalEncoder`, we will
use a `OneHotEncoder`. Repeat the model evaluation using cross-validation.
Compare the score of both models and conclude on the impact of choosing a
specific encoding strategy when using a linear model.


## 🧠 Collaborative Notes

Continuing in [yesterday's notebook](https://hackmd.io/8DiiApNUQCeYiTEFrVWsmg?view#%F0%9F%A7%A0-Collaborative-Notes)...

Import the module:
```python=
from sklearn.neighbors import KNeighborsClassifier
```

Check the documentation for KNeighborsClassifier:
```python=
KNeighborsClassfier?
```
Note that 5 is the default for `n_neighbors`, so we need to actively update this parameter if we want a different value.

Create the model and pass data to train it:
```python=
model = KneighborsClassifier(n_neighbors=50)
model.fit(data, target)

# check the first ten rows of our data to testthe predictions
first_data_values = data[:10]
first_predictions = model.predict(first_data_values)
first_target_values = target[:10]
first_predictions == first_target_values
# or, using the mean
(first_predictions == first_target_values).mean()

# comparing the accuracy of the model's predictions
model.score(data, target) # training data
model.score(data_test, target_test) # test data
```

No one set of parameters works best for everything ("No free lunch" theory). The number of nearest neighbors that works best will be different in different situations. The number to use is determined through trial and error, looking at the accuracy of the model (and the difference in accuracy between training and test data).

The number of neighbors is called a hyperparameter; looping over these (e.g. n_neighbors, but can be others) is called hyperparameter tuning.

### Working with numerical data
We will 
- Identify numerical data in a heterogeneous dataset
- Select the subset of columns with numerical data
- Use a scikit-learn helper to seperate into train/test
- Train and evaluate a more complex model

Load the data & select numerical data
```python=
adult_census = pd.read_csv("../datasets/adult-census.csv")
# drop duplicate column
adult_census = adult_census.drop(columns="education-num")

data = adult_census.drop(columns="class")
target = adult_census["class"]

# quickly check the data with
target.head(2)

numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
data_numeric = data[numerical_columns]
```

Split the data
```python=
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric,
    target,
    random_state=42,
    test_size=0.25) # the function randomly selects a number of samples; the random_state is a randomness seed
```

Take a look at the resulting objects:
```python=
data_numeric.shape
data_train.shape
data_test.shape
target_train.shape
```

Even with a large data set, it is important to keep the test set fixed to prevent over-fitting.

### Logistic regression
... is essentially: trying to come up with a formula like:

if 0.1 * age + 3.3 * hours_per_week - 15.1 > 0 then: predict high income.
Otherwise predict low income

```python=
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(data_train, target_train)

accuracy = model.score(data_test, target_test)
accuracy
```
This model is ~ 80% accurate, so not better than the k nearest neighbors model.

timestamp 9:55 - [Exercise](https://hackmd.io/OffraZphTzmNKFQjRAJhDA?both#Exercise-in-breakout-rooms-Compare-with-simple-baselines)

_break until 10:30_

### Exercise discussion
```python=
adult_census["class"].value_counts()
(target == " <50K").mean()
```
This shows that 76% of people makes less than 50k, so simply assigning everyone to that category has already an "accuracy" baseline of 76%. Running this check prior to designing a model is a good start!

### Preprocessing for numerical features
Introducing:
- an example of preprocessing, namely scaling numerical variables
- using a scikit-learn pipeline to chain preprocessing and model training

```python=
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data_train)
```

The fit method for transformers is similar to the fit method for predictors. The main difference is that the former has a single argument (the data matrix), whereas the latter has two arguments (the data matrix and the target).

![](https://inria.github.io/scikit-learn-mooc/_images/api_diagram-transformer.fit.svg)

We can inspect the computed means and standard deviations.
```python=
scaler.mean_
scaler.scale_ # standard deviations
```
![](https://inria.github.io/scikit-learn-mooc/_images/api_diagram-transformer.transform.svg)
```python=
data_train_scaled = scaler.transform(data_train)
data_train_scaled
```

![](https://inria.github.io/scikit-learn-mooc/_images/api_diagram-transformer.fit_transform.svg)

Let's make a dataframe so we can inspect the transformed data better.
```python=
data_train_scaled_df = pd.DataFrame(data_train_scaled,
                                   columns = data_train.columns)
data_train_scaled_df.describe()
```

The aim of preprocessing is to mold your data in such a way that it improves the modelling.

We can simplify this process by creating a pipeline:

```python=
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(),
                     LogisticRegression())
```

call `model` to take a look at the pipeline.
![](https://i.imgur.com/gTyodwi.png)


To speed up the modelling, we are going to scale the data to a mean of 0 and a standard deviation of 1.

To show that this actually speeds up the process, we'll also use a timer here.
```python=
import time
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
```

Call `elapsed_time` to see the elapsed time (0.080... on the teacher's machine)

```python=
model_without_scaling = LogisticRegression()
start = time.time()
model_without_scaling.fit(data_train, target_train)
elapsed_time = time.time() - start
```
Here the `elapsed_time` is closer to 0.099... on the teacher's machine; it takes a bit longer.

The accuracy is likely not affected by the scaling.

### Cross-validation
<iframe width="740" height="476" src="https://www.youtube.com/embed/kLWvI9fSnKc" title="Validation of a model" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


Take-home messages:
- Full data should not be used for scoring a model
- Train-test split can be used to evaluate the generalization performance on unseen data
- Cross-validation is used for evaluating the variability of our estimation of the generalization performance

```python=
from sklearn.model_selection import cross_validate

cv_result = cross_validate(model, data_numeric, target, cv=5)
```
This uses the k-fold cross-validation with 5 splits.

inspecting `cv_result` shows us the `fit_time` (five times for each time the model is fit), and the `score_time`; we are most interested in the `test_score`.

Under the hood this does `model.fit` as well as `model.score`, multiple times.

timestamp 11:20 - [Exercise](https://hackmd.io/OffraZphTzmNKFQjRAJhDA?both#Exercise-Recap-fitting-a-scikit-learn-model-on-numerical-data)


Answers:
1: B
2: A, B, C 
3: A, C, E
4: A
5: D

_Break until 11:41_

### Data imputation
Missing values in the target usually leads to removal of those datapoints (predicting values would mean that a prediction is based on another prediction..). 
Missing values in the data can be filled in in various ways; e.g. by taking the overall mean value of that parameter. 
See some resource at the bottom.

### Categorical features
(You can continue in the same notebook, or create a new one.)

```python=
import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census.csv")
adult_census = adult_census.drop(columns="education-num")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns = [target_name])
```

inspecting `data` shows us the data, including a lot of categorical data. Let's look at one in particular:

```python=
data["native-country"].value_counts()
```

We can explore the data types of all columns:
```python=
data.dtypes
```

and use this to select only categorical data:
```python=
from sklearn.compose import make_column_selector as selector

categorical_selector = selector(dtype_include = object)
categorical_columns = categorical_selector(data)
```

ℹ️ See [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes) for what more data types you can use

Inspecting `categorical_columns` shows that we now have a list of column names that are categorical. We can use this list to select only categorical data.

```python=
data_categorical = data[categorical_columns]
```

We can explore the content of each of these categories, e.g.

```python=
data["marital-status"].value_counts()
```

We will now encode it into a useful format:

```python=
from sklearn.preprocessing import OrdinalEncoder
mar_status_column = data[["marital-status"]]
ordinal_encoder = OrdinalEncoder()
mar_status_ordinal = ordinal_encoder.fit_transform(mar_status_column)
```
This transforms the data to an array of floating points. Numerical representation is easier for the model to handle. We can inspect the categories in the encoder with `ordinal_encoder.categories_`.

Let's do this on the entire dataset:
```python=
data_ordinal = ordinal_encoder.fit_transform(data_categorical)
```
The transformation to numbers here also introduces a quandary: what is the meaning of these numbers? There is no inherent value to specific categories, so it makes no sense to replace these categories with ordered numbers. This can depend on the category; some categories do have a semi-applicable order where a transformation to values could make sense.

An ordinal encoder therefore is not the best option here. We will look at a different encoder:
```python=
from sklearn.preprocessing import OneHotEncoder
encoder_honehot = OneHotEncoder(sparse_output=False)
mar_status_onehot = encoder_onehot.fit_transform(mar_status_column)
```
Inspecting `mar_status_onehot` shows a very different array; `OneHot` has encoded each category to a list of 0 and 1 values:

![](https://i.imgur.com/tLxVjwx.png)

Thus; the distance between each category pair is now equal.

```python=
feature_names = encoder_onehot.get_feature_names_out(input_features=["marital-status"])

# turning this into a data frame for inspection purposes
marital_status_onehot = pd.DataFrame(
    mar_status_onehot, columns = feature_names)
```

Let's apply the encoder to the entire dataset
```python=
data_onehot = encoder_onehot.fit_transform(data_categorical)
```

Looking at our differenly encoded data:
```python=
print(f"The original dataset contained {data_categorical.shape[1]} features. After ordinal encoding, the dataset still contains {data_ordinal.shape[1]} features. (because they did not get removed, just replaced by numbers). However, the onehot encoded dataset contains {data_onehot.shape[1]} features: categories are replaced by lists of 1/0 for each instance of a category, multiplying the number of features.")
```

 
The categorical data has 8 features; as does the ordinal.
However, the onehot encoded dataset has 102 features: every category becomes its own feature, and the data consists of 0 or 1 values.

Timestamp 12:36 - [Exercise](https://hackmd.io/OffraZphTzmNKFQjRAJhDA?both#Exercise-The-impact-of-using-integer-encoding-for-with-logistic-regression-breakout-rooms-of-3-4-people)


## 📚 Resources
 
 - how to handle missing data in python https://machinelearningmastery.com/handle-missing-data-python/