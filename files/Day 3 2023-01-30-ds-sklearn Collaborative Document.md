![](https://i.imgur.com/iywjz8s.png)


# Day 3 2023-01-30-ds-sklearn Collaborative Document

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



## 🗓️ Agenda
| 09:00 | Welcome and icebreaker         |
|-------|--------------------------------|
| 09:15 | Categorical data, cont.        |
|       | Overfitting and underfitting   |
| 10:15 | Coffee break                   |
| 10:30 | Validation and learning curves |
| 11:30 | Coffee break                   |
| 11:45 | Bias versus variance trade-off |
| 12:45 | Wrap-up                        |
| 13:00 | END                            |

## 🔧 Exercises

### Exercise: The impact of using integer encoding with logistic regression (breakout rooms of 3-4 people):
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


### Exercise: overfitting and underfitting:

#### 1: A model that is underfitting:

a) is too complex and thus highly flexible
**b)** is too constrained and thus limited by its expressivity
**c)** often makes prediction errors, even on training samples
d) focuses too much on noisy details of the training set

Select all answers that apply

#### 2: A model that is overfitting:

**a)** is too complex and thus highly flexible
b) is too constrained and thus limited by its expressivity
c) often makes prediction errors, even on training samples
**d)** focuses too much on noisy details of the training set

Select all answers that apply



## 🧠 Collaborative Notes

re: [exercise:](https://hackmd.io/vKcCDeEgRxaHZtjBj3LcwQ#Exercise-The-impact-of-using-integer-encoding-with-logistic-regression-breakout-rooms-of-3-4-people)
Categorical values with low incidences in the entire dataset will possibly be missing from the training dataset. This is where the `handle_unknown` argument comes in.

Choosing an encoder:
In general OneHotEncoder is the encoding strategy used when the downstream models are linear models while OrdinalEncoder is often a good strategy with tree-based models.

Q: is there no machine learning without numbers? Are numbers always needed?
A: Yes, numbers are necessary for machine learning. The choice of encoder defines whether order is given to categories (OrdinalEncoder), nor not (e.g. OneHotEncoder). Even if this is not logical (e.g. with a categorical variable that does not have order) it can still be done: it won't hurt your model and will speed up computational time.

Q: When do you stop creating new models to see if you can improve it?
A: With large enough datasets, and especially with cross-validation, a model that gives good results will not do so by chance. If you get good results, this means it is a good model.
A risk with "improving" a model is that it could be tuned to the test set. 


### Combining numerical and categorical data
```python=
import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census.csv")
data = adult_census.drop(columns=[target_name, "education-num"])

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name])
```

```python=
from sklearn.compose import make_column_selector as selector

categorical_selector = selector(dtype_include = object)
numerical_selector = selector(dtype_exclude = object)

numerical_columns = numerical_selector(data)
catetegorical_columns = categorical_selector(data)

print('numerical features', numerical_columns)
print('categorical features', categorical_columns)
```

```python=
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([
    ('onehot_encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)
])
```

```python=
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter = 500))
```
Calling `model` shows the pipeline visually:
![](https://i.imgur.com/cx3LZW2.png)

```python=
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size = 0.75, random_state=22)

_ = model.fit(data_train, target_train)

pred = model.predict(data_test)
```

Taking a look at `pred[:10]` shows that indeed predictions are made for <=50K and >50K.

We can compare the results to look at accuracy in various ways
```python=
target_test[:20] == pred[:20] # an overview of the first 20 labels

(pred == target_test).mean() # a summary of the accuracy overall
```
### ChatGPT
Some inspiration...
![](https://i.imgur.com/bDvcipX.png)

### Overfitting and underfitting

<iframe width="740" height="476" src="https://www.youtube.com/embed/xErJGDwWqys" title="Overfitting and underfitting" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### Comparing train and test errors

<iframe width="740" height="476" src="https://www.youtube.com/embed/9uS4sE-UTm0" title="Comparing train and test errors" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

We will be using a dataset provided by `sklearn` for the next part.
```python=
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame = True)
```
Calling `housing.data` shows our data as a dataframe. The `housing.target` shows the target variable.

```python=
data, target = housing.data, housing.target
target *= 100 # multiplies the values in the target by 100.
```

Choose a model type:
```python=
from sklearn.tree import DecisionTreeRegressor
```
Use `?DecisionTreeRegressor` to take a look at the different parameters:

![](https://i.imgur.com/ztn4sK9.png)

Or look at sklearn's website at the API documentation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

```python=
regressor = DecisionTreeRegressor() # initialize object with default parameter values
```
Take a look at the [user guide](https://scikit-learn.org/stable/modules/tree.html) for more information about how decision trees work.

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dtc_002.png)


### Validation curve

```python=
import pandas as pd
from sklearn.model_selection import cross_validate, ShuffleSplit

cv = ShuffleSplit(n_splits = 30, test_size = 0.2)
cv_results = cross_validate(regressor,
                           data,
                           target,
                           cv = cv,
                           scoring = "neg_mean_absolute_error",
                           return_train_score = True,
                           n_jobs = 2)

cv_results = pd.DataFrame(cv_results)
```
Calling `cv_results` shows the results for all different runs of the model:
![](https://i.imgur.com/j6CugMA.png)


We can format the scores slightly differently:
```python=
scores = pd.DataFrame()
scores[["train_error", "test_error"]] = - cv_results[["train_score", "test_score"]]
```
Call `scores` to look at the scores formatted this way:
![](https://i.imgur.com/GA5sPLi.png)

```python=
import matplotlib.pypot as plt

scores.plot.hist(bins = 50)
```

![](https://i.imgur.com/BSIj2wx.png)

There is a huge difference between the train and test error at this point.

```python=
from sklearn.model_selection import validation_curve

max_depth = [1,5,10,15,20,25]

train_scores, test_scores = validation_curve(
    regressor,
    data,
    target,
    param_name = "max_depth",
    param_range = max_depth,
    cv = cv,
    scoring = "neg_mean_absolute_error",
    n_jobs = 2
)
```
Call `validation_curve?` or check the [api documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve) for more information about this method.

```python=
# reformatting results
train_errors, test_errors = -train_scores, -test_scores

# visualize them
## Train errors
plt.errorbar(max_depth,
             train_errors.mean(axis=1),
             yerr = train_errors.std(axis=1),
            label = "Train error")

## Test errors
plt.errorbar(max_depth,
             test_errors.mean(axis=1),
             yerr = test_errors.std(axis=1),
           label = "Test error")

plt.legend()
```
![](https://i.imgur.com/vml2xSv.png)


### Learning curve

```python=
import numpy as np
train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)
```
![](https://i.imgur.com/m9Kx3hO.png)


```python=
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2)
```

```python=
from sklearn.model_selection import learning_curve

results = learning_curve(regressor,
                         data,
                         target,
                         train_sizes=train_sizes,
                         cv=cv,
                         scoring="neg_mean_absolute_error",
                         n_jobs=2
)
```

Extracting and visualizing the results:
```python=
train_size, train_scores, test_scores = results[:3]
train_errors, test_errors = -train_scores, -test_scores

## Train errors
plt.errorbar(train_size,
             train_errors.mean(axis=1),
             yerr = train_errors.std(axis=1),
            label = "Train error")

## Test errors
plt.errorbar(train_size,
             test_errors.mean(axis=1),
             yerr = test_errors.std(axis=1),
            label = "Test error")


plt.legend()
plt.xscale("log")
plt.xlabel("Number of samples in the training set")
plt.ylabel("Mean absolute error (positive), in 1000 USD")

```
No log scale:
![](https://i.imgur.com/LvuQsyP.png)

Log scale:
![](https://i.imgur.com/BMC47Gl.png)

Our goal is to find the plateau. That is not yet reached here. This could be because our total dataset is simply too small.



## 📚 Resources
[SciKit Learn User Guide](https://scikit-learn.org/stable/user_guide.html#user-guide)