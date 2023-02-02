![](https://i.imgur.com/iywjz8s.png)


# Day 1 2023-01-30-ds-sklearn Collaborative Document

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

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city


## 🗓️ Agenda
| 09:00 | Welcome and icebreaker                         |
|-------|------------------------------------------------|
| 09:15 | Machine learning concepts                      |
| 10:15 | Coffee break                                   |
| 10:30 | Tabular data exploration                       |
| 11:30 | Coffee break                                   |
| 11:45 | Fitting a scikit-learn model on numerical data |
| 12:45 | Wrap-up                                        |
| 13:00 | END  |

## 🔧 Exercises

### Exercise: Data exploration (~15min,  4 people / breakout room)

Imagine we are interested in predicting penguins species based on two of their body measurements: culmen length and culmen depth. First we want to do some data exploration to get a feel for the data.

The data is located in `../datasets/penguins_classification.csv`.

Load the data with Python and try to answer the following questions:
1. How many features are numerical? How many features are categorical?
2. What are the different penguins species available in the dataset and how many samples of each species are there?
3. Plot histograms for the numerical features
4. Plot features distribution for each class (Hint: use `seaborn.pairplot`).
5. Looking at the distributions you got, how hard do you think it will be to classify the penguins only using "culmen depth" and "culmen length"?


### 📝 Exercise (in breakout rooms): Adapting your first model
The goal of this exercise is to fit a similar model as we just did to get familiar with manipulating scikit-learn objects and in particular the .fit/.predict/.score API.

Before we used `model = KNeighborsClassifier()`. All scikit-learn models can be created without arguments. This is convenient because it means that you don’t need to understand the full details of a model before starting to use it.

One of the KNeighborsClassifier parameters is n_neighbors. It controls the number of neighbors we are going to use to make a prediction for a new data point.

#### 1. What is the default value of the n_neighbors parameter? 
Hint: Look at the documentation on the scikit-learn website or directly access the description inside your notebook by running the following cell. This will open a pager pointing to the documentation.
```python
from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier?
```

#### 2. Create a KNeighborsClassifier model with n_neighbors=50
a. Fit this model on the data and target loaded above
b. Use your model to make predictions on the first 10 data points inside the data. Do they match the actual target values?
c. Compute the accuracy on the training data.
d. Now load the test data from "../datasets/adult-census-numeric-test.csv" and compute the accuracy on the test data.


## 🧠 Collaborative Notes
### Icebreaker :ice_cream: 
Rate between 0-100% how you feel today and elaborate. Do you have 10 deadlines and are you super stressed, still recovering from a rough weekend, or super highly motivated for this week's workshop?

### What is machine learning?
* Famous examples: predicting which type of iris (flower) it is, based on physical characteristics; predicting people's income-group based on personal data.
* Is somehow related to "expert knowledge", but without using the expert.
* Generalizing != memorizing.
* Generalization: learning something about a dataset that you can then apply to an unseed dataset (many sources of variability, some useful some less so (noise)).
* Memorizing: store all information about all known individuals; given a new individual make a prediction based on closest match in dataase.
* "test data" != "train data": train data is used to "learn", test data is used to check how well the learning worked.
* Machine learning workflow: use train data to "learn" certain characteristics of setosa or other flowers, then use this information to predict an unseen data point.
* Data matrix: rows contain individual observations, columns contain features (petal size, etc) or targets (flower type).
* Supervised machine learning uses a data matrix (*X*) with *n* observations, having a target *y*; goal is to predict *y*.
* Unsupervised machine learning does not have a target (*y*) in the data matrix, but tries to predict *y* values from scratch.
* Regression vs classification: regression is continuous target values (e.g. income); classification has discrete target values (e.g. iris type).
* Take home message: **Machine leaning is using data to extract rules that generalize to new observations**.

### Setting up your environment and open notebook
* After following the setup instrutions (https://tinyurl.com/setup-instructions), make a new folder in your local repository called "lesson", and enter the folder.
* Activate python environment: `conda activate scikit-learn-course`
* Create a new notebook from jupyter notebook, or jupyter lab, or use a VS Code notebook, or whatever other way you are used to working with notebooks.

### Tabular data exploration
* **Always take a look at the data before working with it**, to get some general insights about it.
* Today, we will be working with public/open source US census data.
* All the data we will use is already loaded into the repo
* Take a look at the top 5 data points using the following commands: 
```python
import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census.csv")
adult_census.head()
````
* The target column is final one: "class". All other columns are features (synonymous to variables or covariates)
* Set the class column as your target and check how many types of values it has:
```python
target_column = "class"
adult_census[target_column].value_counts()
```
* You will see that there are 2 potential values: "<=50K" and ">50K" --> this is a binary classification problem.
* Also, you can see that it is an "unbalanced" data set: there are ~3 times as many data points in the "<=50K" than the ">50K" group.
* Some features are numerical (e.g. "age"), others are categorical (e.g. "education")
* Extract the numerical and cateforical columns:
```python
numerical_columns = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]
all_columns = numerical_columns + categorical_columns + [target_column]
adult_census = adult_census[all_columns]
```
* check how many samples are present in the data set

``` python
print(
    f"The data set contains {adult_census.shape[0]} samples and"
    f"{adult_census.shape[1]} columns"
)
print("
    f"The data set contains {adult_census.shape[1] - 1} features"
)
```
* Visual inspection of the data of numerical features using histograms
```python=
_ = adult_census.hist(figsize=(20,14))
```
* Categorical features can be explored using the `value_counts` command
``` python
adult_census["sex"].value_counts()
```
* there seem to be 2 similar features, going by feature name: education and education-num ; let's copare 
``` python
print(adult_census["education"].value_counts())
print(adult_census["education-num"].value_counts()))
```
* Crosstab allows you to make a matrix showing how the two compare to each other
```python=
pd.crosstab(index=adult_census["education"], columns = adult_census["education-num"])
```
* You will notice that the two columns are perfectly correlated (there is a 1-1 mapping between them)
* Pairplot can be used to look at relationships between different numerical features

```python=
import seaborn as sns
```

```python
n_samples_to_plot = 5000
columns = ["age", "education-num", "hours-per-week"]

_ = sns.pairplot(
    data = adult_census[:n_samples_to_plot],
    vars = columns,
    hue = target_column, # make sure that target_column is a string (and not a list with 1 entry)
    plot_kws = {"alpha": 0.2},
    height = 3,
    diag_kind = "hist",
    diag_kws = {"bins": 30}
)
```
* Looking at the age vs hours-per-week plot (left bottom), try to think of some "rules" that would help you to distinguish between the 2 classes (lower and higher income)

### Fitting a scikit-learn model on numerical data
* Open a new notebook, give it an appropriate name
* For now, we real only use numerical features, because it's easier. We will get back on how to deal with categorical data tomorrow
* We want to learn how to use the scikit-learn API
```markdown=
# First model with sci-kit learn
```

```python=
import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census-numerical.csv")
adult_cen
```
* Now let's separate the target from the features
```python=
target_name = 'class'
target = adult_census[target_name]
target
```
* Then remove the target from the feature data
```python
data = adult_census.drop(columns=[target_name])
data.head()
# data = features = X; these are alymous
```
* K nearest neighbour model predicts which category a sample belongs to based on it's k nearest neighbours
```python=
from sklearn import KNeighborClassifier
```
```python
model = KNeighborClassifier()
_ = model.fit(data, target)
# _ in python is a placeholder for a 
```
```python
target_predicted = model.predict(data)
```
* let's look at the first 5 predictions
```python=
target_predicted[:5]
```
* and compare it to the ground truth
```python=
target[:5]
```
* now, let's calculate the overall accuracy of the model
```python=
(target == target_predicted).mean()
```
* in machine learning we want a meaningful prediction for data we haven't seen. The accuracy above only tells us something about the training data itself.
* Now let's look at some unseen data (data that the model has not used to learn)
```python=
adult_census_test = pd.read('../datasets/adult-census-numeric-test.csv')
adult_census_test.head()
```
```python=
target_test = adult_census_test[target_name]
data_test = adult_census_test.drop(columns=[target_name])
data_test.shape
```
* Now let's compute a "score" for this model, i.e. the accuracy on unseen data
```python=
accuracy = model.score(data_test, target_test)
accuracy
```

Partial dependence plots (PDP) can be used to visualize and analyze interaction between the target response and a set of input features of interest. For more information please refer to the documentation: 

https://scikit-learn.org/stable/modules/partial_dependence.html

* Recap:
* we fitted our first model
* we evaluated it's accuracu on unseen data
* and we introduced the scikit-learn API 

### For tomorrow:
* Welcome to join at 8:45, start at 9:00
* Same zoom link!

## 📚 Resources
* Who we are: www.esciencecenter.nl
* Digital skills workshops: www.esciencecenter.nl/digital-skills
* Network of Dutch research software engineers: https://nl-rse.org/
* improve the fairness of your model: www.fairlearn.org