![](https://i.imgur.com/iywjz8s.png)


# Day 4 2023-01-30-ds-sklearn Collaborative Document

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
| 09:00 | Welcome and icebreaker                              |
|-------|-----------------------------------------------------|
| 09:15 | Validations & learning curves cont.                      |
|       | Intuition on linear models and decision tree                         |
| 10:15 | Coffee break                                        |
| 10:30 | Your own machine learning project: penguins dataset |
| 11:30 | Coffee break                                        |
| 11:45 | Machine learning best practices                     |
| 12:45 | Wrap-up and feedback                                            |
| 13:00 | END                                                 |

## Ice breaker
### 🎵 What is your favorite background music for working?


## 🔧 Exercises

### Exercise: Train and test SVM classifier (30 min,  4 people / breakout room)

In the following exercise you will:

- Train and test a support vector machine classifier through cross-validation;
- Study the effect of the parameter gamma (one of the parameters controlling under/over-fitting in SVM) using a validation curve;
- Determine the usefulness of adding new samples in the dataset when building a classifier using a learning curve. 

**Use the instructions below; they will guide you through this process.** Consult your notebooks from yesterday's work, or the [collaborative document](https://hackmd.io/vKcCDeEgRxaHZtjBj3LcwQ#Validation-curve) for more help.

We will use a blood transfusion dataset located in `../datasets/blood_transfusion.csv`. We will predict variable `Class.`

1. Start off by creating a predictive pipeline made of:

    * a [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) with default parameter;
    * a [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

    The following script will help you get started:

    ```python=
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    model = make_pipeline(StandardScaler(), SVC())
    ```
    
    Use `sklearn.model_selection.ShuffleSplit` and `sklearn.model_selection.cross_validate` to evaluate the generalization performance of your model.
    
2. Use `learn.model_selection.validation_curve` with hyperparameter `gamma` to look at the test and train errors respective to model complexity. Study the parameter `gamma`. Make sure to keep the default for parameter `scoring` (`scoring="accuracy"`). You can vary `gamma` between 10e-3 and 10e2 by generating samples on a logarithmic scale with the help of

    ```python=
    gammas = np.logspace(-3, 2, num=30)
    param_name = "svc__gamma"
    ```

3. Construct a learning curve (recall `sklearn.model_selection.learning_curve`) to see the effect of adding new samples to the training set. To manipulate the training size you could use:

    ```python=
    train_sizes = np.linspace(0.1, 1, num=10)
    ```

4. Visualize the learning curve and validation curves using `plt.errorbar`.



## 🧠 Collaborative Notes

### Reflections on yesterday's feedback:
- [How to choose scikit-learn models](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) and check the models used by the research domain
- More theory: see [Andrew Ng's Machine Learning Course on Coursera](https://www.coursera.org/specializations/machine-learning-introduction)

What model to choose?
![](https://scikit-learn.org/stable/_static/ml_map.png)

Another approach would be to check what models are in use in your domain. This will give you a baseline from where to choose; using the same model with more data may already give you improved results.

How to tune multiple hyperparameters simultaneously?

You can perform exhaustive search over specified parameters values using `sklearn.model_selection.GridSearchCV` 
See [this](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) documentation for more details. 

### Discussion of the [exercise](https://hackmd.io/b_TK2oZ8SFmahdl8C58ACQ#Exercise-Train-and-test-SVM-classifier-30-min-4-people--breakout-room)

In the model setup, make sure to set `scoring="accuracy"` (or leave it out; "accuracy" is a default setting).

This is the result for plotting accuracy for different values of `gamma`.
![](https://i.imgur.com/jqfgb3B.png)

And the result for the number of samples in the training set:
![](https://i.imgur.com/hplO9pI.png)

### Linear models

<iframe width="740" height="476" src="https://www.youtube.com/embed/ksEGivkPP7I" title="Intuitions on linear models" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### Decision trees

![](https://i.imgur.com/6T2t96e.png)

![](https://i.imgur.com/9FcI8tv.png)


<iframe width="740" height="476" src="https://www.youtube.com/embed/1kIHC1O_drM" title="Intuitions on tree-based models" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


### ❓❗️ Q&A

Q: Can you say something about running out of memory? Some of my data is really large, which can be a problem. Can we perhaps do things step-by-step so I do not need to load all of the data at the same time?

A: In Deep Learning you can use batches. In Machine Learning it is more common to use small datasets; you could perhaps split the data beforehand, as you may not need more data to improve the model at a certain point. Perhaps you can preprocess the data and reduce the number of features, e.g. with a principal component analysis.

Q: How do we know what the best value is for a specific parameter?

A: By getting experience, you will get a feel for the parameter space. Start with default values and explore the space around that; it can also change based on the data, so do a grid search to explore many different combinations of parameters.

### Best Practices

An overview of all the problems in the universe 🙃
![](https://i.imgur.com/pn3o8PF.png)

In Machine Learning, the machine learns from experience encoded in data. In Deep Learning, an artificial neural network is used. There are more layers in the architecture than in traditional machine learning. The limit of AI is: if a computer cannot solve the problem. For example, data (of whatever kind) needs to be encoded in 0 and 1 to allow a computer to work with it.

Machine learning models are tools; learning how to apply them may be more useful than understanding them completely.

[Slides for ML best practices](https://esciencecenter-digital-skills.github.io/SICSS-odissei-machine-learning/4-Best-practice.html#/title-slide)

Ask yourself:
- What is your scientific problem?
- Can this scientific problem be transformed to a machine learning problem?
  - Keep the problem simple; if not, decompose it
- Do you really have to learn machine learning?
- What is the goal of your ML project?
- Do you have enough high quality data?
- How do you measure the model performance?
  - First design and implement metrics
- Do you have enough infrastructure (enough computing power and other resources, eg)
- Are there any risks related to privacy and ethics?
  - [DEON ethics checklist](https://drivendata.co/open-source/deon-ethics-checklist/)
  - [Check out Fairlearn.org](https://fairlearn.org/).

Q: In evaluating a model, when do we choose error, and when do we choose accuracy?
A: A lot depends on your data and question; in this case, the task is also relevant. Accuracy is a metric that is suitable for classification, whereas error is relevant for a regression problem.

Workflow or pipeline? Having a bad workflow is better than nothing, so make one and then optimize it!

Data processing (identification, cleaning, labelling, etc...) is more than 75% of your time in a machine learning project. Be patient with data engineering, and make sure to split the data into training/validation/test, and do not mix these sets.
Good data often leads to good models, so this is time well spent.

When designing a model, start with a baseline performance (either human performance, or a model already in use). Keep the model as simple as possible, and be patient with training; tuning parameters will take time.




## 🗣 Feedback
👉 [**Please fill out the post-workshop survey!**](https://www.surveymonkey.com/r/TZCGYC8)


## 📚 Resources

- [An intuitive guide to understanding Support Vector Machines (SVM)](https://www.newtechdojo.com/understanding-support-vector-machines-svm/)
- [Not that intuitive intro to SVM](https://scikit-learn.org/stable/modules/svm.html#support-vector-machines)
- [Hyperparameter search for multiple hyperparameters at once](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

### Further studying:
- ['Machine learning yearning' by Andrew Ng](https://github.com/ajaymache/machine-learning-yearning). Small one-page practical tips, about real-world machine learning setting that you often don't learn about in courses. For example: 'How to decide how to include inconsistent data?'.
- [Andrew Ng's Machine Learning Course on Coursera](https://www.coursera.org/specializations/machine-learning-introduction) Really good course! Will give you a deep understanding of machine learning.
- [Fast.ai](https://fast.ai)
