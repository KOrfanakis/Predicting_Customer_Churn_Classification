# Predicting Customer Churn

<br>

**Table of Contents**:

<!--ts-->
* [Motivation](#motivation)
* [Business Objective :star:](#business-objective)
* [Data](#data)
* [Results :star:](#results)
* [References](#references)
<!--te-->

<br>

# Motivation

Imagine that we are hired as data scientists by a major commercial bank with several branches across Europe. Its finance division decided that the bank could benefit from reducing the number of customers that churn, i.e. stop using its services. Therefore, they compiled a dataset containing information on the bank’s current and past customers, such as their credit score, age, account balance, etc. Most importantly, the dataset contains a column stating whether the customer has churned or not.

We are assigned to use the dataset to gain insights into why customers are leaving and build a Machine Learning (ML) model that predicts whether a customer is likely to churn or not.

For more information on customer churn and why it is important, please refer to the introductory section of the notebook.


<br>

# Business Objective

The first question we need to ask our employers is how the bank expects to benefit from our model.

Our employers answer that they will use our model’s output and create a list of customers more likely to churn. Then, the bank will offer additional services and special offers to those customers in an effort to increase customer satisfaction. The total amount will be equal to £1,000 per customer. The bank estimates that the gain from each customer retained will be approximately equal to £5,000 per customer, i.e. five times the initial investment.  


<br>

# Data

The dataset can be retrieved from Kaggle, following [this link](https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers). The dataset consists of 14 columns (synonyms: features or attributes) and 10K rows (synonyms: instances or samples). The last column, ‘Exited’, is the target variable and indicates whether the customer has churned (0 = No, 1 = Yes). The meaning of the rest of the features can be easily inferred from their name.

<br>

# Results

We used eight classification algorithms to build our models. The algorithms are:

1) [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), <br>
2) [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), <br>
3) [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), <br>
4) [Balanced Random Forest Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html), <br>
5) [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), <br>
6) [Xtreme Gradient Boosting Classifier](https://xgboost.readthedocs.io/en/latest/),
7) [Light Gradient Boosting Machine](https://lightgbm.readthedocs.io/en/latest/), and <br>
8) An ensemble of the above classifiers.

After establishing a baseline, hyperparameter tuning was used to calculate their optimal performance on the training set (consisting of 8,000 customers). The best-performing model with an **AUC score ~0.865** is the optimised model based on **LGBM classifier**.

The model was tested on unseen data, using customer instances that were not used for training. The test sample consists of 2,000 customers, mainting the ratio of churned/remained customers of the training set. Using the assumptions mentioned in the Business Objective section, the bank would make a **total profit of £900,000**! As a comparison, our baseline model (based on the Gaussian Naïve Bayes algorithm) yields a total profit of £456,000.

<br>

# References

A complete list of references is provided at the end of the notebook.

<br>
