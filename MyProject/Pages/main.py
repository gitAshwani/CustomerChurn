# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:51:56 2021

@author: ssripada1
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltaccuracy_score
import PyALE
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import PartialDependenceDisplay
import shap
from sklearn.inspection import plot_partial_dependence
shap.initjs()


    
        
train = pd.read_csv(".Data/churn_train.csv")
test = pd.read_csv(".Data/churn_test.csv")

train = train.drop(columns=["state"])
train = train.drop(columns=["area_code"])
test = test.drop(columns=["state","area_code"])

train.churn.replace(('yes', 'no'), (1, 0), inplace=True)
train.international_plan.replace(('yes', 'no'), (1, 0), inplace=True)
train.voice_mail_plan.replace(('yes', 'no'), (1, 0), inplace=True)

test.international_plan.replace(('yes', 'no'), (1, 0), inplace=True)
test.voice_mail_plan.replace(('yes', 'no'), (1, 0), inplace=True)


# features
selected_features = train.columns[0: -1]

# target variable
target = "churn"


# split data
global X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test = train_test_split(train[selected_features],train[target], test_size=0.30, random_state = 1)

global logistic, random_forest
logistic = LogisticRegression()
random_forest = RandomForestClassifier()
decision_tree = DecisionTreeClassifier()

neural_net = MLPClassifier()


models = {
    "decision tree" : decision_tree, 
    "logistic" : logistic,
    "neural_net" : neural_net,
    "random_forest": random_forest
}

#fitting the models
for model in models:
    models[model].fit(X_train,y_train)
    print(f"{model} has been trained successfully")
	
# store training performance
performances_training = {}


for model in models:
    predictions   = models[model].predict(X_train)
    probabilities = pd.DataFrame(models[model].predict_proba(X_train))[1]
    accuracy      = accuracy_score(y_train,predictions)
#     auc           = roc_auc_score(np.array(y_train),np.array(probabilities),multi_class = 'ovr')
    
    performances_training[model] = {"Accuracy":accuracy}
    pd.DataFrame(performances_training)



# store testing performance
performances_test = {}

for model in models:
    predictions   = models[model].predict(X_test)
    probabilities = pd.DataFrame(models[model].predict_proba(X_test))[1]
    accuracy      = accuracy_score(y_test,predictions)
    
    performances_test[model] = {"Accuracy":accuracy}   	
    pd.DataFrame(performances_test)
    
    
