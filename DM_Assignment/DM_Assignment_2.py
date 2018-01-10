# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 07:31:01 2017

@author: Atul Singh
@Id : 2016HT12566@wilp.bits-pilani.ac.in
@Subject : Data Mining
@Assignment : 1
@Question : 1

"""

# importing libraries
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import precision_score, recall_score


# reading the data
data = pd.read_csv("data/dataset.csv", header=None, delimiter=' ')

# splitting the data into features X and target y 
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# instantiate KFold
kf = KFold(n_splits=10, random_state=3)


# implementing the naive_bayes algorithm 
clf_naive = GaussianNB()
y_pred = cross_val_predict(clf_naive, X, y, cv=kf, n_jobs=10)

print("---------NaiveBase Algorithm Scores------------");
print("precision_score: ", precision_score(y, y_pred))
print("recall_score: ", recall_score(y, y_pred))
print("")

# implementing the naive_bayes algorithm 
clf_tree = RandomForestClassifier()
y_pred = cross_val_predict(clf_tree, X, y, cv=kf, n_jobs=10)

print("---------RandomForestClassifier Algorithm Scores------------");
print("precision_score: ", precision_score(y, y_pred))
print("recall_score: ", recall_score(y, y_pred))
print("")

# implementing the AdaBoostClassifier algorithm 
clf_ada = AdaBoostClassifier(base_estimator=clf_tree, algorithm='SAMME')
y_pred = cross_val_predict(clf_ada, X, y, cv=kf, n_jobs=10)

print("---------AdaBoostClassifier Algorithm Scores------------");
print("precision_score: ", precision_score(y, y_pred))
print("recall_score: ", recall_score(y, y_pred))
print("")


# implementing the SVM algorithm 
clf_svc = SVC(kernel='linear', C=1)
y_pred = cross_val_predict(clf_svc, X, y, cv=kf, n_jobs=10)

print("---------SVM Algorithm Scores------------");
print("precision_score: ", precision_score(y, y_pred))
print("recall_score: ", recall_score(y, y_pred))
print("")