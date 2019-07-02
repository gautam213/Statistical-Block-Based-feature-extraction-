# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:24:50 2019

@author: akash
"""

from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

print("Fitting the classifier to the training set")
param_grid = {'C': [1,10,20,100],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf'),
                   param_grid, cv=5)
clf = clf.fit(X_train, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)
y_pred=clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("TEST:",clf.score(X_test,y_test))
print("TRAIN:",clf.score(X_train,y_train))