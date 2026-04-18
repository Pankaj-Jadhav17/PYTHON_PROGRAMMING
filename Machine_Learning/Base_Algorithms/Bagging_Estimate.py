import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


data = load_iris()
X = data.data
y = data.target

X = X[:100, :]
y = y[:100]

# no of folds
K = 5

folds = np.random.choice(K, size=X.shape[0], replace=True)

X_test = X[folds == 0]
y_test = y[folds == 0]

X_train = X[folds != 0]
y_train = y[folds != 0]

# Bagging model

from sklearn.ensemble import BaggingClassifier

# default base estimator is Decision Tree with default hyperparameters

clf = BaggingClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# get performance metrics
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
tp = cm[0,0]
tn = cm[1,1]
fp = cm[1,0]
fn = cm[0,1]

accuaracy = (tp + tn) / cm.sum()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

#################################################################

# base model is Decision Tree with max_depth=3

from sklearn.tree import DecisionTreeClassifier

base_estimator = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    splitter='best')

clf = BaggingClassifier(estimator=base_estimator, 
                        n_estimators=100)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
tp = cm[0,0]
tn = cm[1,1]
fp = cm[1,0]
fn = cm[0,1]

accuaracy = (tp + tn) / cm.sum()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)


###########################################################################

# base model is KNN with k=3

from sklearn.neighbors import KNeighborsClassifier
base_estimator = KNeighborsClassifier(n_neighbors=3)

clf = BaggingClassifier(estimator=base_estimator, 
                        n_estimators=100)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
tp = cm[0,0]
tn = cm[1,1]
fp = cm[1,0]
fn = cm[0,1]

accuaracy = (tp + tn) / cm.sum()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

###########################################################################

mcc_formula = (tp*tn - fp*fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)