import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


data = load_iris()
X = data.data
y = data.target

# no of folds
K = 5

folds = np.random.choice(K, size=X.shape[0], replace=True)

X_test = X[folds == 0]
y_test = y[folds == 0]

X_train = X[folds != 0]
y_train = y[folds != 0]

# Build Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier 
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
# tp = cm[0,0]
# tn = cm[1,1]
# fp = cm[1,0]
# fn = cm[0,1]

# accuaracy = (tp + tn) / cm.sum()
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


########################################################
# Build Decision Tree Classifier with hyperparameters

from sklearn.tree import DecisionTreeClassifier 
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    splitter='best',
    min_samples_split=10,
    min_samples_leaf=2
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
########################################################

#parameter tuning

accuracy_dict = dict()
for depth in range(1, 11):
    for split in range(2,10):
        for leaf in range(1,5):
            clf = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=split,
                min_samples_leaf=leaf
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_dict[(depth, split, leaf)] = accuracy
            # print(f"Depth: {depth}, Split: {split}, Leaf: {leaf} => Accuracy: {accuracy:.4f}")


#####################################################

clf = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# plot decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(clf, 
          feature_names=data.feature_names, 
          class_names=data.target_names, 
          filled=True,
          rounded=True,
          fontsize=10)

plt.show()