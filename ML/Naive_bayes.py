import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X = X[:100,:]
y = y[:100]

K = 5

folds = np.random.choice(K, size=X.shape[0], replace=True)

# for i in range(K):
#     X_test = X[folds == i]
#     y_test = y[folds == i]

#     X_train = X[folds != i]
#     y_train = y[folds != i]

#     print(f"Fold {i}:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    
### Naive Bayes

X_test = X[folds == 0]
y_test = y[folds == 0]

X_train = X[folds != 0]
y_train = y[folds != 0]

import pandas as pd

prior_probs = pd.Series(y_train).value_counts(normalize=True)


def PDF(x, mean, var):
    first = 1 / np.sqrt(2 * np.pi * var)
    second = np.exp(- (x - mean) ** 2 / (2 * var))
    
    return first * second

mean_class0 = X_train[y_train == 0].mean(axis=0)
var_class0 = X_train[y_train == 0].var(axis=0)

mean_class1 = X_train[y_train == 1].mean(axis=0)
var_class1 = X_train[y_train == 1].var(axis=0)

probs = []
y_pred = []
for i in range(X_test.shape[0]):
    likelihood_class0 = PDF(X_test[i, :], mean_class0, var_class0)
    likelihood_product_class0 = likelihood_class0.prod()

    likelihood_class1 = PDF(X_test[i, :], mean_class1, var_class1)
    likelihood_product_class1 = likelihood_class1.prod()

    posterior_class0 = likelihood_product_class0 * prior_probs[0]
    posterior_class1 = likelihood_product_class1 * prior_probs[1]

    denom = posterior_class0 + posterior_class1
    posterior_class0 /= denom
    posterior_class1 /= denom
    
    probs.append([posterior_class0, posterior_class1])
    y_pred.append(int(np.argmax([posterior_class0, posterior_class1])))  
    
    
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
tp = cm[0,0]
tn = cm[1,1]
fp = cm[1,0]
fn = cm[0,1]

accuaracy = (tp + tn) / cm.sum()
precision = tp / (tp + fp)
recall = tp / (tp + fn)