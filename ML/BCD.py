import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import Binarizer

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================
# 1️⃣ Gaussian Naive Bayes (Normal Distribution)
# ============================================

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_g = gnb.predict(X_test)

print("========== Gaussian NB ==========")
print("Accuracy:", accuracy_score(y_test, y_pred_g))
print(classification_report(y_test, y_pred_g))

cm_g = confusion_matrix(y_test, y_pred_g)
sns.heatmap(cm_g, annot=True, fmt="d")
plt.title("Gaussian NB Confusion Matrix")
plt.show()

# ============================================
# 2️⃣ Bernoulli Naive Bayes
# ============================================

# Convert data to binary
binarizer = Binarizer(threshold=X.mean())
X_bin = binarizer.fit_transform(X)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bin, y, test_size=0.3, random_state=42
)

bnb = BernoulliNB()
bnb.fit(X_train_b, y_train_b)
y_pred_b = bnb.predict(X_test_b)

print("========== Bernoulli NB ==========")
print("Accuracy:", accuracy_score(y_test_b, y_pred_b))
print(classification_report(y_test_b, y_pred_b))

cm_b = confusion_matrix(y_test_b, y_pred_b)
sns.heatmap(cm_b, annot=True, fmt="d")
plt.title("Bernoulli NB Confusion Matrix")
plt.show()