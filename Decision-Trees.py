import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
import pandas as pd
import seaborn as sns

data = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/KNN/car_decTree.csv")
data.columns = ("sales", "maintainence", "doors", "passengers", "boot capacity", "safety", "class")
print(data.head())
print(data.describe())
print(data.info())
print(data["class"].value_counts())
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data["sales"] = encoder.fit_transform(data["sales"])
data["maint"] = encoder.fit_transform(data["sales"])
data["sales"] = encoder.fit_transform(data["sales"])

X = data[["sales", "maintainence", "doors", "passengers", "boot capacity", "safety"]]
y = data["class"]
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


encoder.fit_transform(X_train)
encoder.fit_transform(y_train)
encoder.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier
model_classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
model_classifier.fit(X_train, y_train)
yp = model_classifier.predict(X_test)
encoder.fit_transform(y_test)

from sklearn.metrics import classification_report,confusion_matrix
matrix = confusion_matrix(y_test, yp)
sns.heatmap(matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrix")
#plt.legend()
plt.show()
print(classification_report(y_test, yp))