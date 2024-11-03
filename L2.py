import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
import pandas as pd
import seaborn as sns

data = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/KNN/car_data.csv")
print(data.head())

X = data[["User ID", "Age", "AnnualSalary"]]
y = data["Purchased"]
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler,LabelEncoder
scaling = StandardScaler()
scaling.fit_transform(X_train)
encoder = LabelEncoder()
encoder.fit_transform(y_train)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

scaling.transform(X_test)
y_pred = classifier.predict(X_test)

encoder.transform(y_test)

from sklearn.metrics import classification_report,confusion_matrix

matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrix")
#plt.legend()
plt.show()
print(classification_report(y_test, y_pred))