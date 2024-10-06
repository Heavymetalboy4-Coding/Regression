import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
titanic = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/AI Model/titanic.csv")
print(titanic.isnull().sum())
#taking care of the null values
print(titanic["Age"].median(skipna= True))
titanic["Age"].fillna(titanic["Age"].median(skipna= True),inplace=True)
print(titanic.isnull().sum())
#print(titanic.head())

#Dropping the unnecesary data
titanic.drop("PassengerId", axis= 1, inplace=True)
titanic.drop("Name", axis= 1, inplace=True)
titanic.drop("Ticket", axis= 1, inplace=True)
titanic.drop("SibSp", axis= 1, inplace=True)
titanic.drop("Parch", axis= 1, inplace=True)
titanic.drop("Cabin", axis= 1, inplace=True)

#print(titanic.head())

preprocessing_titanic = preprocessing.LabelEncoder()
titanic["Sex"] = preprocessing_titanic.fit_transform(titanic["Sex"])
titanic["Embarked"] = preprocessing_titanic.fit_transform(titanic["Sex"])
print(titanic.head())
X = titanic[["Pclass", "Sex", "Age", "Embarked"]]
y = titanic["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
model = LogisticRegression()
model.fit(X_train, y_train)
ypredict = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
matrix = confusion_matrix(y_test, ypredict)
import seaborn as sns

sns.heatmap(matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrix")
plt.legend()
plt.show()
print(classification_report(y_test, ypredict))