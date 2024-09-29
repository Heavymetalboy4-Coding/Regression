import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
import pandas as pd
from sklearn.linear_model import LogisticRegression

insurance = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/AI Model/insurance_data.csv")

X = insurance[["age"]]
y = insurance.bought_insurance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_test)
""""plt.scatter(insurance.age, insurance.bought_insurance, marker="+",  color = "red", label = "insurance")
plt.show()"""
model = LogisticRegression()
model.fit(X_train, y_train)
ypredict = model.predict(X_test)
print(ypredict)
print(X_test)
print(model.score(X_test, y_test))
print(model.predict_proba(X_test))

print(model.predict([[5000]]))
print("Coefficent is:", model.coef_)
print("Intercept is: ", model.intercept_)

#applying the sigmoid function
import math 
def sigmoid(x):
    return 1/(1+math.exp(-x))
def prediction(age):
    z = 0.042* age - 1.53
    y = sigmoid(z)
    return y
print(prediction(95))