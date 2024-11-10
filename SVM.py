from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/KNN/iris.xlsx")
iris.columns = ["sepal-length", "sepal-width", "petal-length", "petal-width", "species"]
target = iris["species"]
x = iris["sepal-length"]
y = iris["petal-length"]
setosa_x = x[:50]
setosa_y = y[:50]
versicolor_x = x[51:100]
versicolor_y = y[51:100]

plt.figure(figsize = (8,6))
plt.scatter(setosa_x, setosa_y, marker = "+", color = "red")
plt.scatter(versicolor_x, versicolor_y, marker = "*", color = "blue")
plt.show()