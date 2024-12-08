import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

titanic = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/KNN/titanic.csv")
print(titanic.isnull().sum())
titanic = titanic.drop(columns=["Name"])
encoder = LabelEncoder()
titanic["Sex"] = encoder.fit_transform(titanic["Sex"])
print(titanic.head())
X = titanic.drop(columns=["Survived"])
y = titanic["Survived"]

scalar = StandardScaler()
XScaled = scalar.fit_transform(X)
pca = PCA(n_components = 2)
XScaled_pca = pca.fit_transform(XScaled)
print(XScaled_pca)
X_train, X_test, y_train, y_test = train_test_split(XScaled_pca, y, test_size=0.2, random_state= 1)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred= log_reg.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

class_report = classification_report(y_test, y_pred)
print(class_report)

mat = confusion_matrix(y_test, y_pred)
print(mat)