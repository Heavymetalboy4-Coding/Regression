from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

heart = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/KNN/dataset.csv")
print(heart.info())
X = heart.drop(["target"], axis=1)
print(X.head())
y = heart["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
model = SVC()
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
Xscaled = scalar.fit_transform(X_train)
Xtrscaled = scalar.transform(X_test)
model.fit(Xscaled, y_train)
yp = model.predict(Xtrscaled)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test, yp))
print(confusion_matrix(y_test, yp))