from sklearn.model_selection import train_test_split
import pandas as pd
student = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/KNN/student-mat.csv")
print(student.info())
from sklearn import preprocessing

encoding = preprocessing.LabelEncoder()
for i in student:    
    student[i] = encoding.fit_transform(student[i])
print(student.head())
student.drop("G1", axis=1, inplace= True)
student.drop("G2", axis=1, inplace= True)
print(student.info())
X = student.drop("G3", axis=1)
print(X.info())
y = student["G3"]
X_train, X_test, y_train, y_test  = train_test_split(X,  y, test_size=0.2,  random_state=1)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)  
yp = model.predict(X_test)
print(yp)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test, yp))
print(confusion_matrix(y_test, yp))
import seaborn as sns
import matplotlib.pyplot as plt 

matrix = confusion_matrix(y_test, yp)
sns.heatmap(matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrix")
#plt.legend()
plt.show()
print(classification_report(y_test, yp))