import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

cancer =  load_breast_cancer()
"""print(cancer.keys())
print(cancer["target"])
print(cancer["target_names"])
print(cancer["DESCR"])"""
X = pd.DataFrame(cancer['data'], columns= cancer["feature_names"])
print(X.head())
y  = cancer.target
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
scalar.fit(X)

scale_data = scalar.transform(X)


X_train, X_test, y_train, y_test = train_test_split(scale_data, y, test_size=0.2, random_state=1)
model =  LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scale_data)
transform_data = pca.transform(scale_data)
print(transform_data.shape)

X_train, X_test, y_train, y_test = train_test_split(transform_data, y, test_size=0.2, random_state=1)
model =  LogisticRegression()
model.fit(X_train, y_train)
print("After PCA: ", model.score(X_test, y_test))

plt.figure(figsize=(10,10))
plt.scatter(transform_data[:,0],transform_data[:,1],c=cancer["target"])
plt.xlabel("1st Principal component")
plt.ylabel("2nd principal component")
plt.show()
plt.legend()