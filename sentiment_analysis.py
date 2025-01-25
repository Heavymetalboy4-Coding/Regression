import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
train = pd.read_csv("Data/DataScience/KNN/train.txt", delimiter=";", names=["text", "label"])
print(train.head())
print(train["label"].value_counts())
def emotion_converter(data):
    data.replace(to_replace = "surprise", value = 1, inplace = True)
    data.replace(to_replace = "joy", value = 1, inplace = True)
    data.replace(to_replace = "anger", value = 0, inplace = True)
    data.replace(to_replace = "fear", value = 0, inplace = True)
    data.replace(to_replace = "love", value = 1, inplace = True)
    data.replace(to_replace = "sadness", value = 0, inplace = True)
emotion_converter(train["label"])
print(train["label"].head())

lm = WordNetLemmatizer()

def word_remover(data):
    list = []
    for i in data:
        removed_items = re.sub("[^a-zA-Z]", " ", str(i))
        removed_items =removed_items.lower()
        removed_items = removed_items.split()
        removed_items = [lm.lemmatize(i) for i in removed_items if i not in set(stopwords.words("english"))]
        list.append(" ".join(str(x) for x in removed_items ))
    return list
list = word_remover(train["text"])
print(list[1])

#Vectoriser will generate single words and pairs
count_vectorizer = CountVectorizer(ngram_range=(1,2))
X = count_vectorizer.fit_transform(list)
y = train.label
parameters = {"max_features":("auto", "sqrt"),
              "n_estimators":[500, 1000, 1500],
              "max_depth":[5, 10, None],
              "min_samples_leaf":[1, 2, 5, 10],
              "bootstrap":[True, False]}

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
grid.fit(X, y)
print(grid.best_params_)
forest = RandomForestClassifier(max_features=grid.best_params_["max_features"],max_depth=grid.best_params_["max_depth"],n_estimators=grid.best_params_["n_estimators"],min_samples_split=grid.best_params_["min_samples_split"], min_samples_leaf=grid.best_params_["min_samples_leaf"], bootstrap=grid.best_params_["bootstrap"])
forest.fit(X, y) 

test = pd.read_csv("Data/DataScience/KNN/test.txt", delimiter=";", names=["text", "label"])
X_test, y_test = test.text, test.label
y_test = emotion_converter(y_test)
X_test = word_remover(X_test)
X_test = count_vectorizer.transform(X_test)
y_predict = forest.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
score = accuracy_score(y_test, y_predict)
print(score)