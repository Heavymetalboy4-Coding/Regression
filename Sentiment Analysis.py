import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
text = pd.read_csv("Data/DataScience/KNN/train.txt", delimiter=";", names=["sentence", "feelings"])
print(text.head())
print(text.info())
print(text["feelings"].value_counts())
def emotions(data):
    data.replace(to_replace = "surprise", value = 1, inplace = True)
    data.replace(to_replace = "joy", value = 1, inplace = True)
    data.replace(to_replace = "anger", value = 0, inplace = True)
    data.replace(to_replace = "fear", value = 0, inplace = True)
    data.replace(to_replace = "love", value = 1, inplace = True)
    data.replace(to_replace = "sadness", value = 0, inplace = True)
emotions(text["feelings"])
print(text["feelings"].head())

lm = WordNetLemmatizer()

def remover(data):
    list = []
    for i in data:
        removed_items = re.sub("[^a-zA-Z]", " ", str(i))
        removed_items =removed_items.lower()
        removed_items = removed_items.split()
        removed_items = [lm.lemmatize(i) for i in removed_items if i not in set(stopwords.words("english"))]
        list.append(" ".join(str(i) for i in removed_items ))
    return list
transform_text = remover(text["sentence"])
print(transform_text)

cv = CountVectorizer(ngram_range=(1,2))
X = cv.fit_transform(list)
y = text.feelings
parameters = {"max_features":("auto", "sqrt"),
              "n_estimaters":[500, 1000, 1500],
              "max_depth":[5, 10, None],
              "min_samples_leaf":[1, 2, 5, 10],
              "bootstrap":[True, False]}
               
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
grid.fit(X, y)

test = pd.read_csv("Data/DataScience/KNN/test.txt", delimiter=";", names=["sentence", "feelings"])
X_test, y_test = test.sentence, test.feelings
y_test = emotions(y_test)
X_test = remover(X_test)
X_test = cv.transform(X_test, y_test)
y_predict = grid.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
score = accuracy_score(y_test, y_predict)
print(score)