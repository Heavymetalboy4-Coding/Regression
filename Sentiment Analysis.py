import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re

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