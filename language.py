import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
example_string = """
Muad'Dib learned rapidly because his first training was in how to learn.
And the first lesson of all was the basic trust that he could learn.
It's shocking to find how many people do not believe they can learn,
and how many more believe learning to be difficult."""
#print(sent_tokenize(example_string))

#print(word_tokenize(example_string))

"""string = "Hello World!, I am learing python and i am through to the final stages of learning python."
token = word_tokenize(string)
print(token)
stop_words = set(stopwords.words("English"))
print(stop_words)

words = []
for i in token:
    if i.casefold() not in stop_words:
        words.append(i)
print(words)"""

stemmer = PorterStemmer()
string_for_stemming = """
... The crew of the USS Discovery discovered many discoveries.
... Discovering is what explorers do."""

wordtoken = word_tokenize(string_for_stemming)
print(wordtoken)
stemmedword = [stemmer.stem(i) for i in wordtoken]
print(stemmedword)