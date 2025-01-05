import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
#Tokenisation 
text = "Natural Language Processing is a subfield of artificial intelligence that focuses on the interaction between humans and computers using natural language."
tokens = word_tokenize(text)
print(tokens)
#stopwards
stop_words = set(stopwords.words("english"))
filteredtokens = []
for i in tokens:
    if i not in stop_words:
        filteredtokens.append(i)
print(filteredtokens)
wordlemmatizer = WordNetLemmatizer()
word_lemmatised = []
for i in filteredtokens:
    x = wordlemmatizer.lemmatize(i)
    word_lemmatised.append(x)
print(word_lemmatised)