import pandas as pd

movies = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/KNN/movies_metadata.csv")
print(movies.info())

#Content Based Reccomendation
from sklearn.feature_extraction.text import TfidfVectorizer 
vectoriser = TfidfVectorizer(stop_words="english")
print(movies.isnull().sum())
movies["overview"] = movies["overview"].fillna("")
movies_vectorised = vectoriser.fit_transform(movies["overview"])
print(movies_vectorised.shape)
print(vectoriser.get_feature_names_out()[7000:7010])
from sklearn.metrics.pairwise import linear_kernel
similarity = linear_kernel(movies_vectorised, movies_vectorised)
series = pd.Series(movies.index, index=movies["title"]).drop_duplicates()
print(series)

def recommendation(title, cosine_sim = similarity):
    idx = series["title"]
    sim_scores = list[enumerate(cosine_sim[idx])]
    print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movies_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc(movies_indices)
print(recommendation("Batman"))
