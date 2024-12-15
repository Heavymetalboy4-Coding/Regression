import pandas as pd

movies = pd.read_csv("C:/Users/arjun/OneDrive/Documents/Programming/Data/DataScience/KNN/movies_metadata.csv")
print(movies.info())

#calculating the mean report
c = movies["vote_average"].mean()
print(c)

m = movies["vote_count"].quantile(0.90)
print(m)

dataframe = movies.copy().loc[movies["vote_count"] >= m]
print(dataframe.shape) 
def weighted_rating(x, m = m, c = c):
    v = x["vote_count"]
    R = x["vote_average"]
    return v/(v+m)*R + m/(m+v)*c 
dataframe["score"] = dataframe.apply(weighted_rating, axis = 1)

dataframe = dataframe.sort_values("score", ascending= False)
print(dataframe[["title", "vote_count", "vote_average", "score"]].head(20))
