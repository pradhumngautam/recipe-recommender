import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


data = pd.read_csv("cuisines.csv")

df = data.copy()
df = df.dropna()


df['ingredients'].replace('\t', '', regex=True, inplace=True)
df['ingredients'].replace('\n', '', regex=True, inplace=True)
df['ingredients'].replace('[0-9]+', '', regex=True, inplace=True)
df['ingredients'].replace('[^\w\s]', '', regex=True, inplace=True)
df['ingredients'].replace('cup', '', regex=True, inplace=True)
df['ingredients'].replace('teaspoon', '', regex=True, inplace=True)
df['ingredients'].replace('pinch', '', regex=True, inplace=True)


def separate_words_by_commas(text):
    words = text.split()
    return ', '.join(words)
df['ingredients'] = df['ingredients'].apply(separate_words_by_commas)
df['tags'] = df['instructions'].apply(separate_words_by_commas)
df['upvotes'] = 0


def increment(recipe):
    row_index = df[df['name'] == recipe].index[0]
    df.loc[row_index, 'upvotes'] += 1
    print(f'Upvoted {recipe}')





df['prep_time'].replace('Total in', '', regex=True, inplace=True)
df['prep_time'].replace('M', '', regex=True, inplace=True)


cuisines_all = df.cuisine
print(cuisines_all)
cuisines = cuisines_all.unique()
print(cuisines)


cuisines_frequency = {}

for cuisines in cuisines_all:
    if cuisines in cuisines_frequency:
        cuisines_frequency[cuisines] += 1
    else:
        cuisines_frequency[cuisines] = 1

print(cuisines_frequency)


diet = 'Vegetarian'
cuisine = 'Indian'
course = 'Lunch'
df_1 = df[(df['diet'] == diet) & (df['cuisine'] == cuisine) & (df['course'] == course)]
new_index = np.arange(len(df_1))
df_1.index = new_index



tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_1['ingredients'])
user_input = input("Enter your preferred ingredients (separated by commas): ")
user_input = user_input.split(',')
user_input_vector = tfidf_vectorizer.transform([' '.join(user_input)])
similarities = cosine_similarity(user_input_vector, tfidf_matrix)
top_n = 10
top_indices = similarities.argsort()[0, ::-1][:top_n]
print("Recommended Recipes:")
result_1 = []
for idx in top_indices:
    print(df_1['name'].iloc[idx])
    result_1.append(df_1['name'].iloc[idx])


cv = CountVectorizer(max_features = 50000)
vectors = cv.fit_transform(df_1['ingredients']).toarray()
similarity = cosine_similarity(vectors)


def recommend(recipe):
    if not df_1[df_1['name'] == recipe].empty:
        index = df_1[df_1['name'] == recipe].index[0]
        distances = sorted(enumerate(similarity[index]), key=lambda x: x[1], reverse=True)
        similar_recipes = []
        for i, _ in distances[1:10]:
            similar_recipes.append(df_1.iloc[i]['name'])
        return similar_recipes
    else:
        return f"Recipe '{recipe}' not found in the dataset."


recommend('Cabbage Carrot Onion Pudina Thepla Recipe')


filename = 'finalized_model.sav'
joblib.dump(tfidf_vectorizer, filename)






