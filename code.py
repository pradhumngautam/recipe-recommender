import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

data = pd.read_csv("/Users/pradhumngautam/Recipie/cuisines.csv")

df = data.copy()
df = df.dropna()

# ... (Your data preprocessing steps)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['ingredients'])

# Save the TF-IDF vectorizer for future use during inference
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

user_input = input("Enter your preferred ingredients (separated by commas): ")
user_input = user_input.split(',')
user_input_vector = tfidf_vectorizer.transform([' '.join(user_input)])
similarities = cosine_similarity(user_input_vector, tfidf_matrix)
top_n = 10
top_indices = similarities.argsort()[0, ::-1][:top_n]
print("Recommended Recipes:")
result_1 = []
for idx in top_indices:
    print(df['name'].iloc[idx])
    result_1.append(df['name'].iloc[idx])

cv = CountVectorizer(max_features=50000)
vectors = cv.fit_transform(df['ingredients']).toarray()
similarity = cosine_similarity(vectors)

# Save the CountVectorizer and similarity matrix for future use during inference
joblib.dump(cv, 'count_vectorizer.pkl')
joblib.dump(similarity, 'cosine_similarity_matrix.pkl')

def recommend(recipe):
    if not df[df['name'] == recipe].empty:
        index = df[df['name'] == recipe].index[0]
        distances = sorted(enumerate(similarity[index]), key=lambda x: x[1], reverse=True)
        similar_recipes = []
        for i, _ in distances[1:10]:
            similar_recipes.append(df.iloc[i]['name'])
        return similar_recipes
    else:
        return f"Recipe '{recipe}' not found in the dataset."

# Save the recommendation function for future use during inference
joblib.dump(recommend, 'recommendation_function.pkl')
