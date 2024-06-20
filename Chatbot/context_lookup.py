from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import json

file_path = open(f'/Users/uladzimircharniauski/Documents/GetBetterHead/text_database.json', 'r')

text_database = json.load(file_path)

video_texts = {entry['name']: entry['context'] for entry in text_database}

texts = [entry['context'] for entry in text_database]
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)

def get_relevant_text(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, vectors).flatten()
    index = np.argmax(similarity)
    return texts[index]

# # Example usage - uncomment and watch how it works
# query = "How to deal with anxiety?"
# relevant_text = get_relevant_text(query)
# print(relevant_text)