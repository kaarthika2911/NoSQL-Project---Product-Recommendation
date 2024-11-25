# app.py
import os
import pandas as pd
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB connection
client = MongoClient('localhost', 27017)
db = client['recommender_db']
collection = db['reviews']

# Function to fetch data from MongoDB
def fetch_data_from_mongo():
    # Fetch all documents from the collection
    cursor = collection.find()
    data = list(cursor)
    return pd.DataFrame(data)

# Load Dataset from MongoDB
df = fetch_data_from_mongo()

# Data Preprocessing
df = df.dropna(subset=['ProductId', 'UserId', 'Score'])

# Convert 'Time' to datetime (optional, useful for future enhancements)
df['Time'] = pd.to_datetime(df['Time'], unit='s')

# Create User-Item Matrix for Collaborative Filtering
user_item_matrix = df.pivot_table(index='UserId', columns='ProductId', values='Score').fillna(0)

# Compute User Similarity Matrix
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function for Collaborative Filtering Recommendations
def recommend_products_collaborative(user_id, num_recommendations=5):
    if user_id not in user_similarity_df.index:
        return []
    
    # Similarity scores for the user
    similarity_scores = user_similarity_df[user_id]
    
    # Weighted scores for products
    weighted_scores = user_item_matrix.T.dot(similarity_scores) / similarity_scores.sum()
    
    # Exclude already rated products
    user_rated = user_item_matrix.loc[user_id]
    recommendations = weighted_scores[user_rated == 0].sort_values(ascending=False).head(num_recommendations)
    
    return recommendations.index.tolist()

# Home Route
@app.route('/')
def index():
    # Get unique UserIds
    user_ids = sorted(df['UserId'].unique())
    return render_template('index.html', user_ids=user_ids)

# Recommendation Route
@app.route('/recommend', methods=['POST'])
def recommend():
    method = request.form.get('method')
    input_id = request.form.get('input_id')
    
    if method != 'collaborative':
        return "Invalid recommendation method selected.", 400
    
    recommendations = recommend_products_collaborative(input_id)
    
    if not recommendations:
        return render_template('recommendations.html', products=[], message="No recommendations found. Please try a different User ID.", method=method)
    
    # Fetch product details for recommended ProductIds
    recommended_products = df[df['ProductId'].isin(recommendations)].drop_duplicates(subset=['ProductId'])
    
    # Select relevant columns to display
    recommended_products = recommended_products[['ProductId', 'Summary_clean', 'Text_clean']].head(5).to_dict(orient='records')
    
    return render_template('recommendations.html', products=recommended_products, method=method)

if __name__ == '__main__':
    app.run(debug=True) 
