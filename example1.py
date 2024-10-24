import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
import numpy as np
# Indian Cuisine Dish Data (replace with actual data)
indian_dishes = {
    "name": [
        "Butter Chicken",
        "Tandoori Chicken",
        "Dal Makhani",
        "Palak Paneer",
        "Biryani (Chicken/Veg)",
        "Masala Dosa",
        "Vada Pav",
        "Chole Bhature",
        "Aloo Gobi",
        "Samosa Chaat",
        # ... add more dishes here
    ],
    "ingredients": [
        "chicken, tomato, butter, cream",
        "chicken, yogurt, spices",
        "black lentils, butter, cream",
        "spinach, paneer (cottage cheese), spices",
        "rice, meat/vegetables, spices",
        "crepe, potato filling, sambar, chutney",
        "potato dumpling, bun, chutney",
        "chickpea curry, fried bread",
        "potato, cauliflower, spices",
        "samosa, chutney, yogurt",
        # ... add more ingredients here
    ],
    "region": [
        "North India",
        "North India",
        "North India",
        "North India",
        "Across India",
        "South India",
        "West India",
        "North India",
        "North India",
        "North India",
        # ... add more regions here
    ],
    "spice_level": ["Medium", "Medium-High", "Medium", "Medium", "Varies", "Low", "Low", "Medium", "Medium", "High"], # Corrected length
    "vegetarian": [False, False, False, True, False, True, True, False, False, False], # Corrected length
    "img_link": [
        "link_to_image1",
        "link_to_image2",
        "link_to_image3",
        "link_to_image4",
        "link_to_image5",
        "link_to_image6",
        "link_to_image7",
        "link_to_image8",
        "link_to_image9",
        "link_to_image10",
        # ... add more image links here
    ]
}

indian_dishes_df = pd.DataFrame(indian_dishes)

# Initialize session state
if 'cart' not in st.session_state:
    st.session_state['cart'] = []
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []
if 'buy_history' not in st.session_state:
    st.session_state['buy_history'] = []
if 'user_logged_in' not in st.session_state:
    st.session_state['user_logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

    # User Authentication Page (optional, can be removed)
def show_login_page():
    st.title("Welcome to DeepTrendz")
    st.subheader("Please sign in to continue")
    username_input = st.text_input("Username", "")
    if st.button("Login"):
        if username_input:
            st.session_state['user_logged_in'] = True
            st.session_state['username'] = username_input
            st.success(f"Welcome, {username_input}!")
        else:
            st.warning("Please enter a username.")

            # Display Purchase History (optional, can be removed)
def show_purchase_history():
    st.subheader("Your Purchase History")
    if st.session_state['buy_history']:
        for product in st.session_state['buy_history']:
            st.write(f"- {product}")
    else:
        st.write("No purchase history available.")

        # Recommendation System
def get_content_based_recommendations(user_history, num_recommendations=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(indian_dishes_df['ingredients'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(indian_dishes_df.index, index=indian_dishes_df['name']).drop_duplicates()
    recommended_dishes = set()

    for dish in user_history:
        if dish in indices.index:
            idx = indices[dish]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = [(i, score) for i, score in sim_scores if score is not None and (isinstance(score, float) and not np.isnan(score))]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:num_recommendations + 1]
            dish_indices = [i[0] for i in sim_scores]
            recommended_dishes.update(indian_dishes_df['name'].iloc[dish_indices])

    return list(recommended_dishes)

def get_collaborative_recommendations(user_history, num_recommendations=5):
    if len(st.session_state['buy_history']) < 2:
        return []

    user_vector = np.zeros(len(indian_dishes_df))
    for item in user_history:
        if item in indian_dishes_df['name'].values:
            idx = indian_dishes_df[indian_dishes_df['name'] == item].index[0]
            user_vector[idx] += 1

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(user_vector.reshape(-1, 1))
    cluster_label = kmeans.predict(user_vector.reshape(-1, 1))

    similar_users = np.where(kmeans.labels_ == cluster_label)[0]
    recommended_dishes = []

    for user_index in similar_users:
        user_items = st.session_state['buy_history']
        recommended_dishes.extend(user_items)

    return list(set(recommended_dishes))

def get_combined_recommendations():
    user_history = st.session_state['buy_history'] + st.session_state['search_history']
    content_recommendations = get_content_based_recommendations(user_history)
    collaborative_recommendations = get_collaborative_recommendations(user_history)

    combined_recommendations = list(set(content_recommendations) | set(collaborative_recommendations))
    return combined_recommendations

def show_recommendations():
    if st.session_state['user_logged_in']:
        recommendations = get_combined_recommendations()
        st.markdown("### Recommended Indian Dishes")
        if recommendations:
            for dish in recommendations:
                dish_details = indian_dishes_df[indian_dishes_df['name'] == dish].iloc[0]
                st.write(f"*{dish_details['name']}* - {dish_details['spice_level']} - {'Vegetarian' if dish_details['vegetarian'] else 'Non-Vegetarian'}")
                st.image(dish_details['img_link'], width=150)
        else:
            st.warning("No recommendations found.")
    else:
        st.warning("You need to be logged in to see recommendations.")