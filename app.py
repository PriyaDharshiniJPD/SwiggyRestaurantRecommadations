import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Restaurant Recommendation", layout="wide")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df_full = pd.read_csv("swiggy_cleaned_data.csv")

    # SAME major cities logic as Colab
    top_cities = df_full["city"].value_counts().head(10).index
    swiggy_df = df_full[df_full["city"].isin(top_cities)].reset_index(drop=True)

    return swiggy_df

@st.cache_resource
def load_models():
    encoder = pickle.load(open("encoder.pkl", "rb"))
    svd = pickle.load(open("svd_model.pkl", "rb"))
    svd_features = np.load("svd_features.npy")

    return encoder, svd, svd_features


swiggy_df = load_data()
encoder, svd, svd_features = load_models()

cat_cols = list(encoder.feature_names_in_)

# -------------------- UI --------------------
st.title("🍽️ Restaurant Recommendation System")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox("Select City", sorted(swiggy_df["city"].unique()))

with col2:
    cuisine = st.selectbox("Select Cuisine", sorted(swiggy_df["cuisine"].unique()))

with col3:
    #k = st.slider("Number of Recommendations", 3, 15, 5)
    k = st.selectbox(
    "Number of Recommendations",
    [3, 5, 7, 10, 15]
)

col4, col5 = st.columns(2)

with col5:
    # max_cost = st.slider(
    #     "Maximum Cost",
    #     int(swiggy_df["cost"].min()),
    #     int(swiggy_df["cost"].max()),
    #     500
    # )
    budget_map = {
    "Any Budget": (0, 300000),
    "Low (₹0 – ₹200)": (0, 200),
    "Medium (₹200 – ₹500)": (200, 500),
    "High (₹500 – ₹1000)": (500, 1000),
    "Premium (₹1000+)": (1000, 300000)
}

budget_label = st.selectbox("Budget", list(budget_map.keys()))
min_cost, max_cost = budget_map[budget_label]
    

with col4:
    #min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5)
    min_rating = st.selectbox(
    "Rating",
    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    index=5   # default = 3.5
)
    


# -------------------- RECOMMENDATION ENGINE --------------------
if st.button("🔍 Get Recommendations"):

    # 1️⃣ Create query dataframe (EXACT feature order)
    user_input = pd.DataFrame([[cuisine, city]], columns=cat_cols)

    # 2️⃣ Encode
    user_encoded = encoder.transform(user_input)

    # 3️⃣ Apply SVD
    user_svd = svd.transform(user_encoded)

    # 4️⃣ Cosine similarity
    similarity = cosine_similarity(user_svd, svd_features).flatten()

    # 5️⃣ Apply filters (cost + rating)
    filtered_df = swiggy_df[
        (swiggy_df["cost"] <= max_cost) &
        (swiggy_df["cost"] >= min_cost)&
        (swiggy_df["rating"] >= min_rating)
    ]

    if filtered_df.empty:
        st.warning("No restaurants found with selected filters.")
    else:
        filtered_indices = filtered_df.index.tolist()
        filtered_similarity = similarity[filtered_indices]

        # 6️⃣ Top K similar restaurants
        top_indices = np.argsort(filtered_similarity)[-k:][::-1]
        final_indices = [filtered_indices[i] for i in top_indices]

        # 7️⃣ Display results
        recommendations = swiggy_df.loc[final_indices][
            ["name", "city", "cuisine", "rating", "cost"]
        ].reset_index(drop=True)

        st.subheader("✅ Recommended Restaurants")
        st.dataframe(recommendations, use_container_width=True)
        
