import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("K_means_clustering_model.pkl")
scaler = joblib.load("K_means_clustering_scaler.pkl")

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊"
)

st.title("📊 Customer Segmentation using K-Means")

st.write("Predict the customer cluster based on Annual Income and Spending Score.")

income = st.slider("Annual Income (k$)", 10, 150, 50)
score = st.slider("Spending Score (1-100)", 1, 100, 50)

if st.button("Predict Cluster"):

    customer = pd.DataFrame({
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [score]
    })

    customer_scaled = scaler.transform(customer)

    cluster = model.predict(customer_scaled)

    st.success(f"Customer belongs to Cluster {cluster[0]}")