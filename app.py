import streamlit as st
import pickle

# Load model & vectorizer
with open("classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App config
st.set_page_config(page_title="🛍️ Product Review Classifier", layout="centered")
st.title("🛍️ Product Review Sentiment Classifier")

st.markdown("Analyze if a product review is **Positive** or **Negative** using a trained ML model.")

# User input
user_input = st.text_area("✍️ Enter your product review here:")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review first.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = clf.predict(transformed)[0]

        if prediction == "positive":
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
