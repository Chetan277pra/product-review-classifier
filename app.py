import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk import download

# Ensure stopwords are available
download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open("classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="🛍️ Product Review Classifier", layout="centered")
st.title("🛍️ Product Review Sentiment Classifier")
st.write("Analyze if a product review is **Positive** or **Negative** using a trained ML model.")

user_input = st.text_area("✍️ Enter your product review here:")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some review text.")
    else:
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = clf.predict(vectorized)[0]
        st.success(f"✅ **{prediction.capitalize()} Review**")
