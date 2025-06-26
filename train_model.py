import pandas as pd
import re
import nltk
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load full Amazon Fine Food Reviews dataset
df = pd.read_csv("Reviews.csv")
df = df[['Text', 'Score']].dropna()

# Label reviews: 4 or 5 = positive, else negative
def label_sentiment(score):
    return 'positive' if score >= 4 else 'negative'

df['label'] = df['Score'].apply(label_sentiment)

# Sample 10k positive and 10k negative for balance
pos_reviews = df[df['label'] == 'positive'].sample(n=10000, random_state=42)
neg_reviews = df[df['label'] == 'negative'].sample(n=10000, random_state=42)
df = pd.concat([pos_reviews, neg_reviews]).sample(frac=1).reset_index(drop=True)

print(df['label'].value_counts())

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned'] = df['Text'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save artifacts
with open("classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
