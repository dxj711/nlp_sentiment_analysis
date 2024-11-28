import streamlit as st
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Set Streamlit title
st.title("Sentiment Analysis App")

# Load data
try:
    data = pd.read_excel("data/Canva_reviews.xlsx")
    st.write("Data Loaded Successfully")
except FileNotFoundError:
    st.error("File not found. Please ensure 'Canva_reviews.xlsx' exists in the 'Input' folder.")

# Display data summary
if 'data' in locals():
    st.write("Data Shape:", data.shape)
    st.write("Data Preview:")
    st.write(data.head())

    # Basic Analysis
    sentiment_counts = data['Sentiment'].value_counts()
    st.write("Sentiment Distribution:")
    st.bar_chart(sentiment_counts)

    # Text Preprocessing
    st.header("Text Preprocessing")
    stop_words = stopwords.words('english')
    porter = PorterStemmer()

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [porter.stem(word) for word in tokens if word not in stop_words and word.isalnum()]
        return " ".join(tokens)

    data['cleaned_review'] = data['review'].apply(preprocess_text)
    st.write("Cleaned Data Preview:")
    st.write(data[['review', 'cleaned_review']].head())

    # Feature Extraction
    vectorizer = CountVectorizer(min_df=5)
    X = vectorizer.fit_transform(data['cleaned_review'])
    y = data['Sentiment'].apply(lambda x: 1 if x == "Positive" else 0)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluation
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    st.write(f"Train Accuracy: {train_acc:.2f}")
    st.write(f"Test Accuracy: {test_acc:.2f}")

    # Save Model
    with open("Output/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("Output/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    st.write("Model and Vectorizer Saved Successfully")

    # Prediction Example
    st.header("Test the Model")
    user_input = st.text_area("Enter a review for sentiment analysis")
    if st.button("Predict Sentiment"):
        if user_input:
            processed_input = preprocess_text(user_input)
            input_vector = vectorizer.transform([processed_input])
            prediction = model.predict(input_vector)
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            st.write(f"Predicted Sentiment: {sentiment}")
        else:
            st.error("Please enter a review")
