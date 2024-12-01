import os
import nltk
import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# NLTK Resource Setup
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download necessary resources
try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
except Exception as e:
    st.error(f"Failed to download NLTK resources. Error: {e}")

# Streamlit App Title
st.title("Sentiment Analysis App")

# Load Data
try:
    data = pd.read_excel("data/Canva_reviews.xlsx")
    st.write("Data Loaded Successfully")
except FileNotFoundError:
    st.error("File not found. Ensure 'Canva_reviews.xlsx' is in the 'data' folder.")

# Display Data Summary
if 'data' in locals():
    st.write("Data Shape:", data.shape)
    st.write("Data Preview:")
    st.write(data.head())

    # Sentiment Analysis
    if 'Sentiment' in data.columns:
        sentiment_counts = data['Sentiment'].value_counts()
        st.write("Sentiment Distribution:")
        st.bar_chart(sentiment_counts)
    else:
        st.error("The dataset must contain a 'Sentiment' column.")

    # Text Preprocessing
    st.header("Text Preprocessing")
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()

    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        tokens = [porter.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(tokens)

    if 'review' in data.columns:
        try:
            data['cleaned_review'] = data['review'].apply(preprocess_text)
            st.write("Cleaned Data Preview:")
            st.write(data[['review', 'cleaned_review']].head())
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
    else:
        st.error("The dataset must contain a 'review' column.")

    # Feature Extraction and Model Training
    try:
        vectorizer = CountVectorizer(min_df=5)
        X = vectorizer.fit_transform(data['cleaned_review'])
        y = data['Sentiment'].apply(lambda x: 1 if x == "Positive" else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        st.write(f"Train Accuracy: {train_acc:.2f}")
        st.write(f"Test Accuracy: {test_acc:.2f}")

        # Save Model
        output_dir = "Output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
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
                st.error("Please enter a review.")
    except Exception as e:
        st.error(f"Error during model training or prediction: {e}")
