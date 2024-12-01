import os
import nltk
import spacy
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Ensure NLTK resources are downloaded or fallback to SpaCy
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_dir)

try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk_available = True
except:
    nltk_available = False
    nlp = spacy.load("en_core_web_sm")

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    if nltk_available:
        stop_words = set(stopwords.words('english'))
        porter = PorterStemmer()
        tokens = word_tokenize(text.lower())
        tokens = [porter.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    else:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha]
    return " ".join(tokens)

# Streamlit app
st.title("NLP Sentiment Analysis App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(data.head())

    # Preprocessing reviews
    if 'review' in data.columns:
        data['cleaned_review'] = data['review'].apply(preprocess_text)
        st.write("Cleaned Data Preview:")
        st.write(data[['review', 'cleaned_review']].head())
    else:
        st.error("The uploaded file must contain a 'review' column.")

    # Model training
    if 'sentiment' in data.columns:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['cleaned_review'])
        y = data['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Save the model and vectorizer
        with open("sentiment_model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)
        with open("vectorizer.pkl", "wb") as vec_file:
            pickle.dump(vectorizer, vec_file)

        # Display accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Prediction
        st.header("Test the Model")
        user_input = st.text_input("Enter a review to predict sentiment:")
        if user_input:
            cleaned_input = preprocess_text(user_input)
            input_vectorized = vectorizer.transform([cleaned_input])
            prediction = model.predict(input_vectorized)
            st.write(f"Predicted Sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")
    else:
        st.error("The uploaded file must contain a 'sentiment' column for training.")
