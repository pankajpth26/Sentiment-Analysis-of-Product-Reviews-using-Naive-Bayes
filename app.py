import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Download stopwords
nltk.download('stopwords')

# Preprocessing function (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Streamlit app
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# Header
st.title("📱 Product Review Sentiment Analysis")
st.markdown("""
    <style>
    .st-emotion-cache-1kyxreq {justify-content: center;}
    .st-bq {font-weight: bold;}
    .positive {color: #1a936f;}
    .neutral {color: #4ecdc4;}
    .negative {color: #ff6b6b;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
This tool analyzes product reviews to determine sentiment using:
- **NLP Techniques**: Tokenization, Stopword Removal, Stemming
- **TF-IDF Vectorization**
- **Naive Bayes Classifier**
""")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyze Review")
    review = st.text_area("Enter your product review:", "This product is amazing! I love it.")
    
    if st.button("Analyze Sentiment", type="primary"):
        # Preprocess
        cleaned = preprocess_text(review)
        # Vectorize
        vectorized = tfidf.transform([cleaned])
        # Predict
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        
        # Display result
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment_icons = ['😠 Negative', '😐 Neutral', '😊 Positive']
        sentiment_colors = ['#ff6b6b', '#4ecdc4', '#1a936f']
        
        st.subheader("Analysis Result")
        st.markdown(f"<h2 class='{sentiment_labels[prediction].lower()}'>{sentiment_icons[prediction]}</h2>", 
                    unsafe_allow_html=True)
        
       


with col2:
    st.subheader("Model Insights")
    st.image('confusion_matrix.png', caption='Model Performance')
    
    
    st.subheader("Dataset Overview")
    st.image('sentiment_dist.png', caption='Sentiment Distribution')
    
    st.subheader("Word Clouds")
    tab1, tab2, tab3 = st.tabs(["Negative", "Neutral", "Positive"])
    with tab1:
        st.image('wordcloud_0.png')
    with tab2:
        st.image('wordcloud_1.png')
    with tab3:
        st.image('wordcloud_2.png')

# Footer
st.divider()
st.caption("Sentiment Analysis Project | Naive Bayes Classifier | NLP Techniques")