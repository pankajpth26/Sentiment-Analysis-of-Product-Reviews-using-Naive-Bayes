import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib


nltk.download('stopwords')

df = pd.read_csv('reviews.csv')  

# Map ratings to sentiment (0=negative, 1=neutral, 2=positive)
df['sentiment'] = df['rating'].apply(lambda x: 0 if x <= 2 else 1 if x == 3 else 2)

# Text preprocessing func
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Apply preprocessing
df['processed_text'] = df['review'].apply(preprocess_text)

# Split dataset
X = df['processed_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
print("Creating features")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

# Naive Bayes model
model = MultinomialNB(alpha=0.1)
model.fit(X_train_tf, y_train)


y_pred = model.predict(X_test_tf)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# Visualizations
print("Generating visualizations...")
# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', bbox_inches='tight')

# Metrics Comparison
report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    values = [report['Negative'][metric], report['Neutral'][metric], report['Positive'][metric]]
    sns.barplot(x=['Negative', 'Neutral', 'Positive'], y=values, palette='viridis')
    plt.title(metric.capitalize())
    plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('metrics_comparison.png', bbox_inches='tight')

# Sentiment Distribution
plt.figure(figsize=(8, 5))
df['sentiment'].value_counts().plot(kind='bar', color=['#ff6b6b', '#4ecdc4', '#1a936f'])
plt.title('Sentiment Distribution in Dataset')
plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'], rotation=0)
plt.ylabel('Count')
plt.savefig('sentiment_dist.png', bbox_inches='tight')

# Word Clouds
def generate_wordcloud(sentiment, color):
    text = " ".join(df[df['sentiment'] == sentiment]['processed_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap=color, max_words=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f"{['Negative', 'Neutral', 'Positive'][sentiment]} Reviews")
    plt.savefig(f'wordcloud_{sentiment}.png', bbox_inches='tight')

generate_wordcloud(0, 'Reds')
generate_wordcloud(1, 'Blues')
generate_wordcloud(2, 'Greens')

# Save model
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
