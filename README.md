# Sentiment-Analysis-of-Product-Reviews-using-Naive-Bayes
📊 **Sentiment Analysis of Product Reviews using Naive Bayes Classifier** A Model that predicts product review sentiment (Positive,Negative or Neautral) using NLP and machine learning. Built with Python, NLTK, and Machine learning algorithms.
<br>
<br>
📌 Overview
<br>
This project analyzes customer reviews to classify sentiment into Positive, Negative, or Neutral categories. It demonstrates a complete NLP pipeline from text preprocessing to model evaluation, with visualizations for insights.
<br>
<br>
🛠️ Technologies Used
<br>
Category => Tools/Libraries
<br>
Language => Python 3.10
<br>
NLP	NLTK =>  (Stopwords, PorterStemmer)
<br>
ML => Scikit-learn (Naive Bayes, TF-IDF)
<br>
Visualization =>	Matplotlib, Seaborn, WordCloud
<br>
Model Saving => Joblib
<br>
<br>
📂 Dataset
<br>
Source: Custom dataset of 10,000 product reviews
<br>
Columns:
<br>
review: Raw text reviews
<br>
rating: Numerical ratings (1-5 stars)
<br>
Sentiment Mapping:
<br>
1-2★ → Negative (0)
<br>
3★ → Neutral (1)
<br>
4-5★ → Positive (2)     
<br>
<br>
# 🧩Key Techniques:
<br>
Tokenization: (Splits text into individual words or tokens.), 
<br>
Stopword Removal: (Removes common words that carry little meaning (like is, the, and, a))
<br>
Stemming: (Reduces words to their root form.)
<br>
TF-IDF Vectorization: (Converts text into numbers based on how important a word is in a document and across all documents.)
<br>
Naive Bayes Classification: (A simple, fast machine learning algorithm that predicts categories (like Positive, Negative, Neutral) using probabilities based on word frequencies.)
<br>
<br>

# 🖥️ NLP Pipeline:
<br>
Preprocessing:
<br>
Lowercasing
<br>
Remove special characters
<br>
Stopword removal (nltk.corpus.stopwords)
<br>
Stemming (PorterStemmer)
<br>
Feature Extraction:
TF-IDF with 5,000 features
Model:
Naive Bayes (MultinomialNB)
<br>
<br>

# 📊Visualization
<br>
Sentiment analysis:
<img width="876" height="585" alt="image" src="https://github.com/user-attachments/assets/8e43c5b6-6cc9-4e37-82c6-f5d759f2b48a" />
<br>

**Word Cloud:**
<br>
**Negative Word cloud:**
<br>
<img width="751" height="375" alt="image" src="https://github.com/user-attachments/assets/aac0c6b3-4577-40d6-ae31-36601b70e433" />
<br>
**Positive Word Colud:**
<br>
<img width="752" height="375" alt="image" src="https://github.com/user-attachments/assets/61e401b7-1e4a-4a94-936f-8b8b7d4f5480" />
<br>
**Neutral Word Cloud:**
<br>
<img width="751" height="375" alt="image" src="https://github.com/user-attachments/assets/3ac923ff-768b-4d07-ba67-d9af09b897b0" />

# 📱GUI IMPLEMENTATION SCREENSHOT:
<br>
<br>

**Postive Sentiment**
<img width="1920" height="1152" alt="image" src="https://github.com/user-attachments/assets/b3896bfb-d7b9-4701-9094-d56997b6fd16" />
<br>

**Negative Sentiment**
<img width="1920" height="1095" alt="image" src="https://github.com/user-attachments/assets/ce863859-c5b4-40aa-8096-5f1416229143" />

<br>

**Neutral Sentiment**
<img width="1920" height="1149" alt="image" src="https://github.com/user-attachments/assets/b69e4c66-5db2-4508-a2dc-d464d15f9f41" />


 
