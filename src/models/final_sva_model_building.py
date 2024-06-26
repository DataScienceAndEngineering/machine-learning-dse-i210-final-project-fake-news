# -*- coding: utf-8 -*-
"""Final_SVA_Model_Building.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HC-Xhu3Rj85j8Xr_ff--gG52IXocZR8Q

# **Data Load**
"""

!python -m spacy download en_core_web_md

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import spacy

from google.colab import drive
# Mount your Google Drive
drive.mount('/content/drive')

# Read the dataset from the specified path
df = pd.read_csv('/content/drive/MyDrive/ML project/Final /fake_real_vector.csv', sep=',', encoding='utf-8', quotechar='"')
df.head()

# Check the type of the 'vector' column
print(df['vector'].dtype)

unique_types = df['vector'].apply(type).unique()
print(unique_types)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Save the trained model
joblib.dump(svm_classifier, '/content/drive/MyDrive/ML project/svm_classifier.pkl')

# Load the SpaCy model
nlp = spacy.load('en_core_web_md')

# Function to preprocess and vectorize a new article
def preprocess_and_vectorize(news, nlp):
    doc = nlp(news)
    return doc.vector

# Function to check whether a news article is real or fake
def check_news(news, model, nlp):
    news_vector = preprocess_and_vectorize(news, nlp)
    predicted_label = model.predict([news_vector])[0]
    return predicted_label

# Get news input from the user
user_news = input("Enter the news article: ")

# Check the user-provided news
predicted_label = check_news(user_news, svm_classifier, nlp)
print("Predicted Label:", "Real News" if predicted_label else "Fake News")