# Email Spam Detection With Machine Learning
## Description
The goal of this project is to use machine learning and Python to create an email spam detector. Building a system that can automatically assess incoming emails and categorize them as spam or non-spam (ham) is the aim. In order to precisely identify spam content, this entails preprocessing email data, collecting pertinent features, and training a machine learning model. The ultimate goal is to reduce the likelihood that users will become victims of phishing scams, scams, and other dangerous actions that are frequently connected to spam emails in order to improve email security.
## Key Concepts and Challenges
Three main ideas are involved in creating an email spam detector: data preprocessing, feature extraction, and machine learning model selection and training. Data preprocessing is cleaning the email text and transforming it into an analysis-ready format. Feature extraction is finding the most important email attributes to distinguish spam from non-spam, like word frequency or specific keywords. Choosing and training the right machine learning model, like a Naive Bayes classifier or a Support Vector Machine, is essential to obtaining high accuracy. Difficulties in this project include managing imbalanced datasets, making sure the model generalizes well to new data, and continuously updating the model to accommodate evolving spam tactics.
## Learning Objectives

src/main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

### Load the dataset
df = pd.read_csv(r'F:\Oasis', encoding='latin-1')

### Drop unnecessary columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

### Rename the columns for convenience
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

### Encode the labels (spam = 1, ham = 0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

### Define the features (X) and the target (y)
X = df['message']
y = df['label']

### Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Create a pipeline that includes count vectorizer, TF-IDF transformer, and a MultinomialNB classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

### Train the model
pipeline.fit(X_train, y_train)

### Make predictions on the test set
y_pred = pipeline.predict(X_test)

### Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

