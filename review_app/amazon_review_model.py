#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


# In[2]:


#get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu')


# In[3]:


#get_ipython().system('pip install tensorflow')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


# In[5]:


# get_ipython().system('pip install scikit-surprise')


# In[6]:


# get_ipython().system('pip install FuzzyTM')
# get_ipython().system('pip install blosc2')
# get_ipython().system('pip install cython')
# get_ipython().system('pip install numpy==1.23.5  # Since numba requires numpy<1.25')


# In[7]:


# get_ipython().system('pip install scikit-surprise')


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


# In[9]:


import pandas as pd

# Define the file path
file_path = 'C:/Users/User/Downloads/Clothing_Shoes_and_Jewelry_5.json'



def load_data(file_path, chunk_size=10000):
    """
    Load data from a JSON file in chunks.

    Args:
        file_path (str): Path to the JSON file.
        chunk_size (int): Number of rows per chunk.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all chunks.
    """
    # Read the file in chunks
    try:
        chunks = pd.read_json(file_path, lines=True, chunksize=chunk_size)
        data = pd.concat(chunks, ignore_index=True)
        return data
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

# Load the data
data = load_data(file_path)

# If data is loaded successfully, print the first few rows
if data is not None:
    print(data.head())



import matplotlib.pyplot as plt
import seaborn as sns

# After loading and cleaning chunks, perform EDA

# Check the distribution of votes
sns.histplot(chunk['vote'], kde=True)
plt.title("Distribution of Ratings (Vote)")
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show(block=False)

# Check the length of reviews
chunk['reviewLength'] = chunk['reviewText'].apply(lambda x: len(str(x)))
sns.histplot(chunk['reviewLength'], kde=True)
plt.title("Distribution of Review Lengths")
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show(block=False)

# Check for missing values
missing_values = chunk.isnull().sum()
print(f"Missing values:\n{missing_values}")


# In[11]:


# Fill or drop missing values
chunk.dropna(subset=['reviewText', 'vote'], inplace=True)

# Convert reviewTime to datetime
chunk['reviewTime'] = pd.to_datetime(chunk['reviewTime'], errors='coerce')

# Extract the year from reviewTime
chunk['reviewYear'] = chunk['reviewTime'].dt.year

# Feature engineering: Length of the review text
chunk['reviewLength'] = chunk['reviewText'].apply(lambda x: len(str(x)))

# Check the cleaned data
print(chunk.head())

#added externally
def preprocess_data(chunk):
    """
    Preprocess the data chunk: handle missing values, feature engineering, etc.
    
    Args:
        chunk (pd.DataFrame): The raw chunk of data to preprocess.
        
    Returns:
        pd.DataFrame: The preprocessed chunk of data.
    """
    if chunk is None:
        print("Error: Received empty or None chunk for preprocessing.")
        return None
    
    # Drop rows with missing review text or votes
    chunk.dropna(subset=['reviewText', 'vote'], inplace=True)

    # Convert reviewTime to datetime
    chunk['reviewTime'] = pd.to_datetime(chunk['reviewTime'], errors='coerce')

    # Extract the year from reviewTime
    chunk['reviewYear'] = chunk['reviewTime'].dt.year

    # Feature engineering: Length of the review text
    chunk['reviewLength'] = chunk['reviewText'].apply(lambda x: len(str(x)))

    # Sentiment Analysis: Create a new column for sentiment polarity
    chunk['sentiment'] = chunk['reviewText'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    # Word count of the reviews
    chunk['wordCount'] = chunk['reviewText'].apply(lambda x: len(str(x).split()))

    return chunk

def train_model(X_train, y_train):
    """
    Train a machine learning model (Random Forest) on the provided data.

    Args:
        X_train (array-like): The feature matrix for training.
        y_train (array-like): The target variable for training.

    Returns:
        RandomForestClassifier: The trained model.
    """
    # Build Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict(review_text, model, tfidf):
    """
    Predict the target value (size or other feature) based on a new review.

    Args:
        review_text (str): The review text to predict on.
        model (RandomForestClassifier): The trained model.
        tfidf (TfidfVectorizer): The fitted TF-IDF vectorizer.

    Returns:
        int: The predicted value (e.g., size).
    """
    # Transform the input review to match the model's feature space
    review_tfidf = tfidf.transform([review_text])
    
    # Make prediction using the trained model
    predicted_value = model.predict(review_tfidf)
    return predicted_value[0]


# In[13]:


#get_ipython().system('pip install textblob')


# In[14]:


from textblob import TextBlob

# Sentiment Analysis: Create a new column for sentiment polarity
chunk['sentiment'] = chunk['reviewText'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Word count of the reviews
chunk['wordCount'] = chunk['reviewText'].apply(lambda x: len(str(x).split()))

# Check the engineered features
print(chunk[['reviewText', 'sentiment', 'wordCount']].head())


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Feature: reviewText
X = chunk['reviewText']
y = chunk['vote']

# Convert text to numerical features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Build Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Assuming 'chunk' is a data chunk that you've processed
# Feature: reviewText
X = chunk['reviewText']
y = chunk['vote']  # 'vote' would be the size or feature related information

# Convert text to numerical features using TF-IDF (Text Vectorization)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Build Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Extract feature importance to recommend sizes/features
importances = model.feature_importances_
indices = importances.argsort()

# Show top features that influence the predictions (useful for recommendations)
top_n = 10  # Number of important features to display
top_features = [tfidf.get_feature_names_out()[i] for i in indices[-top_n:]]
print(f"\nTop {top_n} important features for recommendations:")
for feature in top_features:
    print(feature)

# You can now use this model to predict new data:
def recommend_size(review_text):
    # Transform the input review to match the model's feature space
    review_tfidf = tfidf.transform([review_text])
    predicted_size = model.predict(review_tfidf)
    return predicted_size[0]

# Example use case:
review = "This shoe fits perfect, I love it!"
recommended_size = recommend_size(review)
print(f"Recommended Size for the review: {recommended_size}")


# In[ ]:




