import pandas as pd
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
from fastapi import FastAPI, HTTPException
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import string
from nltk.stem import PorterStemmer
import re
import pickle
nltk.download('wordnet')
nltk.download('stopwords')

class TweetProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_tweet(self, tweet):
        punctuations = string.punctuation

        # Remove mentions, URLs, and special characters
        tweet = tweet.replace('@USER', '') \
                     .replace('URL', '') \
                     .replace('&amp', 'and') \
                     .replace('&lt','') \
                     .replace('&gt','') \
                     .replace('\d+','') \
                     .lower()

        # Remove punctuations
        for punctuation in punctuations:
            tweet = tweet.replace(punctuation, '')

        # Remove emojis
        tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)

        # Remove stopwords
        tweet = ' '.join([word for word in tweet.split() if word not in self.stop_words])

        # Trim leading and trailing whitespaces
        tweet = tweet.strip()

        return tweet
    
    def tokenize_tweet(self, tweet):
        return word_tokenize(tweet)

    def stem_tweet(self, tokens):
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return stemmed_tokens
    
    def process_tweets(self, df, ngram_range=(1, 3), stop_words='english'):
        # Clean tweets
        df['cleaned_tweet'] = df['tweet'].apply(self.clean_tweet)

        # Tokenize tweets
        df['tokenized_tweet'] = df['cleaned_tweet'].apply(self.tokenize_tweet)

         # Stem tweets
        df['stemmed_tweet'] = df['tokenized_tweet'].apply(self.stem_tweet)

        # Join stemmed tokens back into strings
        df['stemmed_tweet'] = df['stemmed_tweet'].apply(lambda x: ' '.join(x))

        return df
    

    

def load_models():
    try:
        with open('random_forest_model_a.pkl', 'rb') as file:
            loaded_model_a = pickle.load(file)
        
        with open('random_forest_model_b.pkl', 'rb') as file:
            loaded_model_b = pickle.load(file)
        
        with open('random_forest_model_c.pkl', 'rb') as file:
            loaded_model_c = pickle.load(file)
        
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            tfidf_vectorizer = pickle.load(file)

        return loaded_model_a, loaded_model_b, loaded_model_c, tfidf_vectorizer
    except Exception as e:
        raise Exception(status_code=500, detail=f"Error occurred while loading models: {e}")

def make_predictions(unseen_tweet, tfidf_vectorizer, loaded_model_a, loaded_model_b, loaded_model_c):
    try:
        # Preprocess the unseen tweet
        tweet_processor = TweetProcessor()
        preprocessed_tweet = tweet_processor.process_tweets(pd.DataFrame({'tweet': [unseen_tweet]}))['stemmed_tweet'].iloc[0]

        # Vectorize the preprocessed tweet using the TF-IDF vectorizer
        X_unseen = tfidf_vectorizer.transform([preprocessed_tweet])

        # Empty list to store prediction results
        predictions = []

        # Make predictions using Model A
        prediction_a = loaded_model_a.predict(X_unseen)
        if prediction_a == 0:
            predictions.append("Not offensive")
        else:
            predictions.append("Offensive")

            # Make predictions using Model B
            prediction_b = loaded_model_b.predict(X_unseen)
            
            # Check if the tweet is targeted
            if prediction_b == 0:
                predictions.append("Untargeted")
            else:
                predictions.append("Targeted")

                # Make predictions using Model C
                prediction_c = loaded_model_c.predict(X_unseen)
                if prediction_c == 0:
                    predictions.append('Group') 
                elif prediction_c == 1:
                    predictions.append('Individual') 
                else:
                    predictions.append('Others') 

        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred while making predictions: {e}")