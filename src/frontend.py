#importing libraries and dependencies
import streamlit as st
import pickle
import pandas as pd
from utils import load_models, TweetProcessor,  make_predictions

# Streamlit UI
def main():
    st.title('Offensive Text Detection')

    # Sidebar for entering unseen tweet
    unseen_tweet = st.text_input('Enter the text:')

    # Creating an instance of TweetProcessor
    tweet_processor = TweetProcessor()

    # Preprocessing the unseen tweet
    try:
        preprocessed_tweet = tweet_processor.process_tweets(pd.DataFrame({'tweet': [unseen_tweet]}))['stemmed_tweet'].iloc[0]
    except Exception as e:
        st.error(f"Error occurred while preprocessing tweet: {e}")
        preprocessed_tweet = ''

    # Loading the trained models and TF-IDF vectorizer
    try:
        loaded_model_a, loaded_model_b, loaded_model_c, tfidf_vectorizer = load_models()
    except Exception as e:
        st.error(f"Error occurred while loading models: {e}")
        loaded_model_a, loaded_model_b, loaded_model_c, tfidf_vectorizer = None, None, None, None

    # Button to make predictions
    if st.button('Make Predictions'):
        # Making predictions using the function from utils
        try:
            if all((loaded_model_a, loaded_model_b, loaded_model_c)):
                predictions = make_predictions(preprocessed_tweet, tfidf_vectorizer, loaded_model_a, loaded_model_b, loaded_model_c)
                
                # Display predictions
                if predictions:
                    st.subheader('Predictions:')
                    prediction_str = ' '.join(predictions)
                    st.write("The tweet is " ,prediction_str)
            else:
                st.error("Models are not loaded properly.")
        except Exception as e:
            st.error(f"Error occurred while making predictions: {e}")

if __name__ == '__main__':
    main()
