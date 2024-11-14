import streamlit as st
import requests
import pandas as pd
from utils import load_models, TweetProcessor, make_predictions

def main():
    st.title('Offensive Text Detection')

    unseen_tweet = st.text_input('Enter the text:')
    
    # Creating an instance of TweetProcessor
    tweet_processor = TweetProcessor()
    
    # Preprocess the tweet text
    preprocessed_tweet = tweet_processor.process_tweets(pd.DataFrame({'tweet': [unseen_tweet]}))['stemmed_tweet'].iloc[0]

    if st.button('Make Predictions'):
        if preprocessed_tweet:
            try:
                payload = {"tweet": preprocessed_tweet}
                response = requests.post("http://localhost:8001/predict", json=payload)
                if response.status_code == 200:
                    prediction = response.json()["prediction"]
                    st.write("The text is ", prediction)
                else:
                    st.error("Error occurred while making predictions.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()


