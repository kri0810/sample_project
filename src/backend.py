#importing libraries and dependencies
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from final.utils import load_models, TweetProcessor, make_predictions

#Creating an instance of the FastAPI class and assigning it to the variable app
app = FastAPI()

#structure of the request payload for the predict end point
class TweetRequest(BaseModel):
    tweet: str


#defining end point
@app.post("/predict/")
# tweetrequest object is taken as input representing the tweet to be predicted
async def predict(tweet_request: TweetRequest):
    try:
        #calling the load models function to load the trained models
        loaded_model_a, loaded_model_b, loaded_model_c, tfidf_vectorizer = load_models()
        #generating predictions
        prediction = make_predictions(tweet_request.tweet, tfidf_vectorizer, loaded_model_a, loaded_model_b, loaded_model_c)
        return {"prediction" : prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {e}")
