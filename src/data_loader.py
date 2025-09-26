import pandas as pd # type: ignore
import tweepy # type: ignore

def load_csv(file_path):
    return pd.read_csv(file_path)

def fetch_tweets(api_key, api_secret, access_token, access_token_secret, query, count=100):
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en").items(count)
    data = [{"tweet": tweet.text} for tweet in tweets]
    
    return pd.DataFrame(data)
