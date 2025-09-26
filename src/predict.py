import pickle
from src.preprocessing import clean_text

def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_sentiment(tweet):
    model, vectorizer = load_model()
    clean_tweet = clean_text(tweet)
    vec = vectorizer.transform([clean_tweet])
    prediction = model.predict(vec)[0]
    return prediction

if __name__ == "__main__":
    tweet = input("Enter a tweet: ")
    print("Predicted Sentiment:", predict_sentiment(tweet))
