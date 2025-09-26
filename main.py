from src.data_loader import load_csv
from src.train_model import train_model
from src.predict import predict_sentiment

if __name__ == "__main__":
    # Step 1: Train model
    train_model("data/tweets.csv")
    
    # Step 2: Predict new tweet
    tweet = input("Enter a tweet to analyze sentiment: ")
    print("Sentiment:", predict_sentiment(tweet))
