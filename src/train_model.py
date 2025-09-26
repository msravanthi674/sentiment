import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import classification_report # type: ignore
import pickle
from src.preprocessing import clean_text

def train_model(csv_path):
    df = pd.read_csv(csv_path)
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    
    X = df['clean_tweet']
    y = df['sentiment']  # should be 'positive', 'negative', 'neutral'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    with open("sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Model and vectorizer saved!")

if __name__ == "__main__":
    train_model("../data/tweets.csv")
