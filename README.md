# Twitter Sentiment Analysis

### Short Description:
This project builds a Sentiment Analysis tool that classifies tweets as positive, negative, or neutral using Natural Language Processing (NLP) techniques. It leverages Python, NLTK, scikit-learn, and machine learning to train a model on labeled tweet data and allows live sentiment prediction for new tweets.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Libraries](https://img.shields.io/badge/Libraries-Pandas%2C%20NLTK%2C%20Scikit--Learn%2C%20TextBlob-yellow?logo=python&logoColor=white)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-Custom%20CSV-lightgrey?logo=csv&logoColor=orange)](data/tweets.csv)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter&logoColor=white)](notebooks/eda.ipynb)

## Project Features

- Train a sentiment classifier on a labeled tweets dataset (tweets.csv).
- Preprocess tweets using tokenization, lemmatization, and stopword removal.
- Interactive sentiment prediction for new tweets.
- Exploratory Data Analysis (EDA) with sentiment distribution, word clouds, and tweet length analysis.
- Modular structure for easy extension and integration with live Twitter API.

## Project Structure
```bash
twitter_sentiment_analysis/
│
├── data/
│   └── tweets.csv           # Sample dataset of labeled tweets
│
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load tweets from CSV or Twitter API
│   ├── preprocessing.py     # Clean and preprocess text
│   ├── train_model.py       # Train sentiment classifier
│   ├── predict.py           # Predict sentiment on new tweets
│   └── visualize.py         # Optional charts and plots
│
├── main.py                  # Interactive menu: Train / EDA / Predict
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment file
└── README.md
```

## Installation
### Using Conda Environment
```bash
conda env create -f environment.yml
conda activate twitter_sentiment
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

### Using pip
```bash
conda create -n twitter_sentiment python=3.11 -y
conda activate twitter_sentiment
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

### Usage

- Run the interactive menu:
- python main.py
- Menu Options:

1. Train Model – Train the sentiment classifier on the dataset.
2. EDA – Explore dataset with visualizations and word clouds.
3. Predict Sentiment – Input a tweet and get its sentiment prediction. Type 'exit' to quit.

### Sample Output
Enter a tweet: I love this new app!
Predicted Sentiment: positive

### Dependencies

- Python 3.11
- pandas
- NLTK
- scikit-learn
- matplotlib
- seaborn
- spaCy
- TextBlob
- wordcloud
- Tweepy (optional for Twitter API integration)

### Future Improvements

- Integrate live Twitter API to fetch tweets in real-time.
- Improve accuracy with advanced NLP models like BERT.
- Add GUI or web interface for user-friendly predictions.
