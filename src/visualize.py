import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

def plot_sentiment_distribution(df):
    sns.countplot(x='sentiment', data=df)
    plt.title("Sentiment Distribution")
    plt.show()
