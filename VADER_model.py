import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


nltk.download('vader_lexicon')

sia =SentimentIntensityAnalyzer()

while True:
    text=input()
    result=sia.polarity_scores(text)
    # pd.DataFrame(result)
    print(result)