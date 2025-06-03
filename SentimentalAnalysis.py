import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import nltk

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# Setup
plt.style.use('fast')
sia = SentimentIntensityAnalyzer()

# Load the CSV file containing YouTube comments
df = pd.read_csv('/Users/siddhantjain814/Documents/AI_Project/youtube_comments.csv')
print("Columns in the CSV file:", df.columns)

# Ensure the 'comment' column exists
if 'comment' not in df.columns:
    raise ValueError("'comment' column not found in the CSV file!")

# Initialize the RoBERTa model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function for RoBERTa sentiment analysis
def polarity_scores_roberta(eg):
    encoded_text = tokenizer(eg, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

# Initialize result dictionaries
result_vaders = {}
result_roberta = {}
res = {}

# Analyze each comment
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        text = row['comment']
        id = row['Comment_Number']  # Assuming each comment has a unique 'Comment_Number'

        # VADER Sentiment Analysis (can be kept if you want VADER results too)
        result_vaders[id] = sia.polarity_scores(text)

        # RoBERTa Sentiment Analysis
        result_roberta[id] = polarity_scores_roberta(text)

        # Combine both VADER and RoBERTa results
        both = {**result_vaders[id], **result_roberta[id]}
        res[id] = both

    except RuntimeError:
        print(f'Broke for id {id}')

# Convert the results into a DataFrame
result_final = pd.DataFrame(res).T
result_final = result_final.reset_index().rename(columns={'index': 'Comment_Number'})

# Merge with the original dataframe to get more details
result_final = result_final.merge(df, how='left')

# Analyze the sentiments: count positive, negative, and neutral comments based on RoBERTa
positive_comments = result_final[result_final['roberta_pos'] > result_final['roberta_neg']]
neutral_comments = result_final[result_final['roberta_neu'] > result_final['roberta_pos']]
negative_comments = result_final[result_final['roberta_neg'] > result_final['roberta_pos']]

# Print the counts of each sentiment category
print(f"Total Positive Comments: {len(positive_comments)}")
print(f"Total Neutral Comments: {len(neutral_comments)}")
print(f"Total Negative Comments: {len(negative_comments)}")

# Optionally, save the final DataFrame with sentiment scores to a CSV
result_final.to_csv('sentiment_analysis_results.csv', index=False)

# You can also print the first few rows of the result
print(result_final.head())