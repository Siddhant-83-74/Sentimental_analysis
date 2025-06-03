from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import nltk

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


data = []
print("Enter the URL of the youtube video")
inp=input()
youtube_video_url = inp
service = Service(ChromeDriverManager().install())

with webdriver.Chrome(service=service) as driver:
    wait = WebDriverWait(driver, 30)  # Increased wait time
    driver.get(youtube_video_url)

    # Scroll 3 times to load comments
    for item in range(10):
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
        time.sleep(5)

    # Wait for comments to be visible and scrape them
    for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ytd-comment-thread-renderer #content-text"))):
        data.append(comment.text)

df = pd.DataFrame(data, columns=['comment'])
df.index.name = 'Comment_Number'
df.to_csv('youtube_comments.csv')

# Output results
print(f"Total comments collected: {len(df)}")
print("\nSample comments:")
print(df.head(10))

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

# Visualizing the sentiment distribution in a single window
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create 1 row and 3 columns for the plots

# 1. Pie Chart for sentiment distribution
sentiment_counts = {
    'Positive': len(positive_comments),
    'Neutral': len(neutral_comments),
    'Negative': len(negative_comments)
}

axes[0].pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ff6666'])
axes[0].set_title("Sentiment Distribution of Comments")
axes[0].axis('equal')

# 2. Bar Plot for sentiment count
sentiment_labels = ['Positive', 'Neutral', 'Negative']
sentiment_values = [len(positive_comments), len(neutral_comments), len(negative_comments)]

sns.barplot(x=sentiment_labels, y=sentiment_values, palette="Blues_d", ax=axes[1])
axes[1].set_title("Sentiment Count of Comments")
axes[1].set_ylabel("Number of Comments")

# 3. Sentiment score distribution (RoBERTa)
sns.histplot(result_final['roberta_pos'], kde=True, color='green', label='Positive', bins=20, stat='density', ax=axes[2])
sns.histplot(result_final['roberta_neu'], kde=True, color='blue', label='Neutral', bins=20, stat='density', ax=axes[2])
sns.histplot(result_final['roberta_neg'], kde=True, color='red', label='Negative', bins=20, stat='density', ax=axes[2])

axes[2].set_title("Sentiment Score Distribution (RoBERTa)")
axes[2].set_xlabel("Sentiment Score")
axes[2].set_ylabel("Density")
axes[2].legend()

# Display all charts
plt.tight_layout()  # Automatically adjust subplot parameters to fit the figure area
plt.show()

# You can also print the first few rows of the result
print(result_final.head())