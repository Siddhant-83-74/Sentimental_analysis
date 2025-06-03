import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import nltk
# from nltk.tokenize import word_tokenize

plt.style.use('fast')

# import nltk
# nltk.download('punkt')
df = pd.read_csv('/Users/siddhantjain814/Documents/AI_Project/AmazonReviews/Reviews.csv')
# df.head(n=5)
# print(df['Score'].value_counts().sort_index())
sia =SentimentIntensityAnalyzer()

# ax = df['Score'].value_counts().sort_index().plot(kind='bar',title='Count of Reviews by Stars',figsize=(10,5))
# ax.set_xlabel('Review Stars')
# plt.show()
eg=df['Text'][570]
# print(eg)
token = nltk.word_tokenize(eg)
# print(token)_tag()
tagged=nltk.pos_tag(token)
# print(tagged)
# print(df.columns)
df=df.head(500)

result={}
for i,row in tqdm(df.iterrows()):
    text=row['Text']
    id=row['Id']
    result[id]=sia.polarity_scores(text)
# print(result)
vaders=pd.DataFrame(result).T
vaders=vaders.reset_index().rename(columns={'index':'Id'})
vaders=vaders.merge(df,how='left')
print(vaders)

bx = sns.barplot(data=vaders,x='Score',y='compound')
bx.set_title("Compound scores")
plt.show()

