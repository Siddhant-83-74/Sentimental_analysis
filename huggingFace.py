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


plt.style.use('fast')
sia =SentimentIntensityAnalyzer()

df = pd.read_csv('/Users/siddhantjain814/Documents/AI_Project/youtube_comments.csv')
eg=df['comment'][20]

result={}
df=df.head(20)
for i,row in tqdm(df.iterrows()):
    text=row['comment']
    id=row['Comment_Number']
    result[id]=sia.polarity_scores(text)
# print(result)
vaders=pd.DataFrame(result).T
vaders=vaders.reset_index().rename(columns={'index':'Comment_Number'})
vaders=vaders.merge(df,how='left')

# print(vaders)


MODEL=f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL)

#Run for vader
# print(eg) 
# print(sia.polarity_scores(eg))


 #Run for Roberta

def polarity_scores_roberta(eg):
    encoded_text=tokenizer(eg,return_tensors='pt')
    # print(encoded_text)
    output=model(**encoded_text)
    scores=output[0][0].detach().numpy()
    scores=softmax(scores)
    scores_dict={
        'roberta_neg':scores[0],
        'roberta_neu':scores[1],
        'roberta_pos':scores[2]


    } 
    # print(scores_dict)
    return scores_dict
result_vaders={}
result_roberta={}
res={}
for i,row in tqdm(df.iterrows()):
    try:
        text=row['comment']
        id=row['Comment_Number']
        result_vaders[id]=sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in result_vaders.items():
            vader_result_rename[f"vader_{key}"] = value
        result_roberta[id]=polarity_scores_roberta(text)
        both = {**vader_result_rename, **result_roberta}
        res[id] = both
    except RuntimeError:
        print(f'Broke for id {id}')
while True:
    textt=input()
    print(polarity_scores_roberta(textt))
# result_final = pd.DataFrame(res).T
# print(result_final.columns)
# result_final=result_final.reset_index().rename(columns={'index':'Comment_Number'})
# result_final=result_final.merge(df,how='left')
# print(result_final)







