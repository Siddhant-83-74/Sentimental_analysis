import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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


MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

def run_gui():
    app = ttk.Window(themename="darkly")
    app.title("YouTube Comment Sentiment Analyzer")
    app.geometry("1000x700")
    app.resizable(False, False)

    frame = ttk.Frame(app, padding=20)
    frame.pack(fill='both', expand=True)

    label = ttk.Label(frame, text="Paste YouTube Video URL:", font=("Segoe UI", 14))
    label.pack(pady=(0, 10))

    url_var = tk.StringVar()
    entry = ttk.Entry(frame, textvariable=url_var, width=80, font=("Segoe UI", 12))
    entry.pack(ipady=6, pady=(0, 20))

    canvas_frame = ttk.Frame(frame)
    canvas_frame.pack(pady=20, fill='both', expand=True)

    def analyze():
        youtube_video_url = url_var.get()
        if not youtube_video_url:
            messagebox.showerror("Error", "Please paste a YouTube URL.")
            return

        data = []
        service = Service(ChromeDriverManager().install())
        with webdriver.Chrome(service=service) as driver:
            wait = WebDriverWait(driver, 30)
            driver.get(youtube_video_url)

            for item in range(10):
                wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
                time.sleep(5)

            for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ytd-comment-thread-renderer #content-text"))):
                data.append(comment.text)

        df = pd.DataFrame(data, columns=['comment'])
        df.index.name = 'Comment_Number'
        df.to_csv('youtube_comments.csv')

        df = pd.read_csv('youtube_comments.csv')
        if 'comment' not in df.columns:
            messagebox.showerror("Error", "'comment' column not found!")
            return

        sia = SentimentIntensityAnalyzer()
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model_inst = AutoModelForSequenceClassification.from_pretrained(MODEL)

        def polarity_scores_roberta(eg):
            encoded_text = tokenizer(eg, return_tensors='pt')
            output = model_inst(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            return {'roberta_neg': scores[0], 'roberta_neu': scores[1], 'roberta_pos': scores[2]}

        result_vaders, result_roberta, res = {}, {}, {}
        for i, row in df.iterrows():
            try:
                text = row['comment']
                id = row['Comment_Number']
                result_vaders[id] = sia.polarity_scores(text)
                result_roberta[id] = polarity_scores_roberta(text)
                res[id] = {**result_vaders[id], **result_roberta[id]}
            except RuntimeError:
                continue

        result_final = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Comment_Number'})
        result_final = result_final.merge(df, how='left')

        pos = result_final[result_final['roberta_pos'] > result_final['roberta_neg']]
        neu = result_final[result_final['roberta_neu'] > result_final['roberta_pos']]
        neg = result_final[result_final['roberta_neg'] > result_final['roberta_pos']]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].pie([len(pos), len(neu), len(neg)], labels=["Positive", "Neutral", "Negative"],
                    autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ff6666'])
        axes[0].set_title("Sentiment Distribution")
        axes[0].axis('equal')

        sns.barplot(x=["Positive", "Neutral", "Negative"], y=[len(pos), len(neu), len(neg)],
                    palette="Blues_d", ax=axes[1])
        axes[1].set_title("Sentiment Count")
        axes[1].set_ylabel("Comments")

        sns.histplot(result_final['roberta_pos'], kde=True, color='green', label='Positive', bins=20, stat='density', ax=axes[2])
        sns.histplot(result_final['roberta_neu'], kde=True, color='blue', label='Neutral', bins=20, stat='density', ax=axes[2])
        sns.histplot(result_final['roberta_neg'], kde=True, color='red', label='Negative', bins=20, stat='density', ax=axes[2])
        axes[2].set_title("Sentiment Score Distribution")
        axes[2].legend()

        for widget in canvas_frame.winfo_children():
            widget.destroy()
        chart = FigureCanvasTkAgg(fig, master=canvas_frame)
        chart.draw()
        chart.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)

        result_final.to_csv('sentiment_analysis_results.csv', index=False)
        download_btn.pack(pady=(20, 0))

    def download():
        dest = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
        if dest:
            pd.read_csv('sentiment_analysis_results.csv').to_csv(dest, index=False)
            messagebox.showinfo("Saved", "Results saved successfully!")

    analyze_btn = ttk.Button(frame, text="Analyze", bootstyle="primary outline", command=analyze)
    analyze_btn.pack(pady=(0, 10), ipadx=10, ipady=5)

    download_btn = ttk.Button(frame, text="Download CSV", bootstyle="success outline", command=download)
    download_btn.pack_forget()

    app.mainloop()

if __name__ == "__main__":
    run_gui()