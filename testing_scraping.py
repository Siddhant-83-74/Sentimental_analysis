from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

data = []
youtube_video_url = "https://www.youtube.com/watch?v=uQx8jKiIDTI"
service = Service(ChromeDriverManager().install())

with webdriver.Chrome(service=service) as driver:
    wait = WebDriverWait(driver, 30)  # Increased wait time
    driver.get(youtube_video_url)

    # Scroll 3 times to load comments
    for item in range(20):
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