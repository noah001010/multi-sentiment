import pandas as pd
import numpy as np

# 60分ぶんのデータを作る (1分足)
start_time = pd.to_datetime("2023-06-16 15:30:00")
datetimes = [start_time + pd.Timedelta(minutes=i) for i in range(60)]

# Text sentiment
df_txt = pd.DataFrame({
    'datetime': datetimes,
    'text_sentiment': np.random.uniform(-1, 1, 60)
})
df_txt.to_csv("data/mock_text_sentiment.csv", index=False)

# Emotions (10秒ごと、360件)
timestamps = np.arange(0, 3600, 10)
df_emo = pd.DataFrame({
    'timestamp': timestamps,
    'facial_negative_emotion': np.random.uniform(0, 1, len(timestamps)),
    'voice_tone': np.random.uniform(-1, 1, len(timestamps))
})
df_emo.to_csv("data/mock_emotions.csv", index=False)

print("Mock data updated.")
