import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# --- モック感情データの作成 (1秒間隔で300秒) ---
n_seconds = 300
timestamps = np.arange(0, n_seconds, 10) # 10秒ごと
df_emo = pd.DataFrame({
    'timestamp': timestamps,
    'facial_negative_emotion': np.random.normal(0.5, 0.1, len(timestamps)),
    'voice_tone': np.random.uniform(-1, 1, len(timestamps))
})
df_emo.to_csv("data/mock_emotions.csv", index=False)

# --- モック金融データの作成 (1分ごと) ---
start_time = datetime(2026, 5, 13, 15, 30)
times = [start_time + timedelta(minutes=i) for i in range(10)]
df_fin = pd.DataFrame({
    'datetime': times,
    'return': np.random.normal(0.001, 0.005, len(times)) # ダミーの1分足リターン
})
df_fin.to_csv("data/mock_financial.csv", index=False)

# --- モックテキストセンチメントの作成 (1分ごと) ---
df_txt = pd.DataFrame({
    'datetime': times,
    'text_sentiment': np.random.uniform(-1, 1, len(times))
})
df_txt.to_csv("data/mock_text_sentiment.csv", index=False)

print("モックデータを作成しました。")
