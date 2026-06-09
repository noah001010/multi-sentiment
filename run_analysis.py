import sys
sys.path.append('src/features')
from end_to_end_sentiment_pipeline import build_multimodal_features_and_analyze

build_multimodal_features_and_analyze(
    video_path="data/boj_5min.mp4",
    financial_csv="data/DAT_ASCII_USDJPY_M1_2023.csv",
    output_emotions_csv="data/multimodal_emotions.csv"
)
