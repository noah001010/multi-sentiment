import os
import cv2
import pandas as pd
import numpy as np
import librosa
import subprocess
import tempfile
from typing import Dict
from deepface import DeepFace
import opensmile

def extract_facial_emotions(video_path: str, global_averages: Dict[str, float], frame_interval: int = 2, window_size: int = 180) -> pd.DataFrame:
    """
    【要件1】Filippo Curti et al. (2023) に準拠した表情ネガティブ感情スコアの算出
    
    Args:
        video_path: 対象の動画ファイルパス
        global_averages: 全会見を通じた該当総裁の感情の平均値（疑似変数）
            例: {'angry': 5.0, 'disgust': 2.0, 'fear': 1.0}
        frame_interval: フレームを抽出する間隔（秒）
        window_size: 平均値を計算するための移動窓のサイズ（秒）
        
    Returns:
        タイムスタンプごとの facial_negative_emotion を含む DataFrame
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30 # デフォルトフォールバック
        
    frame_skip = int(fps * frame_interval)
    
    results = []
    frame_count = 0
    
    print(f"[{video_path}] 表情分析を開始します...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            timestamp = frame_count / fps
            try:
                # Deepfaceで感情分析 (検出できない場合を考慮して enforce_detection=False)
                # backendは高速化のため適宜 'opencv' 等に変更可能
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                
                # 複数人が検出された場合は最初の人物を使用
                emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
                
                results.append({
                    'timestamp': timestamp,
                    'anger': emotions.get('angry', 0.0),
                    'disgust': emotions.get('disgust', 0.0),
                    'fear': emotions.get('fear', 0.0)
                })
            except Exception:
                # 顔が検出できないフレームはスキップ
                pass
                
        frame_count += 1
        
    cap.release()
    
    df = pd.DataFrame(results)
    if df.empty:
        print("警告: 表情データが抽出できませんでした。")
        return pd.DataFrame(columns=['timestamp', 'facial_negative_emotion'])

    # スライディングウィンドウ（移動平均）の計算
    # 指定した window_size (秒) に含まれるサンプル数を計算
    rolling_window = max(1, window_size // frame_interval)
    
    df['anger_ma'] = df['anger'].rolling(window=rolling_window, min_periods=1).mean()
    df['disgust_ma'] = df['disgust'].rolling(window=rolling_window, min_periods=1).mean()
    df['fear_ma'] = df['fear'].rolling(window=rolling_window, min_periods=1).mean()
    
    # 疑似的な全体平均値で割り、相対的な指標を算出
    df['anger_rel'] = df['anger_ma'] / global_averages.get('angry', 1.0)
    df['disgust_rel'] = df['disgust_ma'] / global_averages.get('disgust', 1.0)
    df['fear_rel'] = df['fear_ma'] / global_averages.get('fear', 1.0)
    
    # 3つの感情の相対値の平均を Negative Emotions 指標とする
    df['facial_negative_emotion'] = (df['anger_rel'] + df['disgust_rel'] + df['fear_rel']) / 3.0
    
    return df[['timestamp', 'facial_negative_emotion']]


def extract_voice_tone(audio_path: str, chunk_interval: int = 10) -> pd.DataFrame:
    """
    【要件2】Gorodnichenko et al. (2023) のロジックに準拠しつつ、
    OpenSMILEを用いたルールベースで音声の VoiceTone スコアを算出する（日本語環境向け）
    
    Args:
        audio_path: 対象の音声ファイルパス
        chunk_interval: 音声を区切る区間（秒）
        
    Returns:
        タイムスタンプごとの voice_tone を含む DataFrame
    """
    print(f"[{audio_path}] OpenSMILEを用いた音声トーン分析を開始します...")
    
    # OpenSMILEの初期化 (eGeMAPS 特徴量セットを使用)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    # 16kHzで音声を読み込む (OpenSMILEの標準)
    audio, sr = librosa.load(audio_path, sr=16000)
    
    chunk_samples = int(chunk_interval * sr)
    total_samples = len(audio)
    
    results = []
    
    for i in range(0, total_samples, chunk_samples):
        chunk = audio[i:i + chunk_samples]
        timestamp = i / sr
        
        # 短すぎるチャンク（1秒未満）はノイズになるため無視
        if len(chunk) < sr:
            continue
            
        try:
            # 特徴量の抽出
            features = smile.process_signal(chunk, sr)
            
            # eGeMAPSから代表的な音響特徴量を取得（欠損時は0）
            # F0 (ピッチ) の平均と変動
            f0_mean = features['F0semitoneFrom27.5Hz_sma3nz_amean'].values[0] if 'F0semitoneFrom27.5Hz_sma3nz_amean' in features else 0
            f0_std = features['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'].values[0] if 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm' in features else 0
            # ラウドネス (声の大きさ)
            loudness = features['Loudness_sma3_amean'].values[0] if 'Loudness_sma3_amean' in features else 0
            # ジッター (声の粗さ。怒りなどで上昇する傾向)
            jitter = features['jitterLocal_sma3nz_amean'].values[0] if 'jitterLocal_sma3nz_amean' in features else 0
            
            # 【ルールベース感情スコアリング（擬似モデル）】
            # Positive (Happy/Surprised): 声が大きく、ピッチが高く、抑揚がある
            positive_score = (loudness * 5.0) + (f0_mean * 0.05) + (f0_std * 2.0)
            
            # Negative (Sad/Angry): 
            # Sad的特徴: 声が小さく、抑揚がない
            sad_score = 1.0 / (loudness * 5.0 + 0.1) + 1.0 / (f0_std * 2.0 + 0.1)
            # Angry的特徴: 声が大きく、ジッター(粗さ)が大きい
            angry_score = (loudness * 5.0) + (jitter * 50.0)
            negative_score = sad_score + angry_score
            
            # マイナス値などを防ぐためのクリッピング
            pos = max(0.001, positive_score)
            neg = max(0.001, negative_score)
            
            # VoiceToneの算出: (Positive - Negative) / (Positive + Negative)
            voice_tone = (pos - neg) / (pos + neg)
                
            results.append({
                'timestamp': timestamp,
                'voice_tone': voice_tone,
                'f0_mean': f0_mean,
                'loudness': loudness
            })
        except Exception as e:
            print(f"音声推論エラー (timestamp: {timestamp}): {e}")
            pass
            
    return pd.DataFrame(results)


def build_multimodal_features(video_path: str, output_csv: str):
    """
    【最終出力】表情と音声の指標を結合し、回帰分析用データフレームを作成してCSV出力
    """
    # 【疑似変数】該当総裁の全会見を通じた表情感情の平均値（事前計算済みと想定）
    global_facial_averages = {
        'angry': 0.15,
        'disgust': 0.05,
        'fear': 0.02
    }
    
    # 1. 音声抽出 (ffmpegを使用して動画からwavを抽出)
    temp_audio = tempfile.mktemp(suffix=".wav")
    print("動画から音声を抽出中...")
    subprocess.run([
        "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", temp_audio, "-y"
    ], capture_output=True)
    
    try:
        # 2. 各モジュールの分析実行
        # (解析負荷に応じて、frame_interval や chunk_interval を調整してください)
        df_facial = extract_facial_emotions(
            video_path=video_path, 
            global_averages=global_facial_averages,
            frame_interval=2, 
            window_size=180
        )
        
        df_voice = extract_voice_tone(
            audio_path=temp_audio,
            chunk_interval=10
        )
        
        # 3. データ結合 (タイムスタンプを10秒単位のビンに丸めて結合)
        print("データを結合し、回帰分析用フォーマットに整形します...")
        if not df_facial.empty:
            df_facial['time_bin'] = (df_facial['timestamp'] // 10) * 10
            df_facial_grouped = df_facial.groupby('time_bin')['facial_negative_emotion'].mean().reset_index()
        else:
            df_facial_grouped = pd.DataFrame(columns=['time_bin', 'facial_negative_emotion'])
            
        if not df_voice.empty:
            df_voice['time_bin'] = (df_voice['timestamp'] // 10) * 10
            df_voice_grouped = df_voice.groupby('time_bin')['voice_tone'].mean().reset_index()
        else:
            df_voice_grouped = pd.DataFrame(columns=['time_bin', 'voice_tone'])
            
        # 外部結合でマージし、時間順にソート
        df_merged = pd.merge(df_facial_grouped, df_voice_grouped, on='time_bin', how='outer').sort_values('time_bin')
        
        # 前後の値で欠損値を補間
        df_merged['facial_negative_emotion'] = df_merged['facial_negative_emotion'].ffill().bfill()
        df_merged['voice_tone'] = df_merged['voice_tone'].ffill().bfill()
        
        # カラム名を回帰分析用に整える
        df_merged.rename(columns={'time_bin': 'timestamp'}, inplace=True)
        
        # 4. CSV出力
        df_merged.to_csv(output_csv, index=False)
        print(f"分析完了: 結果を {output_csv} に保存しました。")
        
    finally:
        # 一時ファイルのクリーンアップ
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    # 実行例
    # TARGET_VIDEO = "data/boj_conference.mp4"
    # OUTPUT_CSV = "data/multimodal_emotions.csv"
    # build_multimodal_features(TARGET_VIDEO, OUTPUT_CSV)
    pass
