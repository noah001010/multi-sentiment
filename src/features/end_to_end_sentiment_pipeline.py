import os
import cv2
import pandas as pd
import numpy as np
import librosa
import subprocess
import tempfile
from typing import Dict, Optional
from deepface import DeepFace
import opensmile
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from faster_whisper import WhisperModel
from transformers import pipeline
import gc
import torch
import tensorflow as tf

def extract_facial_emotions(video_path: str, global_averages: Dict[str, float], frame_interval: int = 2, window_size: int = 180, cache_path: Optional[str] = None) -> pd.DataFrame:
    """
    動画から表情のネガティブ感情スコアを算出する。

    【仕様: Filippo Curti et al. (2023) に準拠】
    1. 動画から指定間隔（frame_interval）でフレームを抽出する。
    2. DeepFaceを用いて各フレームの感情分類を行い、「怒り(Anger)」「嫌悪(Disgust)」「恐れ(Fear)」の3つのスコアを抽出する。
    3. 指定した時間区間（window_size、例：3分間）におけるこれら3つのスコアの移動平均を計算する。
    4. 各移動平均を、該当総裁の全会見を通じた全体平均値（global_averages）で割ることで相対化する。
    5. 相対化された3つの感情スコアの平均値を、最終的な『Negative Emotions』指標として算出する。
    
    Args:
        video_path (str): 対象の動画ファイルパス
        global_averages (Dict[str, float]): 全会見を通じた該当総裁の感情の全体平均値（疑似変数）
            例: {'angry': 5.0, 'disgust': 2.0, 'fear': 1.0}
        frame_interval (int): フレームを抽出する間隔（秒）。デフォルトは2秒。
        window_size (int): 平均値を計算するための移動窓のサイズ（秒）。デフォルトは180秒。
        
    Returns:
        pd.DataFrame: 'timestamp' と 'facial_negative_emotion' を含む DataFrame
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[{video_path}] 表情分析のキャッシュを読み込みます: {cache_path}")
        return pd.read_csv(cache_path)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
        
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
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                emotions = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
                
                results.append({
                    'timestamp': timestamp,
                    'anger': emotions.get('angry', 0.0),
                    'disgust': emotions.get('disgust', 0.0),
                    'fear': emotions.get('fear', 0.0)
                })
            except Exception:
                pass
                
        frame_count += 1
        
    cap.release()
    
    df = pd.DataFrame(results)
    if df.empty:
        print("警告: 表情データが抽出できませんでした。")
        return pd.DataFrame(columns=['timestamp', 'facial_negative_emotion'])

    rolling_window = max(1, window_size // frame_interval)
    
    df['anger_ma'] = df['anger'].rolling(window=rolling_window, min_periods=1).mean()
    df['disgust_ma'] = df['disgust'].rolling(window=rolling_window, min_periods=1).mean()
    df['fear_ma'] = df['fear'].rolling(window=rolling_window, min_periods=1).mean()
    
    df['anger_rel'] = df['anger_ma'] / global_averages.get('angry', 1.0)
    df['disgust_rel'] = df['disgust_ma'] / global_averages.get('disgust', 1.0)
    df['fear_rel'] = df['fear_ma'] / global_averages.get('fear', 1.0)
    
    df['facial_negative_emotion'] = (df['anger_rel'] + df['disgust_rel'] + df['fear_rel']) / 3.0
    
    result_df = df[['timestamp', 'facial_negative_emotion']]
    if cache_path:
        result_df.to_csv(cache_path, index=False)
        
    tf.keras.backend.clear_session()
    gc.collect()
    
    return result_df


def extract_voice_tone(audio_path: str, chunk_interval: int = 10, cache_path: Optional[str] = None) -> pd.DataFrame:
    """
    OpenSMILEを用いて音声から VoiceTone スコアを算出する。

    【要件1: OpenSMILE特徴量の標準化】
    1. eGeMAPS特徴量（Loudness, F0_mean, F0_std, Jitter）をチャンクごとに抽出する。
    2. 全チャンクの特徴量が出揃った後、各特徴量についてZスコア（(値 - 平均) / 標準偏差）を計算し標準化を行う。
    3. Zスコアを指数関数（np.exp）で非負の値に変換した上で、以下のヒューリスティックに当てはめる。
       - Positive: Loudnessが大きく、F0が高く、F0の変動（抑揚）がある
       - Negative (Sad): Loudnessが小さく、F0の変動がない
       - Negative (Angry): Loudnessが大きく、Jitter（粗さ）が大きい
    4. Gorodnichenko et al. (2023) に準拠し、(Positive - Negative) / (Positive + Negative) で VoiceTone を算出。
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[{audio_path}] 音声トーン分析のキャッシュを読み込みます: {cache_path}")
        return pd.read_csv(cache_path)
        
    print(f"[{audio_path}] OpenSMILEを用いた音声トーン分析を開始します...")
    
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    audio, sr = librosa.load(audio_path, sr=16000)
    chunk_samples = int(chunk_interval * sr)
    
    raw_features = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        timestamp = i / sr
        
        if len(chunk) < sr:
            continue
            
        try:
            features = smile.process_signal(chunk, sr)
            raw_features.append({
                'timestamp': timestamp,
                'f0_mean': features['F0semitoneFrom27.5Hz_sma3nz_amean'].values[0] if 'F0semitoneFrom27.5Hz_sma3nz_amean' in features else 0,
                'f0_std': features['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'].values[0] if 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm' in features else 0,
                'loudness': features['Loudness_sma3_amean'].values[0] if 'Loudness_sma3_amean' in features else 0,
                'jitter': features['jitterLocal_sma3nz_amean'].values[0] if 'jitterLocal_sma3nz_amean' in features else 0
            })
        except Exception as e:
            pass
            
    df = pd.DataFrame(raw_features)
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'voice_tone'])
        
    # 特徴量の標準化（Zスコア）
    feature_cols = ['f0_mean', 'f0_std', 'loudness', 'jitter']
    for col in feature_cols:
        std = df[col].std()
        if pd.isna(std) or std == 0:
            df[f'{col}_z'] = 0.0
        else:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / std
            
    # 標準化された特徴量（Zスコア）を指数関数で非負に変換し、スコアを計算
    # 負のZスコア（平均より小さい）は 0~1 に、正のZスコアは 1 以上になる
    df['pos_score'] = np.exp(df['loudness_z']) + np.exp(df['f0_mean_z']) + np.exp(df['f0_std_z'])
    
    # Sad的要素: 声が小さく（loudness_zが負 -> exp(-loudness_z)が大きい）、抑揚がない
    df['neg_sad_score'] = np.exp(-df['loudness_z']) + np.exp(-df['f0_std_z'])
    # Angry的要素: 声が大きく、Jitterが大きい
    df['neg_angry_score'] = np.exp(df['loudness_z']) + np.exp(df['jitter_z'])
    
    df['neg_score'] = df['neg_sad_score'] + df['neg_angry_score']
    
    # VoiceToneの算出
    df['voice_tone'] = (df['pos_score'] - df['neg_score']) / (df['pos_score'] + df['neg_score'])
            
    result_df = df[['timestamp', 'voice_tone']]
    if cache_path:
        result_df.to_csv(cache_path, index=False)
        
    gc.collect()
    return result_df


def extract_text_sentiment(video_path: str, cache_path: Optional[str] = None) -> pd.DataFrame:
    """
    動画（または音声）からWhisperで文字起こし（英訳）を行い、FinBERTで感情スコアを算出する。
    
    【要件1: Whisperによる文字起こし】
    - task="translate" により日本語音声を英語テキストに変換し、タイムスタンプを取得。
    
    【要件2: FinBERTによる感情スコアリング】
    - ProsusAI/finbert を使用し、(Positive - Negative) を連続値スコアとして算出する。
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[{video_path}] テキスト感情分析のキャッシュを読み込みます: {cache_path}")
        return pd.read_csv(cache_path)
        
    print(f"[{video_path}] Whisperによる文字起こしと翻訳を開始します...")
    # モデルサイズは large-v3 を指定。GPUが利用可能なら自動的にcudaで動作します。
    # ユーザー環境（RTX 5080等）に合わせて最適化
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    
    segments, info = whisper_model.transcribe(video_path, beam_size=5, task="translate")
    
    print(f"[{video_path}] FinBERTによるテキスト感情分析を開始します...")
    # 感情分析モデルの初期化（トップ1のスコアだけでなく全ラベルの確率を取得）
    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, device=0)
    
    results = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
            
        try:
            # 感情分析の実行
            scores = sentiment_pipeline(text)[0]
            # scoresは [{'label': 'positive', 'score': 0.1}, {'label': 'negative', 'score': 0.8}, ...] のような形式
            pos_score = next((item['score'] for item in scores if item['label'] == 'positive'), 0)
            neg_score = next((item['score'] for item in scores if item['label'] == 'negative'), 0)
            
            # (Positive - Negative) で -1 〜 1 の連続値スコアを算出
            sentiment_score = pos_score - neg_score
            
            results.append({
                'start': segment.start,
                'end': segment.end,
                'timestamp': (segment.start + segment.end) / 2.0, # 中間点を代表タイムスタンプとする
                'text': text,
                'text_sentiment': sentiment_score
            })
        except Exception as e:
            print(f"FinBERT推論エラー: {e}")
            pass
            
    # メモリ解放
    del whisper_model
    del sentiment_pipeline
    gc.collect()
    torch.cuda.empty_cache()
            
    df = pd.DataFrame(results)
    if df.empty:
        print("警告: テキストデータが抽出できませんでした。")
        return pd.DataFrame(columns=['timestamp', 'text_sentiment', 'text'])
        
    result_df = df[['timestamp', 'text_sentiment', 'text']]
    if cache_path:
        result_df.to_csv(cache_path, index=False)
        
    return result_df


def load_and_filter_forex_data(csv_path: str, conference_start_time_str: str) -> pd.DataFrame:
    """
    HistData.com の汎用ASCIIフォーマット (USD/JPY 1分足など) を読み込み、
    指定された会見開始時刻から1時間分のリターンを抽出する。
    """
    print(f"[{csv_path}] HistDataの読み込みと前処理を開始します...")
    
    # 【要件1: フォーマット解析】
    # ヘッダーなし、セミコロン区切り
    # カラム: ['datetime_str', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv(csv_path, sep=';', header=None, 
                     names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'])
    
    # datetime型に変換 (例: "20230101 170000" -> format='%Y%m%d %H%M%S')
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S')
    df.sort_values('datetime', inplace=True)
    
    # 【要件2: リターンの計算】
    # 1分ごとの対数収益率 (Log Return) を計算してパーセント表記 (* 100)
    df['return'] = np.log(df['close'] / df['close'].shift(1)) * 100
    
    # 【要件3: 会見時間帯の動的スライス】
    start_time = pd.to_datetime(conference_start_time_str)
    end_time = start_time + pd.Timedelta(hours=1)
    
    # 指定範囲を抽出
    filtered_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)].copy()
    
    # NaN リターン(最初の行など)をドロップ
    filtered_df.dropna(subset=['return'], inplace=True)
    
    print(f"  -> 抽出されたデータ件数: {len(filtered_df)}件 (期間: {start_time} 〜 {end_time})")
    
    return filtered_df[['datetime', 'return']]


def run_regression_analysis(emotions_csv: str, financial_csv: str, conference_start_time_str: str = "2026-05-13 15:30:00"):
    """
    マルチモーダル感情指標と金融データを用いた回帰分析を実行し、結果を詳細に出力する。

    【本研究における変数の定義】
    - 被説明変数 (Y): 1分足等の金融リターン (financial_csv内の 'return' 列)
    - 説明変数 (X): 
        1. facial_negative_emotion: 表情のネガティブ感情スコア (Curti et al., 2023準拠)
        2. voice_tone: 音声のVoiceToneスコア (Gorodnichenko et al., 2023準拠)
    - 統制変数 (Control Variable):
        3. text_sentiment: FinBERTモデルが算出したテキスト(言語)の経済的センチメント (-1〜1)

    Args:
        emotions_csv (str): 抽出したマルチモーダル感情指標（表情・音声・言語）のCSVファイルパス
        financial_csv (str): 1分足の金融データ（株価リターン等）のCSVファイルパス
        conference_start_time_str (str): 会見開始時刻（タイムスタンプから実時刻への変換用）
    """
    print("\n--- 回帰分析用データの前処理と結合 ---")
    
    # 1. データの読み込み
    df_emo = pd.read_csv(emotions_csv)
    
    # HistDataの実データ判定と処理
    if "DAT_ASCII" in financial_csv:
        df_fin = load_and_filter_forex_data(financial_csv, conference_start_time_str)
    else:
        df_fin = pd.read_csv(financial_csv)
        df_fin['datetime'] = pd.to_datetime(df_fin['datetime'])
        
    # 2. 会見開始時刻を基準に timestamp (秒) を datetime に変換し、1分単位にリサンプリング
    conference_start_time = pd.to_datetime(conference_start_time_str)
    
    df_emo['datetime'] = conference_start_time + pd.to_timedelta(df_emo['timestamp'], unit='s')
    df_emo.set_index('datetime', inplace=True)
    df_emo_1min = df_emo.resample('1min').mean().reset_index()
    
    # 3. データの結合 (金融データに表情・音声・テキストデータを結合)
    df_merged = pd.merge(df_fin, df_emo_1min, on='datetime', how='inner')
    df_merged = df_merged.dropna()
    
    if df_merged.empty:
        print("警告: 結合データが空です。")
        return
        
    print(f"有効なサンプルサイズ (N): {len(df_merged)}")

    # === 回帰分析の実行 ===
    # 目的変数 (Y)
    Y = df_merged['return']
    
    # 説明変数 (X) + 統制変数 (Control)
    X = df_merged[['facial_negative_emotion', 'voice_tone', 'text_sentiment']]
    X_with_const = sm.add_constant(X)
    
    # HAC標準誤差(Newey-West)を用いたOLS回帰
    model = sm.OLS(Y, X_with_const)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    
    # === 結果の解釈・確認機能 ===
    print("\n============================== 回帰分析 結果サマリー ==============================")
    print(results.summary())
    print("\n============================== 詳細な統計値の抽出 ==============================")
    
    # 自由度調整済み決定係数
    print(f"[モデル適合度]")
    print(f"  Adj. R-squared: {results.rsquared_adj:.4f}")
    print("\n[各変数の係数と有意性]")
    
    def get_stars(p_val):
        if p_val < 0.01: return '***'
        elif p_val < 0.05: return '**'
        elif p_val < 0.1: return '*'
        else: return ''
        
    for var in X_with_const.columns:
        coef = results.params[var]
        p_val = results.pvalues[var]
        stars = get_stars(p_val)
        print(f"  {var:<25} : 係数 = {coef:>8.4f}, p値 = {p_val:>6.4f} {stars}")
    print("  (有意水準: *** p<0.01, ** p<0.05, * p<0.1)")
        
    print("\n[多重共線性 (VIF) の確認]")
    print("  (VIFが10を超える場合は多重共線性の疑いがあります)")
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    # 定数項を含めてVIFを計算
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    
    for _, row in vif_data.iterrows():
        if row['Variable'] != 'const':
            vif_val = row['VIF']
            warning = " [!] 注意" if vif_val > 10 else ""
            print(f"  {row['Variable']:<25} : VIF = {vif_val:.4f}{warning}")
            
    return {
        'results': results,
        'vif_data': vif_data,
        'df_merged': df_merged,
        'Y': Y,
        'X_with_const': X_with_const
    }
            

def build_multimodal_features_and_analyze(video_path: str, financial_csv: str, output_emotions_csv: str, conference_start_time_str: str = "2023-06-16 15:30:00"):
    """
    パイプライン全体を実行するラッパー関数
    """
    global_facial_averages = {'angry': 0.15, 'disgust': 0.05, 'fear': 0.02}
    
    # --- 1. 動画から音声の抽出 ---
    temp_audio = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", temp_audio, "-y"], capture_output=True)
    
    try:
        # --- 2. マルチモーダル感情特徴量の抽出 ---
        df_facial = extract_facial_emotions(video_path, global_facial_averages, cache_path="data/cache_facial.csv")
        df_voice = extract_voice_tone(temp_audio, cache_path="data/cache_voice.csv")
        df_text = extract_text_sentiment(temp_audio, cache_path="data/cache_text.csv") # Whisperには音声ファイルを渡す方が効率的
        
        # --- 3. データのマージと保存 ---
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
            
        if not df_text.empty:
            df_text['time_bin'] = (df_text['timestamp'] // 10) * 10
            df_text_grouped = df_text.groupby('time_bin')['text_sentiment'].mean().reset_index()
        else:
            df_text_grouped = pd.DataFrame(columns=['time_bin', 'text_sentiment'])
            
        # 表情、音声、テキストの3つをマージ
        df_merged = pd.merge(df_facial_grouped, df_voice_grouped, on='time_bin', how='outer')
        df_merged = pd.merge(df_merged, df_text_grouped, on='time_bin', how='outer').sort_values('time_bin')
        
        # 前後の値で補間
        df_merged['facial_negative_emotion'] = df_merged['facial_negative_emotion'].ffill().bfill()
        df_merged['voice_tone'] = df_merged['voice_tone'].ffill().bfill()
        df_merged['text_sentiment'] = df_merged['text_sentiment'].ffill().bfill()
        df_merged.rename(columns={'time_bin': 'timestamp'}, inplace=True)
        
        df_merged.to_csv(output_emotions_csv, index=False)
        print(f"\nマルチモーダル感情指標を {output_emotions_csv} に保存しました。")
        
        # --- 4. 回帰分析の実行 ---
        regression_output = run_regression_analysis(
            emotions_csv=output_emotions_csv,
            financial_csv=financial_csv,
            conference_start_time_str=conference_start_time_str
        )
        
        return {
            'regression': regression_output,
            'df_text': df_text if 'df_text' in locals() and not df_text.empty else pd.DataFrame(),
            'df_merged': regression_output['df_merged'] if regression_output else df_merged
        }
        
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    # 実行例
    # build_multimodal_features_and_analyze(
    #     video_path="data/boj_conference.mp4",
    #     financial_csv="data/high_freq_stock_returns.csv",
    #     text_sentiment_csv="data/text_sentiment.csv",
    #     output_emotions_csv="data/multimodal_emotions.csv"
    # )
    pass
