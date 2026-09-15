"""
compute_integrated_scores.py
============================
既存の中間CSVを読み込み、学術論文準拠（Curti & Kazinnik 2023等）の
マルチモダリティ指標（OpenSMILE除外、Wav2Vec2・Py-Feat・ModernBERT標準）を計算して
integrated_results.csv に追加・更新するスクリプト。

追加・更新される主な列:
  text_score            ModernBERT 回帰値（sentiment_score のエイリアス）
  face_negative_score   Py-Feat (anger + disgust + fear) の発話区間平均
  audio_arousal         Wav2Vec2 音声感情モデル（覚醒度/興奮度）
  audio_valence         Wav2Vec2 音声感情モデル（感情価/快不快）
  is_governor           speaker == governor_id かどうか
  discrepancy_score     |text_score - audio_valence| + |text_score - (-face_negative_score)|

使い方:
  python scripts/compute_integrated_scores.py [--governor_id SPEAKER_15]
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def compute_face_negative_score(
    face_df: pd.DataFrame,
    starts: pd.Series,
    ends: pd.Series,
    fps: float = 30.0,
) -> pd.Series:
    """
    Curti & Kazinnik (2023) に準拠し、
    face_negative_score = anger + disgust + fear の発話区間平均を算出する。
    """
    df = face_df.copy()

    if "face_negative_score" not in df.columns:
        if set(["anger", "disgust", "fear"]).issubset(df.columns):
            df["face_negative_score"] = df["anger"] + df["disgust"] + df["fear"]
        else:
            logger.warning("facial_features_clean.csv に negative 構成要素 (anger, disgust, fear) が存在しません。")
            return pd.Series(np.zeros(len(starts)), index=starts.index)

    df = df.dropna(subset=["face_negative_score"])
    df["timestamp"] = df["frame"] / fps

    scores = []
    for start, end in zip(starts, ends):
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        subset = df.loc[mask, "face_negative_score"]
        scores.append(float(subset.mean()) if not subset.empty else float("nan"))

    return pd.Series(scores, index=starts.index)


def main(governor_id: str = "SPEAKER_15"):
    integrated_path = OUTPUT / "integrated_results.csv"
    audio_path = OUTPUT / "audio_features.csv"
    face_path = OUTPUT / "facial_features_clean.csv"

    for p in [integrated_path, audio_path, face_path]:
        if not p.exists():
            logger.error(f"ファイルが見つかりません: {p}")
            sys.exit(1)

    logger.info("integrated_results.csv を読み込み中...")
    integ = pd.read_csv(integrated_path)

    logger.info("audio_features.csv を読み込み中...")
    audio_df = pd.read_csv(audio_path)

    logger.info("facial_features_clean.csv を読み込み中...")
    face_df = pd.read_csv(face_path)

    # バックアップ作成
    bak_path = integrated_path.with_suffix(".csv.bak")
    shutil.copy2(integrated_path, bak_path)
    logger.info(f"バックアップ作成: {bak_path}")

    # 1. text_score
    if "sentiment_score" not in integ.columns:
        raise ValueError("integrated_results.csv に sentiment_score 列がありません")
    integ["text_score"] = integ["sentiment_score"]

    # 2. 音声指標 (Wav2Vec2: audio_arousal, audio_valence) のマージ
    logger.info("音声感情指標 (Wav2Vec2: audio_arousal, audio_valence) をマージ中...")
    audio_cols = ["audio_arousal", "audio_valence"]
    
    if "sentence_id" in integ.columns and "sentence_id" in audio_df.columns:
        for col in audio_cols:
            if col in audio_df.columns:
                series = audio_df.set_index("sentence_id")[col]
                integ[col] = integ["sentence_id"].map(series)
            else:
                logger.warning(f"audio_features.csv に '{col}' が見つかりません。")
                integ[col] = np.nan
    else:
        for col in audio_cols:
            if col in audio_df.columns and len(audio_df) == len(integ):
                integ[col] = audio_df[col].values
            else:
                integ[col] = np.nan

    # 3. 表情ネガティブ指標 (face_negative_score)
    logger.info("表情ネガティブスコア (face_negative_score: anger+disgust+fear) を計算中...")
    integ["face_negative_score"] = compute_face_negative_score(
        face_df,
        starts=integ["start"],
        ends=integ["end"],
    )
    integ["face_negative_score"] = integ["face_negative_score"].fillna(0.0)

    # 旧独自カラムおよび OpenSMILE パラメータを完全に削除
    drop_cols = ["audio_emotion_score", "face_emotion_score", "face_arousal_score", "F0_mean", "jitter", "shimmer", "loudness"]
    for col in drop_cols:
        if col in integ.columns:
            integ.drop(columns=[col], inplace=True)

    # 4. is_governor
    if "speaker" in integ.columns:
        integ["is_governor"] = integ["speaker"] == governor_id
    else:
        integ["is_governor"] = False

    # 5. 学術的 乖離度 (Discrepancy) スコアの計算
    logger.info("感情乖離度 (discrepancy_score) を計算中...")
    t = integ["text_score"]
    a = integ["audio_valence"].fillna(0.0)
    f_val = -integ["face_negative_score"]

    integ["discrepancy_score"] = (t - a).abs() + (t - f_val).abs()
    integ["discrepancy_score_3"] = integ["discrepancy_score"] + (a - f_val).abs()

    # 保存
    integ.to_csv(integrated_path, index=False)
    logger.info(f"integrated_results.csv を新仕様に更新しました: {integrated_path}")

    # サマリー表示
    cols_to_show = [
        "text_score", "face_negative_score", "audio_arousal", "audio_valence", "discrepancy_score"
    ]
    existing_show = [c for c in cols_to_show if c in integ.columns]
    print("\n=== 学術論文仕様 統合感情データ サマリー ===")
    print(integ[existing_show].describe().round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--governor_id",
        type=str,
        default="SPEAKER_15",
        help="総裁の speaker ID（デフォルト: SPEAKER_15）",
    )
    args = parser.parse_args()
    main(governor_id=args.governor_id)
