"""
compute_integrated_scores.py
============================
既存の中間CSVを読み込み、3モダリティの感情スコアを計算して
integrated_results.csv に新列を追加するスタンドアロンスクリプト。

追加される列:
  text_score            ModernBERT 回帰値（sentiment_score のエイリアス）
  audio_emotion_score   z(loudness)+z(F0_mean)-z(jitter)-z(shimmer)
  face_emotion_score    AU12 - AU04 の発話区間平均
  is_governor           speaker == governor_id かどうか
  discrepancy_score     |text-audio| + |text-face|
  discrepancy_score_3   discrepancy_score + |audio-face|

使い方:
  .venv/bin/python scripts/compute_integrated_scores.py [--governor_id SPEAKER_15]
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


# ---------------------------------------------------------------------------
# Helper: z-score 標準化
# ---------------------------------------------------------------------------

def zscore(series: pd.Series) -> pd.Series:
    """NaN を無視してz-score 正規化する。std==0 の場合は 0 を返す。"""
    mu = series.mean()
    sigma = series.std()
    if sigma == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


# ---------------------------------------------------------------------------
# Audio emotion score
# ---------------------------------------------------------------------------

def compute_audio_emotion_score(audio_df: pd.DataFrame) -> pd.Series:
    """
    audio_emotion_score = z(loudness) + z(F0_mean) - z(jitter) - z(shimmer)
    sentence_id をインデックスとして返す。
    """
    df = audio_df.copy()
    required = ["F0_mean", "jitter", "shimmer", "loudness"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"audio_features.csv に必要な列がありません: {missing}")

    df["z_loudness"] = zscore(df["loudness"])
    df["z_F0_mean"] = zscore(df["F0_mean"])
    df["z_jitter"] = zscore(df["jitter"])
    df["z_shimmer"] = zscore(df["shimmer"])

    df["audio_emotion_score"] = (
        df["z_loudness"] + df["z_F0_mean"] - df["z_jitter"] - df["z_shimmer"]
    )
    return df.set_index("sentence_id")["audio_emotion_score"]


# ---------------------------------------------------------------------------
# Face emotion score
# ---------------------------------------------------------------------------

def compute_face_emotion_score(
    face_df: pd.DataFrame,
    starts: pd.Series,
    ends: pd.Series,
    fps: float = 30.0,
) -> pd.Series:
    """
    face_emotion_score = mean(AU12 - AU04) over frames in [start, end].

    is_blink は全行 0 のため除外。
    frame 列を fps で割って timestamp に変換。
    """
    df = face_df.copy()
    required = ["frame", "AU04", "AU12"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"facial_features_clean.csv に必要な列がありません: {missing}")

    # NaN行をドロップ（AU04 に NaN がある場合がある）
    df = df.dropna(subset=["AU04", "AU12"])
    df["timestamp"] = df["frame"] / fps
    df["frame_score"] = df["AU12"] - df["AU04"]

    scores = []
    for start, end in zip(starts, ends):
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        subset = df.loc[mask, "frame_score"]
        scores.append(float(subset.mean()) if not subset.empty else float("nan"))

    return pd.Series(scores, index=starts.index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(governor_id: str = "SPEAKER_15"):
    # ------ ファイル読み込み ------
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

    # ------ バックアップ ------
    bak_path = integrated_path.with_suffix(".csv.bak")
    shutil.copy2(integrated_path, bak_path)
    logger.info(f"バックアップ作成: {bak_path}")

    # ------ text_score ------
    if "sentiment_score" not in integ.columns:
        raise ValueError("integrated_results.csv に sentiment_score 列がありません")
    integ["text_score"] = integ["sentiment_score"]
    logger.info("text_score 列を追加（sentiment_score エイリアス）")

    # ------ audio_emotion_score ------
    logger.info("audio_emotion_score を計算中...")
    audio_score_series = compute_audio_emotion_score(audio_df)

    # integ の行インデックス（= sentence_id）で紐付け
    if "sentence_id" in integ.columns:
        integ["audio_emotion_score"] = integ["sentence_id"].map(audio_score_series)
    else:
        # sentence_id がない場合は行番号で紐付け
        if len(audio_score_series) == len(integ):
            integ["audio_emotion_score"] = audio_score_series.values
        else:
            logger.warning(
                "sentence_id 列がなく行数も一致しない → audio_emotion_score を NaN で埋めます"
            )
            integ["audio_emotion_score"] = float("nan")

    logger.info(
        f"audio_emotion_score: mean={integ['audio_emotion_score'].mean():.4f}, "
        f"std={integ['audio_emotion_score'].std():.4f}"
    )

    # ------ face_emotion_score ------
    logger.info("face_emotion_score を計算中...")
    integ["face_emotion_score"] = compute_face_emotion_score(
        face_df,
        starts=integ["start"],
        ends=integ["end"],
    )
    nan_face = integ["face_emotion_score"].isna().sum()
    logger.info(
        f"face_emotion_score: mean={integ['face_emotion_score'].mean():.4f}, "
        f"NaN={nan_face}/{len(integ)}"
    )

    # NaN を 0 で補完（顔が映っていない区間）
    integ["face_emotion_score"] = integ["face_emotion_score"].fillna(0.0)

    # ------ is_governor ------
    if "speaker" in integ.columns:
        integ["is_governor"] = integ["speaker"] == governor_id
        governor_cnt = integ["is_governor"].sum()
        logger.info(f"is_governor: {governor_cnt} 行（governor_id={governor_id}）")
    else:
        logger.warning("speaker 列がないため is_governor を False で埋めます")
        integ["is_governor"] = False

    # ------ discrepancy_score ------
    logger.info("discrepancy_score を計算中...")
    t = integ["text_score"]
    a = integ["audio_emotion_score"]
    f = integ["face_emotion_score"]

    integ["discrepancy_score"] = (t - a).abs() + (t - f).abs()
    integ["discrepancy_score_3"] = integ["discrepancy_score"] + (a - f).abs()

    logger.info(
        f"discrepancy_score: mean={integ['discrepancy_score'].mean():.4f}, "
        f"max={integ['discrepancy_score'].max():.4f}"
    )

    # ------ 保存 ------
    integ.to_csv(integrated_path, index=False)
    logger.info(f"integrated_results.csv を更新しました: {integrated_path}")
    logger.info(f"新規追加列: text_score, audio_emotion_score, face_emotion_score, "
                "is_governor, discrepancy_score, discrepancy_score_3")

    # ------ サマリー出力 ------
    new_cols = [
        "text_score", "audio_emotion_score", "face_emotion_score",
        "is_governor", "discrepancy_score", "discrepancy_score_3"
    ]
    print("\n=== 追加列サマリー ===")
    print(integ[new_cols].describe().round(4))
    print(f"\ngovernor 行数: {integ['is_governor'].sum()} / {len(integ)}")


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
