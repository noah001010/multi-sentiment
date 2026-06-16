#!/usr/bin/env python
"""
run_step7_integration.py
========================
各ステップで抽出されたテキスト感情、表情特徴、音声特徴、話者分離データをマージし、
3モダリティの正規化感情スコアおよび「感情の乖離スコア（Discrepancy）」を算出します。

使い方:
  python scripts/run_step7_integration.py \
    [--text_path output/text_features.csv] \
    [--facial_path output/facial_features_clean.csv] \
    [--audio_path output/audio_features.csv] \
    [--diarization_path output/raw/diarization.csv] \
    [--output_path output/integrated_results.csv] \
    [--governor_id SPEAKER_00]
"""
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# プロジェクトのルートディレクトリをシステムパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.integrator import MultimodalIntegrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Step7-Integration")


def zscore(series: pd.Series) -> pd.Series:
    """NaN を無視して z-score 正規化する。std == 0 の場合は 0 を返す。"""
    mu = series.mean()
    sigma = series.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


def compute_audio_emotion_score(audio_df: pd.DataFrame) -> pd.Series:
    """
    audio_emotion_score = z(loudness) + z(F0_mean) - z(jitter) - z(shimmer)
    """
    df = audio_df.copy()
    required = ["F0_mean", "jitter", "shimmer", "loudness"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"音声特徴に一部の必要な列がありません: {missing}. デフォルト値 0 を使用します。")
        for col in missing:
            df[col] = 0.0

    df["z_loudness"] = zscore(df["loudness"])
    df["z_F0_mean"] = zscore(df["F0_mean"])
    df["z_jitter"] = zscore(df["jitter"])
    df["z_shimmer"] = zscore(df["shimmer"])

    df["audio_emotion_score"] = (
        df["z_loudness"] + df["z_F0_mean"] - df["z_jitter"] - df["z_shimmer"]
    )
    return df.set_index("sentence_id")["audio_emotion_score"]


def compute_face_emotion_score(
    face_df: pd.DataFrame,
    starts: pd.Series,
    ends: pd.Series,
    fps: float = 30.0,
) -> pd.Series:
    """
    face_emotion_score = mean(AU12 - AU04) over frames in [start, end].
    """
    df = face_df.copy()
    required = ["frame", "AU04", "AU12"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"表情特徴に必要な列がありません: {missing}. 表情感情スコアは 0 になります。")
        return pd.Series(np.zeros(len(starts)), index=starts.index)

    df = df.dropna(subset=["AU04", "AU12"])
    df["timestamp"] = df["frame"] / fps
    df["frame_score"] = df["AU12"] - df["AU04"]

    scores = []
    for start, end in zip(starts, ends):
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        subset = df.loc[mask, "frame_score"]
        scores.append(float(subset.mean()) if not subset.empty else float("nan"))

    return pd.Series(scores, index=starts.index)


def main():
    parser = argparse.ArgumentParser(description="Step 7: マルチモーダルデータ統合")
    parser.add_argument(
        "--text_path",
        type=str,
        default="output/text_features.csv",
        help="Step 4 で出力したテキスト感情CSVのパス",
    )
    parser.add_argument(
        "--facial_path",
        type=str,
        default="output/facial_features_clean.csv",
        help="Step 5 で出力した表情特徴量CSVのパス",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="output/audio_features.csv",
        help="Step 6 で出力した音声特徴量CSV의パス",
    )
    parser.add_argument(
        "--diarization_path",
        type=str,
        default="output/raw/diarization.csv",
        help="Step 2 で出力した話者分離CSVのパス",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/integrated_results.csv",
        help="最終統合結果CSVの保存パス",
    )
    parser.add_argument(
        "--governor_id",
        type=str,
        default="SPEAKER_00",
        help="分析対象とする総裁の話者ID (デフォルト: SPEAKER_00)",
    )
    args = parser.parse_args()

    text_path = Path(args.text_path)
    facial_path = Path(args.facial_path)
    audio_path = Path(args.audio_path)
    diar_path = Path(args.diarization_path)
    output_path = Path(args.output_path)

    # 必須ファイルチェック
    if not text_path.exists():
        logger.error(f"テキスト感情ファイルが見つかりません: {text_path}. 先に Step 4 を実行してください。")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 各データフレームのロード
    logger.info("各種特徴量ファイルをロード中...")
    text_df = pd.read_csv(text_path)
    
    facial_df = pd.read_csv(facial_path) if facial_path.exists() else pd.DataFrame(columns=["frame", "AU04", "AU12", "is_blink"])
    if not facial_path.exists():
        logger.warning(f"表情ファイル {facial_path} が存在しないため、ダミー値で補完します。")
        
    audio_df = pd.read_csv(audio_path) if audio_path.exists() else pd.DataFrame(columns=["sentence_id", "jitter", "shimmer", "F0_mean", "loudness"])
    if not audio_path.exists():
        logger.warning(f"音声ファイル {audio_path} が存在しないため、ダミー値で補完します。")

    diar_df = pd.read_csv(diar_path) if diar_path.exists() else pd.DataFrame(columns=["start", "end", "speaker"])
    if not diar_path.exists():
        logger.warning(f"話者分離ファイル {diar_path} が存在しないため、ダミー値で補完します。")

    # 1. タイムスタンプに基づくデータ結合 (MultimodalIntegrator)
    integrator = MultimodalIntegrator()
    text_audio = text_df.copy()
    if not audio_df.empty:
        # sentence_id で text と audio を結合
        text_audio = pd.merge(text_df, audio_df, left_index=True, right_on="sentence_id", how="left")
        
    logger.info("特徴量のアライメントとマージを実行中...")
    final_df = integrator.align_and_merge(
        text_df=text_audio,
        visual_df=facial_df,
        audio_prosody=audio_df,
        diarization_df=diar_df
    )

    # 2. スコア正規化・感情乖離度の算出
    logger.info("3モダリティ感情スコアおよび乖離度(Discrepancy)の計算を実行中...")
    
    # 2-1. text_score (BERT値)
    final_df["text_score"] = final_df["sentiment_score"]

    # 2-2. audio_emotion_score
    if not audio_df.empty:
        audio_score_series = compute_audio_emotion_score(audio_df)
        final_df["audio_emotion_score"] = final_df["sentence_id"].map(audio_score_series)
    else:
        final_df["audio_emotion_score"] = 0.0
    final_df["audio_emotion_score"] = final_df["audio_emotion_score"].fillna(0.0)

    # 2-3. face_emotion_score
    if not facial_df.empty:
        final_df["face_emotion_score"] = compute_face_emotion_score(
            facial_df,
            starts=final_df["start"],
            ends=final_df["end"]
        )
    else:
        final_df["face_emotion_score"] = 0.0
    final_df["face_emotion_score"] = final_df["face_emotion_score"].fillna(0.0)

    # 2-4. is_governor
    if "speaker" in final_df.columns:
        final_df["is_governor"] = final_df["speaker"] == args.governor_id
    else:
        final_df["is_governor"] = False

    # 2-5. 乖離度 (Discrepancy)
    t = final_df["text_score"]
    a = final_df["audio_emotion_score"]
    f = final_df["face_emotion_score"]

    final_df["discrepancy_score"] = (t - a).abs() + (t - f).abs()
    final_df["discrepancy_score_3"] = final_df["discrepancy_score"] + (a - f).abs()

    # 3. 統合データの保存
    final_df.to_csv(output_path, index=False)
    logger.info(f"データ統合完了。結果保存先: {output_path} (行数: {len(final_df)})")

    # 統計サマリーの表示
    new_cols = [
        "text_score", "audio_emotion_score", "face_emotion_score",
        "is_governor", "discrepancy_score", "discrepancy_score_3"
    ]
    print("\n=== 統合感情スコアサマリー ===")
    print(final_df[new_cols].describe().round(4))
    print(f"\n総裁 (governor) の発話行数: {final_df['is_governor'].sum()} / 全発話数 {len(final_df)}")


if __name__ == "__main__":
    main()
