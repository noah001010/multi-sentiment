#!/usr/bin/env python
"""
run_step7_integration.py
========================
各ステップで抽出されたテキスト感情、表情特徴、音声特徴、話者分離データをマージし、
3モダリティの正規化感情スコア（学術論文準拠）および感情乖離度を算出します。

使い方:
  python scripts/run_step7_integration.py \
    [--text_path output/text_features.csv] \
    [--facial_path output/facial_features_clean.csv] \
    [--audio_path output/audio_features.csv] \
    [--diarization_path output/raw/diarization.csv] \
    [--output_path output/integrated_results.csv] \
    [--governor_id AUTO]
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


def extract_audio_dimension(audio_df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Extract a specific emotion dimension from audio features.
    """
    df = audio_df.copy()
    if col_name not in df.columns:
        logger.warning(f"音声特徴に列 '{col_name}' が見つかりません。デフォルト値 0.0 を使用します。")
        df[col_name] = 0.0
    return df.set_index("sentence_id")[col_name]


def compute_face_metric(
    face_df: pd.DataFrame,
    starts: pd.Series,
    ends: pd.Series,
    col_name: str = "face_negative_score",
    fps: float = 30.0,
) -> pd.Series:
    """
    Compute mean of a facial metric over frames in [start, end].
    Curti & Kazinnik (2023) に倣い、face_negative_score (= anger + disgust + fear) の区間平均を計算。
    """
    df = face_df.copy()
    if col_name not in df.columns:
        if set(["anger", "disgust", "fear"]).issubset(df.columns) and col_name == "face_negative_score":
            df["face_negative_score"] = df["anger"] + df["disgust"] + df["fear"]
        else:
            logger.warning(f"表情特徴に列 '{col_name}' が見つかりません。デフォルト値 0.0 になります。")
            return pd.Series(np.zeros(len(starts)), index=starts.index)

    df = df.dropna(subset=[col_name])
    df["timestamp"] = df["frame"] / fps
    df = df.sort_values("timestamp")

    scores = []
    timestamps = df["timestamp"].values
    vals = df[col_name].values

    for start, end in zip(starts, ends):
        mask = (timestamps >= (start - 2.0)) & (timestamps <= (end + 2.0))
        subset_vals = vals[mask]

        if len(subset_vals) > 0:
            scores.append(float(np.mean(subset_vals)))
        else:
            if len(timestamps) > 0:
                idx = np.argmin(np.abs(timestamps - start))
                scores.append(float(vals[idx]))
            else:
                scores.append(0.0)

    return pd.Series(scores, index=starts.index)


def main():
    parser = argparse.ArgumentParser(description="Step 7: マルチモーダルデータ統合")
    parser.add_argument(
        "--date_code",
        type=str,
        default="23_0616",
        help="会見日付コード (例: 23_0616)",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        default=None,
        help="Step 4 で出力したテキスト感情CSVのパス",
    )
    parser.add_argument(
        "--facial_path",
        type=str,
        default=None,
        help="Step 5 で出力した表情特徴量CSVのパス",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Step 6 で出力した音声特徴量CSVのパス",
    )
    parser.add_argument(
        "--diarization_path",
        type=str,
        default=None,
        help="Step 2 で出力した話者分離CSVのパス",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="最終統合結果CSVの保存パス",
    )
    parser.add_argument(
        "--governor_id",
        type=str,
        default="AUTO",
        help="分析対象とする総裁の話者ID (デフォルト: AUTO - 最も発言時間の長い話者を自動判定)",
    )
    args = parser.parse_args()

    date_dir = Path("output") / args.date_code

    def resolve_path(arg_path, candidates):
        if arg_path:
            return Path(arg_path)
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    text_path = resolve_path(args.text_path, [date_dir / "text" / "text_features.csv", Path("output/text_features.csv")])
    facial_path = resolve_path(args.facial_path, [date_dir / "face" / "facial_features.csv", date_dir / "face" / "facial_features_clean.csv", Path("output/facial_features_clean.csv"), Path("output/facial_features.csv")])
    audio_path = resolve_path(args.audio_path, [date_dir / "audio" / "audio_features.csv", Path("output/audio_features.csv")])
    diar_path = resolve_path(args.diarization_path, [date_dir / "text" / "diarization.csv", Path("output/raw/diarization.csv")])
    output_path = Path(args.output_path) if args.output_path else date_dir / "integrated_results.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("各種特徴量ファイルをロード中...")
    text_df = pd.read_csv(text_path)
    facial_df = pd.read_csv(facial_path) if facial_path.exists() else pd.DataFrame()
    audio_df = pd.read_csv(audio_path) if audio_path.exists() else pd.DataFrame()
    diar_df = pd.read_csv(diar_path) if diar_path.exists() else pd.DataFrame(columns=["start", "end", "speaker"])

    # 1. アライメントとマージ
    integrator = MultimodalIntegrator()
    text_audio = text_df.copy()
    if not audio_df.empty:
        text_audio = pd.merge(text_df, audio_df, left_index=True, right_on="sentence_id", how="left")
        
    logger.info("特徴量のアライメントとマージを実行中...")
    final_df = integrator.align_and_merge(
        text_df=text_audio,
        visual_df=facial_df,
        audio_prosody=audio_df,
        diarization_df=diar_df
    )

    # 2. スコアおよび変数の設定
    logger.info("学術論文準拠の3モダリティ感情指標を整理中...")
    
    # 2-1. text_score
    final_df["text_score"] = final_df["sentiment_score"]

    # 2-2. 音声指標 (Wav2Vec2: audio_arousal, audio_valence)
    audio_cols = ["audio_arousal", "audio_valence"]
    if not audio_df.empty:
        for col in audio_cols:
            if col in audio_df.columns:
                series = extract_audio_dimension(audio_df, col)
                final_df[col] = final_df["sentence_id"].map(series)
            else:
                final_df[col] = np.nan
    else:
        for col in audio_cols:
            final_df[col] = np.nan

    # 2-3. 表情ネガティブ指標 (face_negative_score)
    if not facial_df.empty:
        final_df["face_negative_score"] = compute_face_metric(
            facial_df,
            starts=final_df["start"],
            ends=final_df["end"],
            col_name="face_negative_score"
        )
    else:
        final_df["face_negative_score"] = 0.0
    final_df["face_negative_score"] = final_df["face_negative_score"].fillna(0.0)

    # 2-4. is_governor
    governor_id = args.governor_id
    if governor_id == "AUTO" and not diar_df.empty:
        diar_df["duration"] = diar_df["end"] - diar_df["start"]
        speaker_durations = diar_df.groupby("speaker")["duration"].sum()
        governor_id = speaker_durations.idxmax()
        logger.info(f"総裁の話者IDを自動判定しました: {governor_id}")
    elif governor_id == "AUTO":
        governor_id = "SPEAKER_00"
        
    if "speaker" in final_df.columns:
        final_df["is_governor"] = final_df["speaker"] == governor_id
    else:
        final_df["is_governor"] = False

    # 2-5. 乖離度 (Discrepancy)
    t = final_df["text_score"]
    a = final_df["audio_valence"].fillna(0.0)
    f_val = -final_df["face_negative_score"]

    final_df["discrepancy_score"] = (t - a).abs() + (t - f_val).abs()
    final_df["discrepancy_score_3"] = final_df["discrepancy_score"] + (a - f_val).abs()

    # 旧独自カラムおよび OpenSMILE パラメータをドロップ
    drop_cols = ["audio_emotion_score", "face_emotion_score", "face_arousal_score", "F0_mean", "jitter", "shimmer", "loudness"]
    for col in drop_cols:
        if col in final_df.columns:
            final_df.drop(columns=[col], inplace=True)

    # 保存
    final_df.to_csv(output_path, index=False)
    logger.info(f"データ統合完了。結果保存先: {output_path} (行数: {len(final_df)})")

    new_cols = [
        "text_score", "face_negative_score", "audio_arousal", "audio_valence",
        "is_governor", "discrepancy_score"
    ]
    avail_cols = [c for c in new_cols if c in final_df.columns]
    print("\n=== 統合感情スコアサマリー ===")
    print(final_df[avail_cols].describe().round(4))
    print(f"\n総裁 (governor) の発話行数: {final_df['is_governor'].sum()} / 全発話数 {len(final_df)}")

if __name__ == "__main__":
    main()
