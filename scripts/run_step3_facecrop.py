#!/usr/bin/env python
"""
run_step3_facecrop.py
=====================
話者分離の結果CSVに基づき、発話時間中の動画から顔画像を自動で検出して切り出します。

使い方:
  python scripts/run_step3_facecrop.py [--video_path data/boj_5min.mp4] [--diarization_path output/raw/diarization.csv] [--output_dir output/faces]
"""
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# プロジェクトのルートディレクトリをシステムパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.asd_pipeline import VisualASD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Step3-FaceCrop")


def main():
    parser = argparse.ArgumentParser(description="Step 3: 発話区間の顔画像切り出し")
    parser.add_argument(
        "--video_path",
        type=str,
        default="data/boj_5min.mp4",
        help="入力動画ファイルのパス",
    )
    parser.add_argument(
        "--diarization_path",
        type=str,
        default="output/raw/diarization.csv",
        help="Step 2 で出力した話者分離CSVのパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/faces",
        help="切り出した顔画像の保存先ディレクトリ",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="特定のSpeaker IDのみ抽出する場合に指定 (例: SPEAKER_00, SPEAKER_15). 指定しない場合は全発話区間を抽出",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    diar_path = Path(args.diarization_path)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        logger.error(f"入力動画ファイルが見つかりません: {video_path}")
        sys.exit(1)

    if not diar_path.exists():
        logger.error(f"話者分離ファイルが見つかりません: {diar_path}. 先に Step 2 を実行してください。")
        sys.exit(1)

    # 話者分離データの読み込みとフィルタリング
    diar_df = pd.read_csv(diar_path)
    if args.speaker:
        logger.info(f"話者 {args.speaker} の区間のみを抽出します。")
        diar_df = diar_df[diar_df["speaker"] == args.speaker]
        if diar_df.empty:
            logger.error(f"指定された話者 {args.speaker} がデータ内に見つかりません。")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 顔切り出しの実行
    asd = VisualASD()
    logger.info(f"顔画像の切り出しを開始します (ターゲット件数: {len(diar_df)} 区間)...")
    asd.extract_active_face_crops(
        video_path=str(video_path),
        asd_df=diar_df,
        output_dir=str(output_dir),
    )
    logger.info(f"顔切り出しが完了しました。画像保存先: {output_dir}")


if __name__ == "__main__":
    main()
