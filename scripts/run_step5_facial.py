#!/usr/bin/env python
"""
run_step5_facial.py
===================
Step 3 で切り出した顔画像ディレクトリから、Py-Featを用いて表情（Action Unit）の感情特徴量を抽出します。

使い方:
  python scripts/run_step5_facial.py [--crop_dir output/faces] [--output_path output/facial_features_clean.csv]
"""
import argparse
import logging
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをシステムパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.facial_analysis import FacialAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Step5-FacialAU")


def main():
    parser = argparse.ArgumentParser(description="Step 5: 表情感情特徴量抽出 (Action Unit)")
    parser.add_argument(
        "--crop_dir",
        type=str,
        default="output/faces",
        help="切り出された顔画像が格納されたディレクトリ",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/facial_features_clean.csv",
        help="表情特徴量CSVの保存パス",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Py-Feat分析時のバッチサイズ (RTX 5080用に最適化)",
    )
    args = parser.parse_args()

    crop_dir = Path(args.crop_dir)
    output_path = Path(args.output_path)

    if not crop_dir.exists() or not list(crop_dir.glob("*.jpg")):
        logger.error(
            f"顔画像ディレクトリが見つからないか、または画像（.jpg）が存在しません: {crop_dir}. "
            "先に Step 3 を実行してください。"
        )
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 分析の実行
    logger.info(f"Py-Featによる表情分析を開始します（対象フォルダ: {crop_dir}, バッチサイズ: {args.batch_size}）...")
    analyzer = FacialAnalyzer()
    
    facial_df = analyzer.process_face_crops(str(crop_dir), batch_size=args.batch_size)
    
    if facial_df.empty:
        logger.error("表情特徴量の抽出結果が空です。")
        sys.exit(1)

    # 結果の保存
    facial_df.to_csv(output_path, index=False)
    logger.info(f"表情分析完了。結果保存先: {output_path} (行数: {len(facial_df)})")


if __name__ == "__main__":
    main()
