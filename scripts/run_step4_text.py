#!/usr/bin/env python
"""
run_step4_text.py
=================
Step 1 の文字起こし結果（CSV）を読み込み、BERTモデルを使用してテキスト感情分析（Sentiment Analysis）を実行します。

使い方:
  python scripts/run_step4_text.py [--input_path output/transcription.csv] [--output_path output/text_features.csv]
"""
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import torch

# プロジェクトのルートディレクトリをシステムパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.text_analysis import TextAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Step4-TextSentiment")


def main():
    parser = argparse.ArgumentParser(description="Step 4: テキスト感情分析")
    parser.add_argument(
        "--input_path",
        type=str,
        default="output/transcription.csv",
        help="入力文字起こしCSVのパス",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/text_features.csv",
        help="感情スコアの出力CSVパス",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="ローカルのHugging Faceモデルへのディレクトリパス (省略時はデフォルトFin-BERT)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス (cuda, cpu)",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=None,
        help="動作確認用に先頭のN件だけ分析する場合に指定",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        logger.error(f"入力文字起こしファイルが見つかりません: {input_path}. 先に Step 1 を実行してください。")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # データの読み込み
    df = pd.read_csv(input_path)
    logger.info(f"データをロードしました: {input_path} (行数: {len(df)})")

    # サンプル数制限
    if args.n_sample is not None:
        logger.info(f"先頭 {args.n_sample} 行のみをサンプリングして分析します。")
        df = df.head(args.n_sample).copy()

    # パスが揃っているか検証
    for col in ["text", "start", "end"]:
        if col not in df.columns:
            logger.error(f"入力CSVに必要なカラム '{col}' が存在しません。")
            sys.exit(1)

    # テキスト感情分析の実行
    logger.info("BERTモデルによるテキスト感情分析を実行中...")
    analyzer = TextAnalyzer(model_path=args.model_path)
    
    texts = df["text"].fillna("").tolist()
    starts = df["start"].tolist()
    ends = df["end"].tolist()

    result_df = analyzer.analyze_texts(texts, starts=starts, ends=ends)

    # 保存
    result_df.to_csv(output_path, index=False)
    logger.info(f"感情分析結果を保存しました: {output_path} (行数: {len(result_df)})")


if __name__ == "__main__":
    main()
