#!/usr/bin/env python
"""
text 単体テスト / text_features.csv 再生成スクリプト。

既存の中間ファイル（transcription_clean.csv）を入力として
text モジュールだけを実行し、text_features.csv を出力します。

使い方:
  # デフォルト（Fin-BERT）で実行
  ./.venv/bin/python scripts/run_text_only.py

  # ModernBERT（ローカルモデル）で実行
  ./.venv/bin/python scripts/run_text_only.py \
      --model_path /home/sano/work/mbert32/model_32/checkpoint-25137

  # 入出力パスを明示指定
  ./.venv/bin/python scripts/run_text_only.py \
      --model_path /home/sano/work/mbert32/model_32/checkpoint-25137 \
      --input  output/transcription_clean.csv \
      --output output/text_features_modernbert.csv \
      --n_sample 20   # 動作確認用に先頭N行だけ処理（省略時は全件）
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# boj-emo-antigravity のルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="text モジュール単体実行")
    p.add_argument(
        "--model_path",
        default=None,
        help="ローカルHFモデルのディレクトリ (省略時はデフォルトFin-BERT)",
    )
    p.add_argument(
        "--input",
        default="output/transcription_clean.csv",
        help="入力CSVファイル (text, start, end カラムを含むこと)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="出力CSVパス (省略時は output/text_features.csv を上書き)",
    )
    p.add_argument(
        "--n_sample",
        type=int,
        default=None,
        help="動作確認用：先頭N行だけ処理する（省略時は全件）",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # --- 入力ファイル確認 ---
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"入力ファイルが見つかりません: {input_path}")
        logger.error("まず main.py でフルパイプラインを実行して transcription_clean.csv を生成してください。")
        sys.exit(1)

    # --- 出力パス決定 ---
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output/text_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 入力データ読み込み ---
    df = pd.read_csv(input_path)
    logger.info(f"入力: {input_path}  ({len(df)} 行)")

    # 最低限必要なカラム確認
    for col in ["text", "start", "end"]:
        if col not in df.columns:
            logger.error(f"入力CSVに '{col}' カラムがありません。列一覧: {list(df.columns)}")
            sys.exit(1)

    # サンプリング（動作確認用）
    if args.n_sample is not None:
        df = df.head(args.n_sample).copy()
        logger.info(f"--n_sample {args.n_sample} 指定: 先頭 {len(df)} 行だけ処理します")

    # --- TextAnalyzer 初期化 ---
    from src.features.text_analysis import TextAnalyzer

    if args.model_path:
        logger.info(f"ローカルモデルで初期化: {args.model_path}")
    else:
        logger.info("デフォルトモデル（Fin-BERT）で初期化")

    analyzer = TextAnalyzer(model_path=args.model_path)

    # --- 推論実行 ---
    texts  = df["text"].fillna("").tolist()
    starts = df["start"].tolist()
    ends   = df["end"].tolist()

    result_df = analyzer.analyze_texts(texts, starts=starts, ends=ends)

    # --- 結果プレビュー ---
    preview_cols = ["start", "end", "text", "sentiment_score", "sentiment_label", "uncertainty_score"]
    preview = result_df[[c for c in preview_cols if c in result_df.columns]].head(5)
    print("\n=== 出力プレビュー（先頭5行） ===")
    print(preview.to_string(index=False))
    print(f"\nスコア統計:\n{result_df['sentiment_score'].describe().round(4)}\n")

    # --- CSV 保存 ---
    result_df.to_csv(output_path, index=False)
    logger.info(f"✅ 保存完了: {output_path}  ({len(result_df)} 行, {len(result_df.columns)} 列)")
    logger.info(f"   カラム: {list(result_df.columns)}")


if __name__ == "__main__":
    main()
