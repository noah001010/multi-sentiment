#!/usr/bin/env python
"""
run_step8_regression.py
=======================
Step 7 の統合データ（CSV）と、為替データ（CSV）を読み込んで結合し、
感情指標が市場（為替リターン）に与える影響について OLS 回帰分析を行います。

HAC標準誤差（Newey-West）によるロバスト推計と多重共線性（VIF）チェックに対応。

使い方:
  python scripts/run_step8_regression.py \
    [--integrated_path output/integrated_results.csv] \
    [--financial_path data/DAT_ASCII_USDJPY_M1_2023.csv] \
    [--start_time "2023-06-16 15:30:00"]
"""
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# プロジェクトのルートディレクトリをシステムパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.end_to_end_sentiment_pipeline import load_and_filter_forex_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Step8-Regression")


def main():
    parser = argparse.ArgumentParser(description="Step 8: 金融データマージ & OLS回帰分析")
    parser.add_argument(
        "--integrated_path",
        type=str,
        default="output/integrated_results.csv",
        help="Step 7 で出力した最終統合結果CSVのパス",
    )
    parser.add_argument(
        "--financial_path",
        type=str,
        default="data/DAT_ASCII_USDJPY_M1_2023.csv",
        help="USD/JPYの為替ヒストリカルデータ(HistData形式)または為替リターンCSVのパス",
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="2023-06-16 15:30:00",
        help="会見開始の基準日時 (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--governor_only",
        action="store_true",
        default=True,
        help="回帰分析に総裁 (is_governor == True) の区間のみを使用するかどうか",
    )
    args = parser.parse_args()

    integrated_path = Path(args.integrated_path)
    fin_path = Path(args.financial_path)

    if not integrated_path.exists():
        logger.error(f"統合結果ファイルが見つかりません: {integrated_path}. 先に Step 7 を実行してください。")
        sys.exit(1)

    if not fin_path.exists():
        logger.error(f"為替金融データファイルが見つかりません: {fin_path}")
        sys.exit(1)

    # 1. 統合データのロードとフィルタリング
    logger.info("統合結果データをロード中...")
    df_integ = pd.read_csv(integrated_path)

    if args.governor_only and "is_governor" in df_integ.columns:
        logger.info("分析対象を総裁（is_governor == True）の発話区間に絞り込みます。")
        df_integ = df_integ[df_integ["is_governor"] == True].copy()
    
    if df_integ.empty:
        logger.error("分析対象データが空です。総裁IDの設定などを確認してください。")
        sys.exit(1)

    # 2. 為替データのロードと対数収益率（return）の計算
    logger.info(f"金融為替データを処理中 (基準会見時刻: {args.start_time})...")
    if "DAT_ASCII" in fin_path.name:
        # HistData.com 形式
        df_fin = load_and_filter_forex_data(str(fin_path), args.start_time)
    else:
        # 一般的な return 列を含むCSV
        df_fin = pd.read_csv(fin_path)
        df_fin['datetime'] = pd.to_datetime(df_fin['datetime'])
        if 'return' not in df_fin.columns:
            logger.error("金融CSVに 'return' カラムが存在しません。")
            sys.exit(1)

    # 3. 感情データと為替データの時間マージ (1分単位)
    conference_start_time = pd.to_datetime(args.start_time)
    
    # 秒数 timestamp から実際の日時を算出してインデックス化
    df_integ['datetime'] = conference_start_time + pd.to_timedelta(df_integ['start'], unit='s')
    df_integ.set_index('datetime', inplace=True)
    
    # 1分単位でリサンプリングして平均
    logger.info("感情特徴量を1分足単位にリサンプリングしてアライメント中...")
    df_integ_1min = df_integ.resample('1min').mean().reset_index()

    # マージ
    df_merged = pd.merge(df_fin, df_integ_1min, on='datetime', how='inner')
    df_merged = df_merged.dropna(subset=['return', 'face_emotion_score', 'audio_emotion_score', 'text_score'])

    if df_merged.empty:
        logger.error(
            f"データ結合に失敗しました。会見の開始日時（{args.start_time}）と、"
            "為替データの日時が正しく一致しているか確認してください。"
        )
        sys.exit(1)

    logger.info(f"アライメント完了。有効サンプルサイズ (N): {len(df_merged)}")

    # 4. OLS回帰分析 (HAC robust standard errors) の実行
    Y = df_merged['return']
    # 説明変数 (表情感情、音声感情、テキスト感情スコア)
    X_vars = ['face_emotion_score', 'audio_emotion_score', 'text_score']
    X = df_merged[X_vars]
    X_with_const = sm.add_constant(X)

    logger.info("OLS回帰分析（Newey-West HAC標準誤差、maxlags=1）を実行します...")
    model = sm.OLS(Y, X_with_const)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})

    # 5. 結果の出力
    print("\n============================== 回帰分析 結果サマリー ==============================")
    print(results.summary())
    print("\n============================== 詳細な統計値の抽出 ==============================")
    
    # 決定係数
    print(f"[モデル適合度]")
    print(f"  Adj. R-squared: {results.rsquared_adj:.4f}")
    print("\n[各変数の係数と有意性]")
    
    def get_stars(p_val):
        if p_val < 0.01: return '***'
        elif p_val < 0.05: return '**'
        elif p_val < 0.1: return '*'
        return ''
        
    for var in X_with_const.columns:
        coef = results.params[var]
        p_val = results.pvalues[var]
        stars = get_stars(p_val)
        print(f"  {var:<25} : 係数 = {coef:>8.4f}, p値 = {p_val:>6.4f} {stars}")
    print("  (有意水準: *** p<0.01, ** p<0.05, * p<0.1)")

    # VIF 多重共線性検証
    print("\n[多重共線性 (VIF) の確認]")
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    
    for _, row in vif_data.iterrows():
        if row['Variable'] != 'const':
            vif_val = row['VIF']
            warning = " [!] 注意 (VIF > 10)" if vif_val > 10 else ""
            print(f"  {row['Variable']:<25} : VIF = {vif_val:.4f}{warning}")


if __name__ == "__main__":
    main()
