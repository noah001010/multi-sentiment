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
import yfinance as yf

# プロジェクトのルートディレクトリをシステムパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def load_and_filter_forex_data(csv_path: str, conference_start_time_str: str):
    """
    HistData.com の汎用ASCIIフォーマット (USD/JPY 1分足など) を読み込み、
    指定された会見開始時刻から1時間分のリターンと、会見前30分間のドリフトを抽出する。
    """
    print(f"[{csv_path}] HistDataの読み込みと前処理を開始します...")
    df = pd.read_csv(csv_path, sep=';', header=None, 
                     names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S')
    # HistData.com は DSTなしの米国東部標準時(EST=GMT-5)固定。日本(JST=GMT+9)とは常に14時間時差。
    df['datetime'] = df['datetime'] + pd.Timedelta(hours=14)
    df.sort_values('datetime', inplace=True)
    df['return'] = np.log(df['close'] / df['close'].shift(1)) * 100
    
    start_time = pd.to_datetime(conference_start_time_str)
    
    # プレドリフト (前30分間の変化率 bp)
    pre_start = start_time - pd.Timedelta(minutes=30)
    pre_df = df[(df['datetime'] >= pre_start) & (df['datetime'] <= start_time)]
    if not pre_df.empty and len(pre_df) >= 2:
        close_start = pre_df.iloc[0]['close']
        close_end = pre_df.iloc[-1]['close']
        pre_drift = (close_end / close_start - 1) * 10000
    else:
        pre_drift = np.nan
        
    end_time = start_time + pd.Timedelta(hours=1)
    filtered_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)].copy()
    filtered_df.dropna(subset=['return'], inplace=True)
    print(f"  -> 抽出されたデータ件数: {len(filtered_df)}件 (期間: {start_time} 〜 {end_time})")
    print(f"  -> 会見前ドリフト(bp): {pre_drift:.2f}" if not pd.isna(pre_drift) else "  -> 会見前ドリフト(bp): N/A")
    return filtered_df[['datetime', 'return']], pre_drift

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
        type=str,
        default="true",
        help="回帰分析に総裁 (is_governor == True) の区間のみを使用するかどうか (true/false)",
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

    is_gov_only = str(args.governor_only).lower() == 'true'
    if is_gov_only and "is_governor" in df_integ.columns:
        logger.info("分析対象を総裁（is_governor == True）の発話区間に絞り込みます。")
        df_integ = df_integ[df_integ["is_governor"] == True].copy()
    
    if df_integ.empty:
        logger.error("分析対象データが空です。総裁IDの設定などを確認してください。")
        sys.exit(1)

    conference_start_time = pd.to_datetime(args.start_time)

    # 2. 為替データのロードと対数収益率（return）の計算
    logger.info(f"金融為替データを処理中 (基準会見時刻: {args.start_time})...")
    if "DAT_ASCII" in fin_path.name:
        # HistData.com 形式
        df_fin, pre_drift = load_and_filter_forex_data(str(fin_path), args.start_time)
    else:
        # 一般的な return 列を含むCSV
        df_fin = pd.read_csv(fin_path)
        df_fin['datetime'] = pd.to_datetime(df_fin['datetime'])
        if 'return' not in df_fin.columns:
            logger.error("金融CSVに 'return' カラムが存在しません。")
            sys.exit(1)
        pre_drift = np.nan

    # === 追加: コントロール変数の抽出 ===
    logger.info("コントロール変数 (MPU, Policy Rate, TOPIX) を抽出中...")
    try:
        df_mpu = pd.read_excel('data/Japan_Policy_Uncertainty_Data.xlsx')
        # 年月に合致する行の3列目（インデックス2）の値をMPU指数として取得
        mpu_val = df_mpu[(df_mpu.iloc[:, 0] == conference_start_time.year) & (df_mpu.iloc[:, 1] == conference_start_time.month)].iloc[:, 2].values
        mpu_japan = float(mpu_val[0]) if len(mpu_val) > 0 else np.nan
    except Exception as e:
        logger.warning(f"MPUデータの読み込みに失敗しました: {e}")
        mpu_japan = np.nan

    try:
        df_rate = pd.read_csv('data/boj_policy_rate.csv')
        df_rate['meeting_date'] = pd.to_datetime(df_rate['meeting_date'])
        df_rate = df_rate.sort_values('meeting_date').reset_index(drop=True)
        
        current_date = pd.to_datetime(conference_start_time.strftime('%Y-%m-%d'))
        
        # 過去の会合日を取得
        past_meetings = df_rate[df_rate['meeting_date'] < current_date]
        prev_meeting_date = past_meetings.iloc[-1]['meeting_date'] if not past_meetings.empty else pd.NaT

        idx_match = df_rate[df_rate['meeting_date'] == current_date].index
        if len(idx_match) > 0:
            idx = idx_match[0]
            rate_change_bp = float(df_rate.loc[idx, 'rate_change_bp'])
            ycc_change_dummy = float(df_rate.loc[idx, 'ycc_change_dummy'])
        else:
            # CSVに当日がない場合は変更なし（0）とみなす
            rate_change_bp = 0.0
            ycc_change_dummy = 0.0
            logger.info(f"{current_date.month}/{current_date.day}のデータはCSVにないため、直近の会合データを参照し、据え置き（Rate_Change_BP=0, YCC=0）として処理しました")
    except Exception as e:
        logger.warning(f"政策金利データの読み込みに失敗しました: {e}")
        rate_change_bp, ycc_change_dummy, prev_meeting_date = np.nan, np.nan, pd.NaT

    try:
        if not pd.isna(prev_meeting_date):
            # yfinanceを使って日経平均(^N225)のデータを取得
            start_dl = '2022-01-01'
            end_dl = (current_date + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            df_n225 = yf.download('^N225', start=start_dl, end=end_dl, progress=False)
            
            if not df_n225.empty:
                df_n225 = df_n225.reset_index()
                # 日付列の名前は 'Date'
                df_n225['Date'] = pd.to_datetime(df_n225['Date']).dt.tz_localize(None)
                
                # 前回の会合日（または直前の営業日）の終値
                prev_close_rows = df_n225[df_n225['Date'] <= prev_meeting_date].sort_values('Date')
                # yfinance のカラムはマルチインデックスになる場合があるため、float()に変換する前にilocを使用
                prev_close = float(prev_close_rows.iloc[-1]['Close'].iloc[0] if isinstance(prev_close_rows.iloc[-1]['Close'], pd.Series) else prev_close_rows.iloc[-1]['Close']) if not prev_close_rows.empty else np.nan

                # 今回の会合日の前営業日の終値
                biz_days = df_n225[df_n225['Date'] < current_date].sort_values('Date')
                prev_biz_close = float(biz_days.iloc[-1]['Close'].iloc[0] if isinstance(biz_days.iloc[-1]['Close'], pd.Series) else biz_days.iloc[-1]['Close']) if not biz_days.empty else np.nan

                if not pd.isna(prev_close) and not pd.isna(prev_biz_close) and prev_close != 0:
                    market_conditions = float((prev_biz_close / prev_close - 1) * 100)
                else:
                    market_conditions = np.nan
                    logger.warning("日経平均の前回終値または前営業日終値が取得できませんでした。")
            else:
                market_conditions = np.nan
                logger.warning("yfinanceから日経平均のデータが取得できませんでした。")
        else:
            market_conditions = np.nan
            logger.warning("前回の会合日が特定できないため日経平均を計算できませんでした。")
    except Exception as e:
        logger.warning(f"日経平均データ(yfinance)の読み込みに失敗しました: {e}")
        market_conditions = np.nan
    # ====================================
    
    # 3. 感情データと為替データの時間マージ (1分単位)
    
    # 秒数 timestamp から実際の日時を算出してインデックス化
    df_integ['datetime'] = conference_start_time + pd.to_timedelta(df_integ['start'], unit='s')
    df_integ.set_index('datetime', inplace=True)
    
    # 1分単位でリサンプリングして平均
    logger.info("感情特徴量を1分足単位にリサンプリングしてアライメント中...")
    df_integ_1min = df_integ.resample('1min').mean(numeric_only=True).reset_index()

    # マージ
    available_vars = [v for v in ['text_score', 'face_emotion_score', 'audio_emotion_score', 'face_arousal_score', 'audio_arousal_score'] if v in df_integ_1min.columns]
    df_merged = pd.merge(df_fin, df_integ_1min, on='datetime', how='inner')
    
    # コントロール変数の付与
    df_merged['USDJPY_Pre_Drift'] = pre_drift
    df_merged['MPU_Japan'] = mpu_japan
    df_merged['Rate_Change_BP'] = rate_change_bp
    df_merged['YCC_Change_Dummy'] = ycc_change_dummy
    df_merged['Market_Conditions'] = market_conditions

    # 4. OLS回帰分析 (HAC robust standard errors) の実行
    Y = df_merged['return']
    
    # 【回帰分析の変数構成】
    # 被説明変数：USD/JPYのリターン（〇分足）
    # 説明変数：テキスト感情スコア、表情スコア、音声スコア、乖離スコア
    # コントロール変数：USDJPY_Pre_Drift, MPU_Japan, Rate_Change_BP, YCC_Change_Dummy, Market_Conditions
    
    X_vars = available_vars + ['USDJPY_Pre_Drift', 'MPU_Japan', 'Rate_Change_BP', 'YCC_Change_Dummy', 'Market_Conditions']
    # 欠損値を含む行をドロップ
    df_merged = df_merged.dropna(subset=['return'] + X_vars)

    if df_merged.empty:
        logger.error(
            f"データ結合に失敗したか、変数の欠損値によりデータが空になりました。会見の開始日時（{args.start_time}）を確認してください。"
        )
        sys.exit(1)

    logger.info(f"アライメント完了。有効サンプルサイズ (N): {len(df_merged)}")

    Y = df_merged['return']
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
