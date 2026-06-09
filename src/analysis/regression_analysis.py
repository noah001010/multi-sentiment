import pandas as pd
import statsmodels.api as sm
from typing import str

def run_regression_analysis(emotions_csv: str, financial_csv: str, output_summary: str = "regression_summary.txt"):
    """
    出力された感情データ（Negative Emotions, VoiceTone）と、
    1分足の株価リターンなどの金融高頻度データを用いて回帰分析を行うサンプル。
    
    Args:
        emotions_csv: 抽出したマルチモーダル感情指標のCSVファイルパス
        financial_csv: 1分足の金融データ（株価リターン等）のCSVファイルパス
        output_summary: 回帰分析の結果サマリーを保存するテキストファイルパス
    """
    print("回帰分析用のデータを読み込みます...")
    
    # 1. 感情データの読み込み
    # timestamp(秒), facial_negative_emotion, voice_tone などを保持
    df_emo = pd.read_csv(emotions_csv)
    
    # 2. 金融データの読み込み
    # 期待される形式: datetime列（時間情報）、および return列（1分間のリターン）
    df_fin = pd.read_csv(financial_csv)
    
    # === データの前処理と結合 ===
    
    # 感情データ側を、金融データ側の時間粒度（1分足など）に合わせるための処理
    # 例として、会見の開始時刻(conference_start_time)を定義し、経過秒数(timestamp)を実時刻に変換する
    # ここでは仮の会見開始時刻として 2026-05-13 15:30:00 を設定します。
    # 実際にはデータに合わせて動的に取得するか、引数で渡してください。
    conference_start_time = pd.to_datetime("2026-05-13 15:30:00")
    df_emo['datetime'] = conference_start_time + pd.to_timedelta(df_emo['timestamp'], unit='s')
    
    # 感情データを1分単位で集計（リサンプリング）
    df_emo.set_index('datetime', inplace=True)
    df_emo_1min = df_emo.resample('1min').mean().reset_index()
    
    # 金融データ側のdatetimeをDatetime型に変換
    if 'datetime' in df_fin.columns:
        df_fin['datetime'] = pd.to_datetime(df_fin['datetime'])
    else:
        # datetimeカラムが無い場合は、便宜的にインデックスをタイムスタンプとするなどの処理が必要
        print("エラー: 金融データに 'datetime' カラムが存在しません。")
        return
        
    # datetimeをキーにしてデータを結合 (inner joinで時間が一致する部分のみ取得)
    df_merged = pd.merge(df_fin, df_emo_1min, on='datetime', how='inner')
    
    # 欠損値があれば除去
    df_merged = df_merged.dropna()
    
    if df_merged.empty:
        print("警告: 感情データと金融データの結合結果が空です。時間帯が一致しているか確認してください。")
        return

    print(f"結合されたデータ件数: {len(df_merged)} 件")
    
    # === 回帰分析の実行 (statsmodels) ===
    # 目的変数(Y): 株価リターンなど
    Y = df_merged['return']
    
    # 説明変数(X): 表情と音声の感情スコア
    # 定数項(切片)を追加
    X = df_merged[['facial_negative_emotion', 'voice_tone']]
    X = sm.add_constant(X)
    
    # OLS (最小二乗法) モデルの構築とフィッティング
    # 必要に応じて、金融の文脈に即した Newey-West のHAC標準誤差(cov_type='HAC')などを使用します。
    model = sm.OLS(Y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    
    # 結果の出力
    print("\n--- 回帰分析結果 ---")
    print(results.summary())
    
    # 結果をテキストファイルに保存
    with open(output_summary, "w") as f:
        f.write(results.summary().as_text())
        
    print(f"\n分析サマリーを {output_summary} に保存しました。")


if __name__ == "__main__":
    # 実行例（データファイルが存在する前提）
    # run_regression_analysis(
    #     emotions_csv="data/multimodal_emotions.csv",
    #     financial_csv="data/high_freq_stock_returns.csv",
    #     output_summary="analysis_results/regression_summary.txt"
    # )
    pass
