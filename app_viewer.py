import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- ページ設定 ---
st.set_page_config(page_title="Multimodal Sentiment & Market Dashboard", layout="wide", page_icon="📈")

# ダークモード用のカスタムCSS
st.markdown("""
<style>
    body { font-family: 'Inter', sans-serif; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #ff4b4b; color: white; border: none; }
    .stButton>button:hover { background-color: #ff3333; color: white; }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

def load_and_filter_forex_data(csv_path: str, conference_start_time_str: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=';', header=None, 
                     names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S')
    df['datetime'] = df['datetime'] + pd.Timedelta(hours=14) # +14h JST offset
    df.sort_values('datetime', inplace=True)
    df['return'] = np.log(df['close'] / df['close'].shift(1)) * 100
    start_time = pd.to_datetime(conference_start_time_str)
    end_time = start_time + pd.Timedelta(hours=1)
    filtered_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)].copy()
    filtered_df.dropna(subset=['return'], inplace=True)
    return filtered_df[['datetime', 'close', 'return']]

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ Configuration")
    video_path = st.text_input("動画のパス (.mp4)", value="data/boj_conference.mp4")
    integrated_path = st.text_input("統合結果 CSVのパス", value="output/integrated_results.csv")
    forex_path = st.text_input("為替ヒストリカル CSVのパス", value="data/DAT_ASCII_USDJPY_M1_2023.csv")
    start_time_str = st.text_input("会見開始時刻", value="2023-06-16 15:30:00")
    governor_only = st.checkbox("総裁（is_governor）の発話のみに絞る", value=True)

# データ存在チェック
if not os.path.exists(integrated_path):
    st.warning(f"⚠️ `{integrated_path}` が見つかりません。先にStep 7のデータ統合を完了してください。")
    st.stop()
if not os.path.exists(forex_path):
    st.warning(f"⚠️ `{forex_path}` が見つかりません。パスを確認してください。")
    st.stop()

# --- データロード & 前処理 ---
df_integ = pd.read_csv(integrated_path)
df_fin = load_and_filter_forex_data(forex_path, start_time_str)

# 絞り込み
if governor_only and "is_governor" in df_integ.columns:
    df_integ = df_integ[df_integ["is_governor"] == True].copy()

# 時間アライメント (秒 -> datetime -> 1分足リサンプル)
conference_start_time = pd.to_datetime(start_time_str)
df_integ['datetime'] = conference_start_time + pd.to_timedelta(df_integ['start'], unit='s')
df_integ_time = df_integ.set_index('datetime')
df_integ_1min = df_integ_time.resample('1min').mean(numeric_only=True).reset_index()

# 為替と感情のマージ
available_vars = [v for v in ['text_score', 'face_emotion_score', 'audio_emotion_score', 'face_arousal_score', 'audio_arousal_score'] if v in df_integ_1min.columns]
df_merged = pd.merge(df_fin, df_integ_1min, on='datetime', how='inner')
df_merged = df_merged.dropna(subset=['return'] + available_vars)

# --- OLS回帰計算 ---
Y = df_merged['return']
X = df_merged[available_vars]
X_with_const = sm.add_constant(X)
regression_model = sm.OLS(Y, X_with_const)
regression_results = regression_model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})

# VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

# --- メイン UI ---
st.title("📊 日銀総裁会見 マルチモーダル感情分析ダッシュボード")
st.markdown("学術的な Valence（感情価）と Arousal（緊張度）モデルを使用した、意思決定プロセスと為替市場のリアルタイムアライメントシステム")

# 1. 感情 & 為替チャート（二重軸）
st.subheader("🕒 感情スコアと為替価格の推移 (時系列)")

fig = make_subplots(specs=[[{"secondary_y": True}]])
# 感情価 (Valence)
if 'text_score' in df_merged.columns:
    fig.add_trace(go.Scatter(x=df_merged['datetime'], y=df_merged['text_score'], name="Text Valence", line=dict(color="#00CC96", width=2)), secondary_y=False)
if 'face_emotion_score' in df_merged.columns:
    fig.add_trace(go.Scatter(x=df_merged['datetime'], y=df_merged['face_emotion_score'], name="Face Valence (Smile)", line=dict(color="#EF553B", width=2)), secondary_y=False)
if 'face_arousal_score' in df_merged.columns:
    fig.add_trace(go.Scatter(x=df_merged['datetime'], y=df_merged['face_arousal_score'], name="Face Arousal (Tension)", line=dict(color="#AB63FA", width=2, dash='dash')), secondary_y=False)
if 'audio_emotion_score' in df_merged.columns:
    fig.add_trace(go.Scatter(x=df_merged['datetime'], y=df_merged['audio_emotion_score'], name="Audio Valence", line=dict(color="#636EFA", width=2)), secondary_y=False)

# 為替 Close価格
fig.add_trace(go.Scatter(x=df_merged['datetime'], y=df_merged['close'], name="USD/JPY Close", line=dict(color="#FFD700", width=3)), secondary_y=True)

fig.update_layout(template="plotly_dark", hovermode="x unified", height=450, margin=dict(l=0, r=0, t=30, b=0))
fig.update_yaxes(title_text="感情スコア (Valence / Arousal)", secondary_y=False)
fig.update_yaxes(title_text="為替レート (USD/JPY)", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# 2. 動画プレイヤー & 乖離点 (Discrepancy)
st.markdown("---")
st.subheader("⚠️ 感情乖離点の分析 & 動画シーク連動 (Discrepancy Video Player)")
st.markdown("テキスト感情と非言語感情（表情・音声）の絶対差が最大値を示すセグメントを自動検知しています。カードをクリックすると動画がそのシーンにジャンプします。")

# 乖離トップ5
df_integ_copy = df_integ.copy()
df_integ_copy['discrepancy_score'] = df_integ_copy['discrepancy_score'].fillna(0.0)
top_disc = df_integ_copy.sort_values('discrepancy_score', ascending=False).head(5)

# 静的フォルダへの動画のコピー・シンボリックリンク確認
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
video_basename = os.path.basename(video_path)
target_static = static_dir / video_basename
if not target_static.exists() and os.path.exists(video_path):
    try:
        import shutil
        shutil.copy(video_path, target_static)
    except Exception as e:
        st.error(f"動画をstaticフォルダにロードできませんでした: {e}")

video_url = f"/app/static/{video_basename}"

# 乖離データリストの作成
discrepancy_list = []
for idx, row in top_disc.iterrows():
    discrepancy_list.append({
        "timestamp": float(row["start"]),
        "time_str": str(pd.Timedelta(seconds=int(row["start"]))),
        "score": round(float(row["discrepancy_score"]), 2),
        "text": str(row["text"]),
        "type": "非言語情報の感情乖離 (Text vs Nonverbal)"
    })

data_json = json.dumps(discrepancy_list)

custom_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #fff; background-color: #0e1117; margin: 0; padding: 0; }}
    .container {{ display: flex; gap: 20px; }}
    .video-section {{ flex: 2; }}
    .list-section {{ flex: 1; height: 380px; overflow-y: auto; background-color: #1e2130; border-radius: 8px; padding: 15px; border: 1px solid #3e4451; }}
    video {{ width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); outline: none; }}
    .discrepancy-card {{ background-color: #2d1a1a; border-left: 4px solid #ff4b4b; padding: 10px; margin-bottom: 10px; border-radius: 4px; cursor: pointer; transition: 0.2s; }}
    .discrepancy-card:hover {{ background-color: #3d2222; transform: translateX(5px); }}
    .time-badge {{ background-color: #ff4b4b; padding: 3px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
    .score {{ float: right; color: #ffdd57; font-weight: bold; }}
</style>
</head>
<body>
<div class="container">
    <div class="video-section">
        <video id="conference_video" controls>
            <source src="{video_url}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <div class="list-section" id="list_container">
        <h3 style="margin-top: 0; border-bottom: 1px solid #3e4451; padding-bottom: 10px;">⚠️ 乖離シーン一覧</h3>
    </div>
</div>
<script>
    const discrepancyData = {data_json};
    const video = document.getElementById("conference_video");
    const listContainer = document.getElementById("list_container");

    function jumpToTime(seconds) {{
        video.currentTime = seconds;
        video.play();
    }}

    discrepancyData.forEach((item) => {{
        const card = document.createElement("div");
        card.className = "discrepancy-card";
        card.onclick = () => jumpToTime(item.timestamp);
        card.innerHTML = `
            <div>
                <span class="time-badge">🕒 ${{item.time_str}}</span>
                <span class="score">乖離度: ${{item.score}}</span>
            </div>
            <div style="margin-top: 8px; font-size: 14px; font-weight: bold; color: #ff9999;">${{item.type}}</div>
            <div style="margin-top: 5px; font-size: 13px; color: #ccc; font-style: italic;">"${{item.text}}"</div>
        `;
        listContainer.appendChild(card);
    }});
</script>
</body>
</html>
"""
components.html(custom_html, height=400, scrolling=False)

# 3. OLS回帰分析結果
st.markdown("---")
st.subheader("📈 回帰分析結果サマリー (OLS Regression with Newey-West HAC)")

col1, col2 = st.columns([1, 1])

# 回帰サマリーテーブル
res_df = pd.DataFrame({
    "係数 (Coef)": regression_results.params,
    "P値 (P-value)": regression_results.pvalues,
    "標準誤差 (Std.Err)": regression_results.bse,
    "VIF": vif_data.set_index('Variable')['VIF']
})

def get_stars(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.1: return "*"
    return ""

res_df["有意性"] = res_df["P値 (P-value)"].apply(get_stars)

with col1:
    st.markdown(f"**自由度調整済み決定係数 (Adj. R-squared)**: `{regression_results.rsquared_adj:.4f}`")
    st.markdown(f"**F値のP値 (Prob (F-statistic))**: `{regression_results.f_pvalue:.6f}`")
    st.dataframe(res_df.style.format({
        "係数 (Coef)": "{:.4f}",
        "P値 (P-value)": "{:.4f}",
        "標準誤差 (Std.Err)": "{:.4f}",
        "VIF": "{:.2f}"
    }), use_container_width=True)
    st.caption("有意水準: *** p<0.01, ** p<0.05, * p<0.1 (Newey-West HAC標準誤差)")

# フォレストプロット
with col2:
    conf_int = regression_results.conf_int()
    forest_df = pd.DataFrame({
        "Variable": regression_results.params.index,
        "Coef": regression_results.params,
        "Lower": conf_int[0],
        "Upper": conf_int[1]
    }).reset_index(drop=True)
    
    # const はプロットから除外
    forest_df = forest_df[forest_df["Variable"] != "const"]
    
    fig_forest = go.Figure()
    fig_forest.add_trace(go.Scatter(
        x=forest_df["Coef"],
        y=forest_df["Variable"],
        mode="markers",
        error_x=dict(
            type="data",
            symmetric=False,
            array=forest_df["Upper"] - forest_df["Coef"],
            arrayminus=forest_df["Coef"] - forest_df["Lower"],
            color="#ff4b4b"
        ),
        marker=dict(size=12, color="#ff4b4b"),
        name="Coefficient"
    ))
    fig_forest.add_shape(type="line", x0=0, y0=-0.5, x1=0, y1=len(forest_df)-0.5, line=dict(color="white", width=1, dash="dash"))
    fig_forest.update_layout(
        title="各感情特徴量の係数と95%信頼区間 (Forest Plot)",
        template="plotly_dark",
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_forest, use_container_width=True)
