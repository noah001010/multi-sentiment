import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import sys

sys.path.append('src/features')
from end_to_end_sentiment_pipeline import build_multimodal_features_and_analyze

# --- Page Configuration ---
st.set_page_config(page_title="Multimodal Sentiment Analysis Dashboard", layout="wide", page_icon="📈")

st.markdown("""
<style>
    body { font-family: 'Inter', sans-serif; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #636efa; color: white; border: none; }
    .stButton>button:hover { background-color: #525ce0; color: white; }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #636efa;
    }
</style>
""", unsafe_allow_html=True)

# --- Pipeline Execution Function ---
@st.cache_resource(show_spinner=False)
def run_pipeline(video_path, finance_path, start_time):
    return build_multimodal_features_and_analyze(
        video_path=video_path,
        financial_csv=finance_path,
        output_emotions_csv="data/multimodal_emotions.csv",
        conference_start_time_str=start_time
    )

def generate_discrepancy_list(df_merged, start_time_str):
    """
    パイプライン出力から乖離点トップ3を算出する
    """
    df = df_merged.copy()
    # 連続値としての差分の絶対値を乖離スコアとする
    df['discrepancy_score'] = abs(df['text_sentiment'] - df['facial_negative_emotion'])
    top_n = df.sort_values('discrepancy_score', ascending=False).head(5)
    
    start_time = pd.to_datetime(start_time_str)
    
    discrepancy_data = []
    for _, row in top_n.iterrows():
        seconds = (row['datetime'] - start_time).total_seconds()
        discrepancy_data.append({
            "timestamp": seconds,
            "time_str": str(pd.Timedelta(seconds=int(seconds))),
            "type": "Text vs Face Discrepancy",
            "score": round(row['discrepancy_score'], 2),
            "text": f"Text Score: {row['text_sentiment']:.2f} | Face Score: {row['facial_negative_emotion']:.2f}"
        })
    return sorted(discrepancy_data, key=lambda x: x['timestamp'])

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    video_path = st.text_input("動画のパス (.mp4)", value="data/boj_5min.mp4", placeholder="例: data/boj_5min.mp4 (相対・絶対パスどちらも可)")
    finance_path = st.text_input("HistData CSVのパス", value="data/DAT_ASCII_USDJPY_M1_2023.csv", placeholder="例: data/DAT_ASCII_USDJPY_M1_2023.csv")
    start_time = st.text_input("会見開始時刻", value="2023-06-16 15:30:00", placeholder="例: 2023-06-16 15:30:00")
    
    run_btn = st.button("🚀 分析を実行する")

if run_btn:
    st.session_state['run_analysis'] = True

if st.session_state.get('run_analysis', False):
    
    # 実行前のエラーハンドリング（パスの存在チェック）
    if not os.path.exists(video_path):
        st.error(f"⚠️ エラー: 指定された動画ファイルが見つかりません。\n入力されたパス: `{video_path}`\nプロジェクトのルートディレクトリからの相対パス、または絶対パスを指定してください。")
        st.stop()
        
    if not os.path.exists(finance_path):
        st.error(f"⚠️ エラー: 指定された為替CSVファイルが見つかりません。\n入力されたパス: `{finance_path}`\nプロジェクトのルートディレクトリからの相対パス、または絶対パスを指定してください。")
        st.stop()

    with st.status("マルチモーダル分析パイプラインを実行中...", expanded=True) as status:
        try:
            st.write("⏳ 1/4: バックグラウンドで音声と動画の分離、およびキャッシュの確認を行っています...")
            # 実際には一括実行だが、ユーザー体験向上のためにフェーズ表示
            st.write("⏳ 2/4: 各モダリティ (DeepFace, OpenSMILE, Whisper) の抽出とメモリ最適化処理...")
            st.write("⏳ 3/4: FinBERTによる感情スコアリング...")
            st.write("⏳ 4/4: データの結合とOLS回帰分析の実行...")
            output = run_pipeline(video_path, finance_path, start_time)
            status.update(label="✅ 分析が完了しました！", state="complete", expanded=False)
        except Exception as e:
            status.update(label="⚠️ エラーが発生しました", state="error", expanded=True)
            st.error(f"⚠️ 分析パイプラインの実行中にエラーが発生しました:\n\n{e}")
            st.stop()
            
    df_merged = output['df_merged']
    regression = output['regression']
    
    st.title("📊 マルチモーダル感情分析ダッシュボード")
    
    # 1. 感情分析の時系列グラフ (Plotly)
    st.subheader("🕒 感情スコアの推移")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_merged['datetime'], y=df_merged['text_sentiment'], mode='lines', name='Text Sentiment', line=dict(color='#00CC96', width=2)))
    fig.add_trace(go.Scatter(x=df_merged['datetime'], y=df_merged['facial_negative_emotion'], mode='lines', name='Facial (Negative)', line=dict(color='#EF553B', width=2)))
    fig.add_trace(go.Scatter(x=df_merged['datetime'], y=df_merged['voice_tone'], mode='lines', name='Voice Tone', line=dict(color='#636EFA', width=2)))
    fig.update_layout(template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. 動画プレイヤーと乖離点リスト
    st.markdown("---")
    st.subheader("⚠️ 感情乖離点の分析 (Discrepancy Points)")
    
    discrepancy_data = generate_discrepancy_list(df_merged, start_time)
    data_json = json.dumps(discrepancy_data)
    
    # Static フォルダへの動画コピー（安全策）
    if not os.path.exists("static"):
        os.makedirs("static")
    target_static_path = f"static/{os.path.basename(video_path)}"
    if not os.path.exists(target_static_path) and os.path.exists(video_path):
        import shutil
        shutil.copy(video_path, target_static_path)
        
    video_url = f"app/static/{os.path.basename(video_path)}"
    
    custom_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: 'Inter', sans-serif; color: #fff; background-color: transparent; margin: 0; padding: 0; }}
        .container {{ display: flex; gap: 20px; }}
        .video-section {{ flex: 2; }}
        .list-section {{ flex: 1; height: 400px; overflow-y: auto; background-color: #1e2130; border-radius: 8px; padding: 15px; border: 1px solid #3e4451; }}
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
            </video>
        </div>
        <div class="list-section" id="list_container">
            <h3 style="margin-top: 0;">乖離ポイント検出</h3>
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
                    <span class="score">Score: ${{item.score}}</span>
                </div>
                <div style="margin-top: 8px; font-size: 14px; font-weight: bold;">${{item.type}}</div>
                <div style="margin-top: 5px; font-size: 13px; color: #aaa; font-style: italic;">${{item.text}}</div>
            `;
            listContainer.appendChild(card);
        }});
    </script>
    </body>
    </html>
    """
    components.html(custom_html, height=450, scrolling=False)
    
    # 3. 回帰分析結果
    st.markdown("---")
    st.subheader("📈 回帰分析結果サマリー (OLS Regression)")
    
    col1, col2 = st.columns([1, 1])
    
    results = regression['results']
    
    # 結果テーブルの作成
    res_df = pd.DataFrame({
        "係数 (Coef)": results.params,
        "P値 (P-value)": results.pvalues,
        "標準誤差 (Std.Err)": results.bse,
        "VIF": regression['vif_data'].set_index('Variable')['VIF']
    })
    
    def get_stars(p):
        if p < 0.01: return "***"
        elif p < 0.05: return "**"
        elif p < 0.1: return "*"
        return ""
        
    res_df["有意性"] = res_df["P値 (P-value)"].apply(get_stars)
    
    with col1:
        st.markdown(f"**Adjusted R-squared**: `{results.rsquared_adj:.4f}`")
        st.dataframe(res_df.style.format({
            "係数 (Coef)": "{:.4f}",
            "P値 (P-value)": "{:.4f}",
            "標準誤差 (Std.Err)": "{:.4f}",
            "VIF": "{:.2f}"
        }), use_container_width=True)
        
    # フォレストプロット (グラフA)
    with col2:
        conf_int = results.conf_int()
        forest_df = pd.DataFrame({
            "Variable": results.params.index,
            "Coef": results.params,
            "Lower": conf_int[0],
            "Upper": conf_int[1]
        }).reset_index(drop=True)
        
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
            title="各モダリティの係数と95%信頼区間 (Forest Plot)",
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_forest, use_container_width=True)
        
    # Fitted values (グラフB)
    st.subheader("📊 予測リターン vs 実際のリターン")
    fitted = results.fittedvalues
    actual = regression['Y']
    datetime_idx = df_merged['datetime']
    
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(x=datetime_idx, y=actual, mode='lines+markers', name='Actual Return (1min)', opacity=0.6))
    fig_fit.add_trace(go.Scatter(x=datetime_idx, y=fitted, mode='lines', name='Predicted Return (Fitted)', line=dict(color='#ffdd57', width=3)))
    fig_fit.update_layout(template="plotly_dark", hovermode="x unified", height=400)
    st.plotly_chart(fig_fit, use_container_width=True)

else:
    st.info("👈 サイドバーから動画・為替データ・会見開始時刻を設定し、「🚀 分析を実行する」をクリックしてください。")
