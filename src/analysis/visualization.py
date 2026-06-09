import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re
import numpy as np

# --- ページ設定とデザイン (UI v3.0 Research Edition) ---
st.set_page_config(
    page_title="日銀総裁会見 マルチモーダル分析 Ver 3.0",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ライトテーマ（白背景・黒文字）の強制とスタイリング
st.markdown("""
    <style>
    /* 全体背景を白に */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    /* ヘッダー・サブヘッダーの配色 */
    h1, h2, h3, p {
        color: #1a1a1a !important;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    /* サイドバー */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
    /* 乖離ランキングカード */
    .discrepancy-card {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #ffffff;
        border: 1px solid #ff4b4b;
        border-left: 8px solid #ff4b4b;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .discrepancy-card b {
        color: #d32f2f;
    }
    /* チャットUIの調整 */
    .stChatMessage {
        background-color: #f1f3f5 !important;
        border-radius: 12px !important;
        padding: 10px !important;
        margin-bottom: 8px !important;
    }
    /* ボタンのクリーン化 */
    .stButton>button {
        border-radius: 20px;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- データ読み込みと前処理 ---
# Conference metadata
CONFERENCE_START_JST = "2023-06-16 15:30:00"  # Update this to actual conference start time
CONFERENCE_DATE = "2023年6月16日"

SPEAKER_MAP = {
    "SPEAKER_15": "黒田総裁 (Kuroda, Governor)",
    "SPEAKER_12": "司会/記者 (Host/Press)",
    "SPEAKER_00": "黒田総裁 (Kuroda, Governor)"  # Fallback
}

SPEAKER_COLORS = {
    "黒田総裁 (Kuroda, Governor)": "#1f77b4",  # Blue for Governor
    "司会/記者 (Host/Press)": "#ff7f0e"  # Orange for Press
}

@st.cache_data
def load_data(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    # Map SOTA column names to Internal Dashboard names
    rename_map = {
        'sentiment_score': '感情スコア',
        'discrepancy_score_sota': '乖離度'
    }
    df = df.rename(columns=rename_map)
    
    # Speaker Mapping
    df['speaker_name'] = df['speaker'].map(lambda x: SPEAKER_MAP.get(x, x))
    
    # Calculate "Positive Expression" proxy if not present
    if '表情の肯定的表情' not in df.columns and 'mean_AU12' in df.columns and 'mean_AU04' in df.columns:
        df['表情の肯定的表情'] = df['mean_AU12'] - df['mean_AU04']
    
    # Add elapsed time formatting (MM:SS)
    df['elapsed_mmss'] = df['start'].apply(lambda s: f"{int(s//60):02d}:{int(s%60):02d}")
    
    # Add actual timestamp (HH:MM:SS)
    from datetime import datetime, timedelta
    conference_start = datetime.strptime(CONFERENCE_START_JST, "%Y-%m-%d %H:%M:%S")
    df['actual_time'] = df['start'].apply(lambda s: (conference_start + timedelta(seconds=s)).strftime("%H:%M:%S"))
    
    # Add moving average for AU smoothing (30-second window, ~30 frames at 1fps)
    df['AU04_ma'] = df['mean_AU04'].rolling(window=30, min_periods=1, center=True).mean()
    df['AU12_ma'] = df['mean_AU12'].rolling(window=30, min_periods=1, center=True).mean()
    
    return df

@st.cache_data
def load_market_data(output_path, data_dir_path):
    # Prioritize real data in data/ if available, else check output/
    paths = [Path(data_dir_path) / "usd_jpy_historical.csv", Path(output_path)]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                return df
    return None

def align_market_to_video(df, market_df, video_start_jst="2023-06-16 15:30:00"):
    if market_df is None:
        return df
    from datetime import timedelta
    start_dt = pd.to_datetime(video_start_jst).tz_localize(None)
    market_df_clean = market_df.copy()
    market_df_clean['Datetime'] = market_df_clean['Datetime'].dt.tz_localize(None)
    
    prices = []
    for _, row in df.iterrows():
        target_time = start_dt + timedelta(seconds=row['start'])
        # Find closest price
        idx = (market_df_clean['Datetime'] - target_time).abs().idxmin()
        prices.append(market_df_clean.loc[idx, 'Close'])
    
    df['USD/JPY'] = prices
    return df

# データロード
data_file = "output/integrated_results.csv"
market_file = "output/market_data.csv"
market_dir = "data"
df = load_data(data_file)
market_df = load_market_data(market_file, market_dir)

if df is not None and market_df is not None:
     df = align_market_to_video(df, market_df)

video_path = "data/boj_conference.mp4"

if df is None:
    st.error("分析データが見つかりません。パイプラインを実行して `integrated_results.csv` を生成してください。")
    st.stop()

# セッション状態での再生位置管理
if 'current_time' not in st.session_state:
    st.session_state.current_time = 0.0

# --- メインレイアウト ---
st.title("📊 日銀総裁会見 マルチモーダル検証ツール (Ver 3.0)")

# Metadata Banner
if market_df is not None:
    fx_start = market_df['Datetime'].min().strftime("%Y年%m月%d日 %H:%M")
    fx_end = market_df['Datetime'].max().strftime("%H:%M")
    fx_range = f"{fx_start} ~ {fx_end}"
else:
    fx_range = "Not available"

video_duration_min = int(df['end'].max() / 60)
st.markdown(f"""
### 📅 データ概要
- **会見日時:** {CONFERENCE_DATE} {CONFERENCE_START_JST.split()[1]}開始 (総時間: {video_duration_min}分)
- **為替データ範囲:** {fx_range}
- **分析モード:** SOTA Multimodal (Fin-BERT + Py-Feat + Whisper)
""")
st.markdown("---")

col_main, col_chat = st.columns([2.5, 1])

with col_main:
    # --- 1. Dual-Axis Chart: AU vs USD/JPY ---
    st.subheader("⏱️ マルチモーダル同期タイムライン (AU + 為替)")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # AU04 (Left Y-axis) - Raw + Moving Average
    fig.add_trace(
        go.Scatter(
            x=df['start'], 
            y=df['mean_AU04'], 
            name="AU04 (眉間の寄せ)",
            line=dict(color='#dc3545', width=1, dash='dot'),
            opacity=0.3
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['start'], 
            y=df['AU04_ma'], 
            name="AU04 (移動平均)",
            line=dict(color='#dc3545', width=3),
            hovertemplate="AU04: %{y:.3f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # AU12 (Left Y-axis) - Raw + Moving Average
    fig.add_trace(
        go.Scatter(
            x=df['start'], 
            y=df['mean_AU12'], 
            name="AU12 (口角上げ)",
            line=dict(color='#28a745', width=1, dash='dot'),
            opacity=0.3
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['start'], 
            y=df['AU12_ma'], 
            name="AU12 (移動平均)",
            line=dict(color='#28a745', width=3),
            hovertemplate="AU12: %{y:.3f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # USD/JPY (Right Y-axis)
    if 'USD/JPY' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['start'], 
                y=df['USD/JPY'], 
                name="USD/JPY",
                line=dict(color='#ff7f0e', width=3),
                hovertemplate="USD/JPY: %{y:.2f}<extra></extra>"
            ),
            secondary_y=True
        )
    
    # Set axes titles
    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="<b>AU Intensity</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>USD/JPY</b>", secondary_y=True)
    
    # Update layout
    fig.update_layout(
        height=500,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 2. Zoom Subgraph: Top 3 Discrepancy Moments ---
    st.subheader("🔍 注目: 最大乖離区間 (Top 3)")
    
    # Find top 3 discrepancy moments
    top_disc = df.nlargest(3, '乖離度').sort_values('start')
    
    if not top_disc.empty:
        # Create zoom view around each moment
        zoom_range = 60  # +/- 60 seconds
        for idx, disc_row in top_disc.iterrows():
            center_time = disc_row['start']
            zoom_df = df[(df['start'] >= center_time - zoom_range) & (df['start'] <= center_time + zoom_range)]
            
            if not zoom_df.empty:
                zoom_fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # AU04 in zoom
                zoom_fig.add_trace(
                    go.Scatter(x=zoom_df['start'], y=zoom_df['AU04_ma'], 
                               name="AU04", line=dict(color='#dc3545', width=2)),
                    secondary_y=False
                )
                
                # USD/JPY in zoom
                if 'USD/JPY' in zoom_df.columns:
                    zoom_fig.add_trace(
                        go.Scatter(x=zoom_df['start'], y=zoom_df['USD/JPY'], 
                                   name="USD/JPY", line=dict(color='#ff7f0e', width=2)),
                        secondary_y=True
                    )
                
                # Mark the exact discrepancy moment
                zoom_fig.add_vline(x=center_time, line_dash="dash", line_color="red", 
                                   annotation_text=f"乖離度: {disc_row['乖離度']:.2f}")
                
                zoom_fig.update_layout(
                    height=250,
                    title=f"Time: {disc_row['elapsed_mmss']} | {disc_row['text'][:50]}...",
                    template="plotly_white",
                    showlegend=False
                )
                
                zoom_fig.update_xaxes(title_text="Time (s)")
                zoom_fig.update_yaxes(title_text="AU", secondary_y=False)
                zoom_fig.update_yaxes(title_text="USD/JPY", secondary_y=True)
                
                st.plotly_chart(zoom_fig, use_container_width=True)

    # ビデオプレイヤー (現在のシーク位置に連動)
    play_time = st.slider("再生位置の調整 (秒)", 0.0, float(df['end'].max()), float(st.session_state.current_time), step=0.1)
    st.video(video_path, start_time=int(play_time))

with col_chat:
    # --- 2. マルチモーダル・エビデンス・ログ (横並び表示) ---
    st.subheader("📋 マルチモーダル・検証ログ")
    st.markdown("話者・テキスト・表情（AU）を横並びで表示")
    
    # フィルタリングオプション
    only_governor = st.checkbox("黒田総裁のみ表示", value=True)
    
    log_df = df.copy()
    if only_governor:
        log_df = log_df[log_df['speaker_name'].str.contains("黒田")]

    # テーブルヘッダー
    head_col1, head_col2, head_col3, head_col4 = st.columns([0.6, 0.6, 2.0, 1.2])
    head_col1.write("**経過時間**")
    head_col2.write("**実時刻**")
    head_col3.write("**テキスト**")
    head_col4.write("**AU/感情**")
    st.markdown("---")

    chat_container = st.container(height=800)
    with chat_container:
        for idx, row in log_df.iterrows():
            c1, c2, c3, c4 = st.columns([0.6, 0.6, 2.0, 1.2])
            
            # Determine speaker color
            speaker_color = SPEAKER_COLORS.get(row['speaker_name'], "#6c757d")
            
            # Column 1: Elapsed Time (MM:SS)
            c1.markdown(f"<span style='color:{speaker_color}; font-weight:bold;'>{row['elapsed_mmss']}</span>", unsafe_allow_html=True)
            
            # Column 2: Actual Time (HH:MM:SS)
            c2.caption(row['actual_time'])
            
            # Column 3: Text
            c3.write(row['text'])
            if st.button("再生", key=f"play_{idx}"):
                st.session_state.current_time = row['start']
                st.rerun()
            
            # Column 4: Metrics
            sentiment_color = "#28a745" if row['感情スコア'] > 0 else "#dc3545" if row['感情スコア'] < 0 else "#6c757d"
            c4.markdown(f"<span style='color:{sentiment_color}'>感情: {row['感情スコア']:.2f}</span>", unsafe_allow_html=True)
            c4.caption(f"AU04: {row.get('mean_AU04', 0):.2f}")
            c4.caption(f"AU12: {row.get('mean_AU12', 0):.2f}")
            
            st.markdown("---")

# --- 下部セクション: 高度な分析 ---
st.markdown("---")
col_rank, col_stats = st.columns([1, 1])

with col_rank:
    # --- 3. 感情の乖離（Discrepancy）ランキング ---
    st.header("⚠️ 感情の乖離ランキング")
    st.markdown("「言葉」と「表情」が最も矛盾している場面（乖離度 > 0.5）を表示します。")
    
    high_disc = df[df['乖離度'] > 0.5].sort_values('乖離度', ascending=False).head(10)
    
    if not high_disc.empty:
        for idx, row in high_disc.iterrows():
            reasoning = row.get('discrepancy_reasoning', 'N/A')
            st.markdown(f"""
            <div class="discrepancy-card">
                <b>乖離度 {row['乖離度']:.2f}</b> (時間: {row['start']:.1f}s)<br/>
                <b>判定根拠:</b> {reasoning}<br/>
                内容: 「{row['text']}」<br/>
                <small>テキスト感情: {row['感情スコア']:.2f} / 表情肯定的(AU12-AU04): {row['表情の肯定的表情']:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
            if st.button("詳細を動画で確認", key=f"rank_{idx}"):
                st.session_state.current_time = row['start']
                st.rerun()
    else:
        st.info("顕著な感情の乖離は見つかりませんでした。")

with col_stats:
    # --- 4. キーワード別 微表情 統計分析 ---
    st.header("🔍 統計的キーワード探索")
    search_keyword = st.text_input("分析したいキーワードを入力 (例: 利上げ, 不透明, 物価)", "")
    
    if search_keyword:
        matches = df[df['text'].str.contains(search_keyword, na=False)]
        if not matches.empty:
            st.success(f"「{search_keyword}」を含む発言を {len(matches)} 件検出しました。")
            
            # 平均的な表情強度の比較
            avg_au_match = matches['mean_AU04'].mean()
            avg_au_global = df['mean_AU04'].mean()
            
            st.metric("キーワード出現時の AU04 (眉間の寄せ) 平均", f"{avg_au_match:.3f}", 
                      delta=f"{avg_au_match - avg_au_global:.3f}", 
                      delta_color="inverse")
            st.caption("※デルタは会見全体平均との差分を表示しています。プラスは緊張が高いことを示唆。")
            
            # ヒストグラム
            fig_hist = px.histogram(matches, x="mean_AU04", nbins=10, 
                                   title=f"「{search_keyword}」発言時の微表情分布",
                                   template="plotly_white", color_discrete_sequence=['#ff4b4b'])
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("該当するキーワードは見つかりませんでした。")

# フッター
st.markdown("---")
st.caption("日銀マルチモーダル分析 Ver 3.0 | 学術研究用プロトタイプ")
