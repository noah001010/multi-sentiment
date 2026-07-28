import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import streamlit.components.v1 as components

# --- ページ設定 ---
st.set_page_config(
    page_title="BOJ Multimodal Sentiment Dashboard", 
    layout="wide", 
    page_icon="🤖"
)

# ガラスモーフィズムとStripe/OpenAI調のプレミアムスタイルの適用
st.markdown("""
<style>
    /* 全体背景 */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    /* サイドバーの背景とテキスト色（高コントラスト化） */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(99, 91, 255, 0.3) !important;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    
    /* 有名テック企業のような美しいグラデーション見出し */
    h1, h2, h3 {
        background: linear-gradient(135deg, #635bff 0%, #a388ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        text-shadow: 0 4px 12px rgba(99, 91, 255, 0.15);
    }
    
    /* ガラスモーフカード */
    .metric-card {
        background: rgba(18, 24, 38, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 91, 255, 0.15);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
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

# --- システム固定設定 ---
video_path = "data/boj_conference.mp4"
integrated_path = "output/integrated_results.csv"
forex_path = "data/DAT_ASCII_USDJPY_M1_2023.csv"
start_time_str = "2023-06-16 15:30:00"
governor_only = True

# --- サイドバー ---
with st.sidebar:
    st.header("⚡ System Status")
    st.success("🤖 Pipeline Status: ONLINE")
    
    st.markdown("### 📁 Dataset Configurations")
    st.code(
        f"Video Path: {video_path}\n"
        f"Integ CSV : {integrated_path}\n"
        f"Forex CSV : {forex_path}\n"
        f"Start Time: {start_time_str}\n"
        f"Gov Only  : {governor_only}"
    )
    st.info("💡 デモ実演用に動画パスおよび開始時刻設定は固定されています。")

# データ存在チェック
if not os.path.exists(integrated_path):
    st.warning(f"⚠️ `{integrated_path}` が見つかりません。先にデータ統合を完了してください。")
    st.stop()
if not os.path.exists(forex_path):
    st.warning(f"⚠️ `{forex_path}` が見つかりません。パスを確認してください。")
    st.stop()

# --- 規格化ヘルパー ---
def min_max_normalize(series: pd.Series) -> pd.Series:
    s_min, s_max = series.min(), series.max()
    if s_min == s_max:
        return pd.Series(0.0, index=series.index)
    return -1.0 + 2.0 * (series - s_min) / (s_max - s_min)

# --- データの読み込み ---
df_integ = pd.read_csv(integrated_path)
df_fin = load_and_filter_forex_data(forex_path, start_time_str)

# 静的フォルダへの動画のコピー確認
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

# 時間アライメント処理 (回帰分析用)
conference_start_time = pd.to_datetime(start_time_str)
df_integ['datetime'] = conference_start_time + pd.to_timedelta(df_integ['start'], unit='s')
df_integ_time = df_integ.set_index('datetime')
df_integ_1min = df_integ_time.resample('1min').mean(numeric_only=True).reset_index()

available_vars = [v for v in ['text_score', 'face_emotion_score', 'audio_emotion_score', 'face_arousal_score', 'audio_arousal_score'] if v in df_integ_1min.columns]
df_merged = pd.merge(df_fin, df_integ_1min, on='datetime', how='inner')
df_merged = df_merged.dropna(subset=['return'] + available_vars)

# 回帰モデル OLS (Newey-West)
Y = df_merged['return']
X = df_merged[available_vars]
X_with_const = sm.add_constant(X)
regression_model = sm.OLS(Y, X_with_const)
regression_results = regression_model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})

# VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

# JS/HTML コンポーネント用のデータ準備
# 1. 1分足の感情&為替データ (可視化用に感情・緊張データを [-1, 1] に規格化)
df_plot = df_merged.copy()
for col in ['text_score', 'face_emotion_score', 'face_arousal_score', 'audio_emotion_score', 'audio_arousal_score']:
    if col in df_plot.columns:
        df_plot[col] = min_max_normalize(df_plot[col])

chart_data_list = []
for _, row in df_plot.iterrows():
    minutes = (row["datetime"] - conference_start_time).total_seconds() / 60.0
    chart_data_list.append({
        "m": round(minutes, 2),
        "close": round(float(row["close"]), 4),
        "text": round(float(row["text_score"]), 4) if 'text_score' in row else 0.0,
        "face_val": round(float(row["face_emotion_score"]), 4) if 'face_emotion_score' in row else 0.0,
        "face_aro": round(float(row["face_arousal_score"]), 4) if 'face_arousal_score' in row else 0.0,
        "audio_val": round(float(row["audio_emotion_score"]), 4) if 'audio_emotion_score' in row else 0.0,
        "audio_aro": round(float(row["audio_arousal_score"]), 4) if 'audio_arousal_score' in row else 0.0,
    })

# 2. 全セグメントの文字起こしデータ
transcript_list = []
for idx, row in df_integ.iterrows():
    transcript_list.append({
        "id": int(idx),
        "start": float(row["start"]),
        "end": float(row["end"]),
        "text": str(row["text"]),
        "speaker": str(row.get("speaker", "UNKNOWN")),
        "is_gov": bool(row.get("is_governor", False))
    })

chart_json = json.dumps(chart_data_list)
transcript_json = json.dumps(transcript_list)

# --- メインダッシュボード ---
st.title("🤖 BOJ Governor Multimodal Real-Time Aligner")
st.markdown("オープンキャンパス実演用：日銀総裁の発話・表情・声のトーンと、為替市場のリアルタイム同期可視化デモシステム")

# リアルタイム同期用 HTML/CSS/JS コンポーネント
custom_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<!-- Tailwind CSS CDN -->
<script src="https://cdn.tailwindcss.com"></script>
<!-- Chart.js CDN -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
    body {{
        font-family: 'Outfit', sans-serif;
        background-color: #0b0f19;
        color: #f8fafc;
    }}
    .glow-border {{
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(99, 91, 255, 0.15);
    }}
    .active-glow {{
        background: rgba(99, 91, 255, 0.15) !important;
        border: 2px solid #635bff !important;
        box-shadow: 0 0 20px rgba(99, 91, 255, 0.4);
    }}
    /* スクロールバーのカスタマイズ */
    ::-webkit-scrollbar {{
        width: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: #101625;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #4b5563; /* grey thumb */
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #635bff;
    }}
</style>
</head>
<body class="p-3">

<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
    <!-- 左・中カラム: 動画プレイヤー & 同期チャート -->
    <div class="lg:col-span-2 space-y-4">
        <!-- 動画プレイヤー -->
        <div class="bg-[#101625] p-3 rounded-xl glow-border">
            <video id="boj_video" class="w-full rounded-lg shadow-2xl" controls>
                <source src="{video_url}" type="video/mp4">
            </video>
        </div>
        
        <!-- 同期ラインチャート -->
        <div class="bg-[#101625] p-4 rounded-xl glow-border">
            <h3 class="text-[#8a7eff] font-bold text-lg mb-2 flex justify-between items-center">
                📊 マルチモーダル時系列アライメント
                <span id="current_time_display" class="text-sm bg-[#0b0f19] px-3 py-1 rounded text-white border border-gray-700">Elapsed: 00:00</span>
            </h3>
            <div style="position: relative; height: 260px; width: 100%;">
                <canvas id="sync_chart"></canvas>
            </div>
        </div>
    </div>
    
    <!-- 右カラム: リアルタイムスクロール文字起こし -->
    <div class="bg-[#101625] p-4 rounded-xl glow-border flex flex-col h-[755px]">
        <h3 class="text-[#8a7eff] font-bold text-lg mb-3 pb-2 border-b border-gray-700">💬 発話セグメント (Transcript)</h3>
        <p class="text-xs text-gray-400 mb-3">カードをクリックすると、動画の該当発話シーンへ直接ジャンプします。</p>
        <div id="transcript_container" class="flex-1 overflow-y-auto space-y-3 pr-2">
            <!-- JSで動的生成 -->
        </div>
    </div>
</div>

<script>
    const chartData = {chart_json};
    const transcriptData = {transcript_json};

    const video = document.getElementById("boj_video");
    const transcriptContainer = document.getElementById("transcript_container");
    const timeDisplay = document.getElementById("current_time_display");

    // 1. チャートの初期化 (Chart.js)
    const labels = chartData.map(d => d.m);
    const textScores = chartData.map(d => d.text);
    const faceValScores = chartData.map(d => d.face_val);
    const faceAroScores = chartData.map(d => d.face_aro);
    const audioValScores = chartData.map(d => d.audio_val);
    const fxPrices = chartData.map(d => d.close);

    // カスタム垂直線描画プラグイン
    const verticalLinePlugin = {{
        id: 'verticalLine',
        afterDraw: (chart) => {{
            if (chart.config.options.plugins.verticalLine && chart.config.options.plugins.verticalLine.xValue !== undefined) {{
                const xValue = chart.config.options.plugins.verticalLine.xValue;
                const xAxis = chart.scales.x;
                const yAxis = chart.scales.y;
                const xPixel = xAxis.getPixelForValue(xValue);
                
                const ctx = chart.ctx;
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(xPixel, yAxis.top);
                ctx.lineTo(xPixel, yAxis.bottom);
                ctx.lineWidth = 3;
                ctx.strokeStyle = '#635bff'; // ストライプインディゴ
                ctx.shadowColor = '#635bff';
                ctx.shadowBlur = 8;
                ctx.stroke();
                ctx.restore();
            }}
        }}
    }};
    Chart.register(verticalLinePlugin);

    const ctx = document.getElementById('sync_chart').getContext('2d');
    const chart = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: labels,
            datasets: [
                {{
                    label: 'Text Valence (言語感情)',
                    data: textScores,
                    borderColor: '#2ecc71',
                    borderWidth: 2,
                    pointRadius: 0,
                    yAxisID: 'y_emotion'
                }},
                {{
                    label: 'Face Valence (表情ポジネガ)',
                    data: faceValScores,
                    borderColor: '#e74c3c',
                    borderWidth: 2,
                    pointRadius: 0,
                    yAxisID: 'y_emotion'
                }},
                {{
                    label: 'Face Arousal (表情緊張度)',
                    data: faceAroScores,
                    borderColor: '#9b59b6',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    yAxisID: 'y_emotion'
                }},
                {{
                    label: 'Audio Valence (音声感情)',
                    data: audioValScores,
                    borderColor: '#3498db',
                    borderWidth: 2,
                    pointRadius: 0,
                    yAxisID: 'y_emotion'
                }},
                {{
                    label: 'USD/JPY 為替 Close',
                    data: fxPrices,
                    borderColor: '#f1c40f',
                    borderWidth: 3,
                    pointRadius: 0,
                    yAxisID: 'y_forex'
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                x: {{
                    title: {{
                        display: true,
                        text: '会見開始からの経過時間 (分足)',
                        color: '#c5c6c7'
                    }},
                    grid: {{ color: '#1f2833' }},
                    ticks: {{ color: '#c5c6c7' }}
                }},
                y_emotion: {{
                    position: 'left',
                    title: {{ display: true, text: '感情・緊張スコア (規格化 [-1, 1])', color: '#c5c6c7' }},
                    grid: {{ color: '#1f2833' }},
                    ticks: {{ color: '#c5c6c7' }},
                    min: -1.2,
                    max: 1.2
                }},
                y_forex: {{
                    position: 'right',
                    title: {{ display: true, text: '為替価格 (USD/JPY)', color: '#c5c6c7' }},
                    grid: {{ drawOnChartArea: false }},
                    ticks: {{ color: '#c5c6c7' }}
                }}
            }},
            plugins: {{
                legend: {{
                    position: 'top',
                    labels: {{ color: '#c5c6c7', boxWidth: 12, font: {{ size: 10 }} }}
                }},
                verticalLine: {{
                    xValue: 0.0
                }}
            }}
        }}
    }});

    // 2. 文字起こしリストの動的生成
    transcriptData.forEach((item, index) => {{
        const card = document.createElement("div");
        card.id = `card-${{index}}`;
        card.className = `p-3 rounded-lg cursor-pointer transition duration-300 border ${{
            item.is_gov 
                ? "bg-[#0b0c10] border-cyan-900 hover:bg-cyan-950" 
                : "bg-gray-900 border-gray-800 hover:bg-gray-800"
        }}`;
        
        card.onclick = () => {{
            video.currentTime = item.start;
            video.play();
        }};

        card.innerHTML = `
            <div class="flex justify-between items-center mb-1 text-xs">
                <span class="${{item.is_gov ? "text-[#8a7eff] font-bold" : "text-gray-400"}}">
                    👤 ${{item.is_gov ? "総裁 (Governor)" : "記者/その他"}}
                </span>
                <span class="text-gray-500 font-mono">🕒 ${{formatTime(item.start)}}</span>
            </div>
            <p class="text-sm leading-relaxed">${{item.text}}</p>
        `;
        transcriptContainer.appendChild(card);
    }});

    // 3. 時間フォーマット
    function formatTime(seconds) {{
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}}`;
    }}

    // 4. 動画再生とアライメント・シークの同期
    video.addEventListener("timeupdate", () => {{
        const curTime = video.currentTime;
        timeDisplay.innerText = "Elapsed: " + formatTime(curTime);

        // チャートのシークバー移動
        chart.config.options.plugins.verticalLine.xValue = curTime / 60.0;
        chart.update('none');

        // 文字起こしのハイライト & 自動スクロール
        let activeIdx = -1;
        for (let i = 0; i < transcriptData.length; i++) {{
            if (curTime >= transcriptData[i].start && curTime <= transcriptData[i].end) {{
                activeIdx = i;
                break;
            }}
        }}

        if (activeIdx !== -1) {{
            // 以前のハイライトを削除
            document.querySelectorAll('.active-glow').forEach(el => el.classList.remove('active-glow'));
            
            const activeCard = document.getElementById(`card-${{activeIdx}}`);
            if (activeCard) {{
                activeCard.classList.add('active-glow');
                
                // スクロールコンテナ内で中央にスクロール
                const containerHeight = transcriptContainer.clientHeight;
                const cardTop = activeCard.offsetTop;
                const cardHeight = activeCard.clientHeight;
                transcriptContainer.scrollTo({{
                    top: cardTop - (containerHeight / 2) + (cardHeight / 2),
                    behavior: 'smooth'
                }});
            }}
        }}
    }});
</script>
</body>
</html>
"""
# リアルタイムビューアーの描画 (高さ790px)
components.html(custom_html, height=790, scrolling=False)

# 4. 回帰統計結果の表示
st.markdown("---")
st.subheader("📈 感情指標と為替リターンのOLS回帰分析 (HAC補正)")

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
            color="#635bff"
        ),
        marker=dict(size=12, color="#635bff"),
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
