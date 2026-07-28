import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import streamlit.components.v1 as components

# --- ページ設定 ---
st.set_page_config(
    page_title="BOJ Multimodal Sentiment Dashboard", 
    layout="wide"
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
    
    /* 美しいグラデーション見出し */
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

# データ存在チェック
if not os.path.exists(integrated_path):
    st.warning(f"⚠️ `{integrated_path}` が見つかりません。先にデータ統合を完了してください。")
    st.stop()
if not os.path.exists(forex_path):
    st.warning(f"⚠️ `{forex_path}` が見つかりません。パスを確認してください。")
    st.stop()

# --- 規格化ヘルパー ---
def min_max_normalize(series: pd.Series) -> pd.Series:
    series_clean = series.fillna(0.0)
    s_min, s_max = series_clean.min(), series_clean.max()
    if s_min == s_max:
        return pd.Series(0.0, index=series.index)
    return -1.0 + 2.0 * (series_clean - s_min) / (s_max - s_min)

# --- データの読み込み ---
df_integ = pd.read_csv(integrated_path)
df_fin = load_and_filter_forex_data(forex_path, start_time_str)

# 欠損値 (NaN) の事前クレンジング (JS構文エラー防止)
df_integ['start'] = df_integ['start'].fillna(0.0)
df_integ['end'] = df_integ['end'].fillna(0.0)
df_integ['text'] = df_integ['text'].fillna('')
df_integ['speaker'] = df_integ['speaker'].fillna('UNKNOWN')
if 'is_governor' in df_integ.columns:
    df_integ['is_governor'] = df_integ['is_governor'].fillna(False)
else:
    df_integ['is_governor'] = False

if 'text_score' in df_integ.columns:
    df_integ['text_score'] = df_integ['text_score'].fillna(0.0)
else:
    df_integ['text_score'] = 0.0

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

# 時間アライメント処理 (1分足集計)
conference_start_time = pd.to_datetime(start_time_str)
df_integ['datetime'] = conference_start_time + pd.to_timedelta(df_integ['start'], unit='s')
df_integ_time = df_integ.set_index('datetime')
df_integ_1min = df_integ_time.resample('1min').mean(numeric_only=True).reset_index()

available_vars = [v for v in ['text_score', 'face_emotion_score', 'audio_emotion_score', 'face_arousal_score', 'audio_arousal_score'] if v in df_integ_1min.columns]
df_merged = pd.merge(df_fin, df_integ_1min, on='datetime', how='inner')
df_merged = df_merged.dropna(subset=['return'] + available_vars)

# JS/HTML コンポーネント用のデータ準備
df_plot = df_merged.copy()
for col in ['text_score', 'face_emotion_score', 'face_arousal_score', 'audio_emotion_score', 'audio_arousal_score']:
    if col in df_plot.columns:
        df_plot[col] = min_max_normalize(df_plot[col])

chart_data_list = []
for _, row in df_plot.iterrows():
    minutes = (row["datetime"] - conference_start_time).total_seconds() / 60.0
    chart_data_list.append({
        "m": round(float(minutes), 2),
        "close": round(float(row["close"]), 4),
        "text": round(float(row["text_score"]), 4) if 'text_score' in row else 0.0,
        "face_val": round(float(row["face_emotion_score"]), 4) if 'face_emotion_score' in row else 0.0,
        "face_aro": round(float(row["face_arousal_score"]), 4) if 'face_arousal_score' in row else 0.0,
        "audio_val": round(float(row["audio_emotion_score"]), 4) if 'audio_emotion_score' in row else 0.0,
        "audio_aro": round(float(row["audio_arousal_score"]), 4) if 'audio_arousal_score' in row else 0.0,
    })

# 全セグメントの文字起こしデータ
transcript_list = []
for idx, row in df_integ.iterrows():
    transcript_list.append({
        "id": int(idx),
        "start": float(row["start"]),
        "end": float(row["end"]),
        "text": str(row["text"]),
        "speaker": str(row.get("speaker", "UNKNOWN")),
        "is_gov": bool(row.get("is_governor", False)),
        "text_score": round(float(row.get("text_score", 0.0)), 2)
    })

chart_json = json.dumps(chart_data_list)
transcript_json = json.dumps(transcript_list)

# --- メインダッシュボード ---
st.title("BOJ Governor Multimodal Real-Time Aligner")

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
        background: rgba(99, 91, 255, 0.18) !important;
        border: 2px solid #635bff !important;
        box-shadow: 0 0 20px rgba(99, 91, 255, 0.4);
    }}
    ::-webkit-scrollbar {{
        width: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: #101625;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #4b5563;
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
            <video id="boj_video" class="w-full rounded-lg shadow-2xl" controls preload="auto">
                <source src="{video_url}" type="video/mp4">
                <source src="http://localhost:8000/{video_basename}" type="video/mp4">
            </video>
        </div>
        
        <!-- 同期ラインチャート -->
        <div class="bg-[#101625] p-4 rounded-xl glow-border">
            <h3 class="text-[#8a7eff] font-bold text-lg mb-2 flex justify-between items-center">
                感情分析の推移
                <span id="current_time_display" class="text-sm bg-[#0b0f19] px-3 py-1 rounded text-white border border-gray-700">00:00</span>
            </h3>
            <div style="position: relative; height: 260px; width: 100%;">
                <canvas id="sync_chart"></canvas>
            </div>
        </div>
    </div>
    
    <!-- 右カラム: 発言内容 -->
    <div class="bg-[#101625] p-4 rounded-xl glow-border flex flex-col h-[755px]">
        <h3 class="text-[#8a7eff] font-bold text-lg mb-3 pb-2 border-b border-gray-700">発言内容</h3>
        <p class="text-xs text-gray-400 mb-3">カードまたはタイムスタンプをクリックすると該当シーンへジャンプします。</p>
        <div id="transcript_container" class="flex-1 overflow-y-auto space-y-3 pr-2">
            <!-- JSで動的生成 -->
        </div>
    </div>
</div>

<script>
    // 0. 時間フォーマットヘルパー関数 (最上部に配置)
    function formatTime(seconds) {{
        if (isNaN(seconds) || seconds < 0) return "00:00";
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}}`;
    }}

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

    // カスタム垂直線描画プラグイン（現在再生時間の動的バー表示）
    const verticalLinePlugin = {{
        id: 'verticalLine',
        afterDraw: (chart) => {{
            if (chart.config.options.plugins.verticalLine && chart.config.options.plugins.verticalLine.xValue !== undefined) {{
                const xValue = chart.config.options.plugins.verticalLine.xValue;
                const xAxis = chart.scales.x;
                const yAxis = chart.scales.y;
                const xPixel = xAxis.getPixelForValue(xValue);
                
                if (xPixel >= xAxis.left && xPixel <= xAxis.right) {{
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(xPixel, yAxis.top);
                    ctx.lineTo(xPixel, yAxis.bottom);
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.strokeStyle = '#ffffff';
                    ctx.shadowColor = '#635bff';
                    ctx.shadowBlur = 8;
                    ctx.stroke();
                    ctx.restore();
                }}
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
                        text: '経過時間 (分足)',
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

    // 2. 安全な DOM 生成による発言内容リストのレンダリング
    transcriptData.forEach((item, index) => {{
        const card = document.createElement("div");
        card.id = `card-${{index}}`;
        card.className = `p-3 rounded-lg cursor-pointer transition duration-300 border ${{
            item.is_gov 
                ? "bg-[#0b0c10] border-indigo-900/60 hover:bg-indigo-950/40" 
                : "bg-gray-900 border-gray-800 hover:bg-gray-800"
        }}`;
        
        card.onclick = () => {{
            video.currentTime = item.start;
            video.play();
        }};

        const headerDiv = document.createElement("div");
        headerDiv.className = "flex justify-between items-center mb-1 text-xs";
        
        const speakerSpan = document.createElement("span");
        speakerSpan.className = item.is_gov ? "text-[#8a7eff] font-bold" : "text-gray-400";
        speakerSpan.textContent = item.is_gov ? "総裁" : "記者/その他";
        
        const timeSpan = document.createElement("span");
        timeSpan.className = "text-gray-400 hover:text-white font-mono underline cursor-pointer";
        timeSpan.textContent = formatTime(item.start);
        timeSpan.onclick = (e) => {{
            e.stopPropagation();
            video.currentTime = item.start;
            video.play();
        }};

        headerDiv.appendChild(speakerSpan);
        headerDiv.appendChild(timeSpan);

        const pText = document.createElement("p");
        pText.className = "text-sm leading-relaxed text-gray-100";
        pText.textContent = item.text;

        const scoreDiv = document.createElement("div");
        scoreDiv.className = "mt-2 text-[11px] text-indigo-300 font-mono bg-indigo-950/60 px-2 py-0.5 rounded inline-block";
        const scoreSign = item.text_score > 0 ? "+" : "";
        scoreDiv.textContent = `テキスト感情スコア: ${{scoreSign}}${{item.text_score}}`;

        card.appendChild(headerDiv);
        card.appendChild(pText);
        card.appendChild(scoreDiv);

        transcriptContainer.appendChild(card);
    }});

    // 3. 動画再生とアライメント・シークの同期イベント
    video.addEventListener("timeupdate", () => {{
        const curTime = video.currentTime;
        timeDisplay.innerText = formatTime(curTime);

        // チャートの現在再生時間を示す縦線の移動
        chart.config.options.plugins.verticalLine.xValue = curTime / 60.0;
        chart.update('none');

        // 発言内容のハイライト & 自動スクロール
        let activeIdx = -1;
        for (let i = 0; i < transcriptData.length; i++) {{
            if (curTime >= transcriptData[i].start && curTime <= transcriptData[i].end) {{
                activeIdx = i;
                break;
            }}
        }}

        if (activeIdx !== -1) {{
            document.querySelectorAll('.active-glow').forEach(el => el.classList.remove('active-glow'));
            
            const activeCard = document.getElementById(`card-${{activeIdx}}`);
            if (activeCard) {{
                activeCard.classList.add('active-glow');
                
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

components.html(custom_html, height=790, scrolling=False)
