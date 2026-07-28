import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import html as html_lib
from pathlib import Path
import streamlit.components.v1 as components

# --- ページ設定 ---
st.set_page_config(
    page_title="BOJ Multimodal Sentiment Dashboard",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(99, 91, 255, 0.3) !important;
    }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }
    h1, h2, h3 {
        background: linear-gradient(135deg, #635bff 0%, #a388ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def safe_float(v, default=0.0):
    """NaN/Infを安全にfloatに変換する"""
    try:
        f = float(v)
        if f != f or f == float('inf') or f == float('-inf'):
            return default
        return f
    except Exception:
        return default


def load_and_filter_forex_data(csv_path: str, start_str: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=';', header=None,
                     names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S')
    df['datetime'] += pd.Timedelta(hours=14)
    df.sort_values('datetime', inplace=True)
    df['return'] = np.log(df['close'] / df['close'].shift(1)) * 100
    t0 = pd.to_datetime(start_str)
    t1 = t0 + pd.Timedelta(hours=1)
    out = df[(df['datetime'] >= t0) & (df['datetime'] <= t1)].copy()
    out.dropna(subset=['return'], inplace=True)
    return out[['datetime', 'close', 'return']]


def min_max_normalize(series: pd.Series) -> pd.Series:
    s = series.fillna(0.0)
    lo, hi = s.min(), s.max()
    if lo == hi:
        return pd.Series(0.0, index=series.index)
    return -1.0 + 2.0 * (s - lo) / (hi - lo)


# ─────────────────────────────────────────────
# 固定設定
# ─────────────────────────────────────────────
VIDEO_PATH = "data/boj_conference.mp4"
INTEG_PATH = "output/integrated_results.csv"
FOREX_PATH = "data/DAT_ASCII_USDJPY_M1_2023.csv"
START_STR  = "2023-06-16 15:30:00"

# ─────────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚡ System Status")
    st.success("Pipeline: ONLINE")
    st.code(f"Video : {VIDEO_PATH}\nCSV   : {INTEG_PATH}\nForex : {FOREX_PATH}\nStart : {START_STR}")

# ─────────────────────────────────────────────
# ファイル存在チェック
# ─────────────────────────────────────────────
for p in [INTEG_PATH, FOREX_PATH]:
    if not os.path.exists(p):
        st.error(f"ファイルが見つかりません: {p}")
        st.stop()

# ─────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────
df = pd.read_csv(INTEG_PATH)
df_fin = load_and_filter_forex_data(FOREX_PATH, START_STR)

# ── 列名の吸収 ──────────────────────────────
def alias(df, target, candidates):
    if target not in df.columns:
        for c in candidates:
            if c in df.columns:
                df[target] = df[c]; return
        df[target] = 0.0

alias(df, 'text',              ['sentence', 'content', 'transcript'])
alias(df, 'text_score',        ['sentiment_score', 'text_score_mean', 'sentiment'])
alias(df, 'face_emotion_score',['mean_valence',  'valence',  'face_valence'])
alias(df, 'face_arousal_score',['mean_arousal',  'arousal',  'face_arousal'])
alias(df, 'audio_emotion_score',['audio_valence', 'audio_sentiment'])
alias(df, 'audio_arousal_score',['audio_arousal'])
alias(df, 'start',             ['start_time', 'start_sec'])
alias(df, 'end',               ['end_time',   'end_sec'])

# ── is_governor 変換 ──────────────────────────
if 'is_governor' in df.columns:
    df['is_governor'] = df['is_governor'].astype(str).str.lower().isin(['true','1','t','1.0'])
elif 'speaker' in df.columns:
    df['is_governor'] = df['speaker'].astype(str).str.upper().str.startswith('SPEAKER_00')
else:
    df['is_governor'] = False

# ── NaN クレンジング ──────────────────────────
for col in ['start','end','text_score','face_emotion_score','face_arousal_score',
            'audio_emotion_score','audio_arousal_score']:
    df[col] = df[col].fillna(0.0)
df['text']    = df['text'].fillna('').astype(str)
df['speaker'] = df.get('speaker', pd.Series(['UNKNOWN']*len(df))).fillna('UNKNOWN')

# ─────────────────────────────────────────────
# 動画 static コピー
# ─────────────────────────────────────────────
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
video_basename = os.path.basename(VIDEO_PATH)
target_static = static_dir / video_basename
if not target_static.exists() and os.path.exists(VIDEO_PATH):
    try:
        import shutil; shutil.copy(VIDEO_PATH, target_static)
    except Exception as e:
        st.warning(f"動画コピー失敗: {e}")

video_url = f"/app/static/{video_basename}"

# ─────────────────────────────────────────────
# チャートデータ（1分足集計）
# ─────────────────────────────────────────────
t0 = pd.to_datetime(START_STR)
df['datetime'] = t0 + pd.to_timedelta(df['start'], unit='s')
df_1min = df.set_index('datetime').resample('1min').mean(numeric_only=True).reset_index()

avail = [v for v in ['text_score','face_emotion_score','audio_emotion_score',
                      'face_arousal_score','audio_arousal_score'] if v in df_1min.columns]

df_merged = pd.merge(df_fin, df_1min, on='datetime', how='inner').dropna(subset=['return'] + avail)

df_plot = df_merged.copy()
for col in avail:
    df_plot[col] = min_max_normalize(df_plot[col])

chart_data = []
for _, row in df_plot.iterrows():
    mins = (row['datetime'] - t0).total_seconds() / 60.0
    chart_data.append({
        "m":         round(safe_float(mins), 2),
        "close":     round(safe_float(row['close']), 4),
        "text":      round(safe_float(row.get('text_score', 0)), 4),
        "face_val":  round(safe_float(row.get('face_emotion_score', 0)), 4),
        "face_aro":  round(safe_float(row.get('face_arousal_score', 0)), 4),
        "audio_val": round(safe_float(row.get('audio_emotion_score', 0)), 4),
    })

chart_json = json.dumps(chart_data)

# ─────────────────────────────────────────────
# 発言カード HTML をPython側で生成（JS不要）
# ─────────────────────────────────────────────
def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec//60:02d}:{sec%60:02d}"

cards_html = ""
for _, row in df.iterrows():
    is_gov   = bool(row['is_governor'])
    start_s  = safe_float(row['start'])
    text     = html_lib.escape(str(row['text']))
    score    = safe_float(row['text_score'])
    sign     = "+" if score > 0 else ""
    speaker  = "総裁" if is_gov else "記者/その他"
    card_cls = ("bg-[#0c0d1a] border-indigo-800/50 hover:bg-indigo-950/40"
                if is_gov else
                "bg-[#111827] border-gray-700/50 hover:bg-gray-700/40")
    sp_cls   = "text-[#8a7eff] font-bold" if is_gov else "text-gray-400"
    time_str = fmt_time(start_s)

    cards_html += f"""
<div class="card p-3 rounded-xl border {card_cls} cursor-pointer transition-all duration-200"
     onclick="seekTo({start_s})">
  <div class="flex justify-between items-center mb-1 text-xs">
    <span class="{sp_cls}">{speaker}</span>
    <span class="font-mono text-indigo-300 hover:text-white underline"
          onclick="event.stopPropagation(); seekTo({start_s})">{time_str}</span>
  </div>
  <p class="text-sm leading-relaxed text-gray-100">{text}</p>
  <div class="mt-1 text-[10px] font-mono text-indigo-400/80">
    感情スコア: {sign}{score:.2f}
  </div>
</div>
"""

# ─────────────────────────────────────────────
# サイドバーにデバッグ件数を表示
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown(f"**発言セグメント:** {len(df)} 件")
    st.markdown(f"**チャートデータ:** {len(chart_data)} 件")

# ─────────────────────────────────────────────
# メインダッシュボード
# ─────────────────────────────────────────────
st.title("BOJ Governor Multimodal Real-Time Aligner")

custom_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
  body {{ font-family:'Outfit',sans-serif; background:#0b0f19; color:#f8fafc; margin:0; padding:8px; }}
  ::-webkit-scrollbar {{ width:5px; }}
  ::-webkit-scrollbar-track {{ background:#101625; }}
  ::-webkit-scrollbar-thumb {{ background:#4b5563; border-radius:3px; }}
  .active-card {{
    background: rgba(99,91,255,0.2) !important;
    border-color: #635bff !important;
    box-shadow: 0 0 16px rgba(99,91,255,0.35);
  }}
</style>
</head>
<body>
<div class="grid grid-cols-1 lg:grid-cols-3 gap-3" style="height:790px">

  <!-- 左2/3: 動画 + チャート -->
  <div class="lg:col-span-2 flex flex-col gap-3">

    <!-- 動画 -->
    <div class="bg-[#101625] p-2 rounded-xl border border-indigo-900/30">
      <video id="vid" class="w-full rounded-lg" controls preload="auto">
        <source src="{video_url}" type="video/mp4">
        <source src="http://localhost:8000/{video_basename}" type="video/mp4">
      </video>
    </div>

    <!-- チャート -->
    <div class="bg-[#101625] p-3 rounded-xl border border-indigo-900/30 flex-1">
      <div class="flex justify-between items-center mb-1">
        <span class="text-[#8a7eff] font-bold text-base">感情分析の推移</span>
        <span id="timedisp" class="text-xs font-mono bg-black/40 px-3 py-1 rounded border border-gray-700 text-white">00:00</span>
      </div>
      <div style="position:relative;height:210px;width:100%">
        <canvas id="chart"></canvas>
      </div>
    </div>
  </div>

  <!-- 右1/3: 発言内容 (Python生成HTML) -->
  <div class="bg-[#101625] p-3 rounded-xl border border-indigo-900/30 flex flex-col" style="height:790px">
    <div class="text-[#8a7eff] font-bold text-base pb-2 border-b border-gray-700 mb-2">発言内容</div>
    <p class="text-xs text-gray-400 mb-2">タイムスタンプをクリックすると該当シーンへジャンプします。</p>
    <div id="tc" class="flex-1 overflow-y-auto space-y-2 pr-1">
      {cards_html}
    </div>
  </div>

</div>

<script>
const chartData = {chart_json};
const video     = document.getElementById('vid');
const timeDisp  = document.getElementById('timedisp');
const tc        = document.getElementById('tc');

// ── 時間フォーマット ──────────────────────────
function fmt(s) {{
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s/60), sec = Math.floor(s%60);
  return `${{String(m).padStart(2,'0')}}:${{String(sec).padStart(2,'0')}}`;
}}

// ── シーク関数（全カードから呼ばれる） ──────────
function seekTo(t) {{
  video.currentTime = t;
  video.play().catch(()=>{{}});
}}

// ── Chart.js ──────────────────────────────────
const vlPlugin = {{
  id:'vl',
  afterDraw(c) {{
    const xv = c.config.options.plugins.vl?.x;
    if (xv === undefined) return;
    const xa = c.scales.x, ya = c.scales.y;
    const px = xa.getPixelForValue(xv);
    if (px < xa.left || px > xa.right) return;
    const ctx = c.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(px, ya.top); ctx.lineTo(px, ya.bottom);
    ctx.lineWidth = 2;
    ctx.setLineDash([5,4]);
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.shadowColor = '#635bff'; ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.restore();
  }}
}};
Chart.register(vlPlugin);

const chart = new Chart(document.getElementById('chart').getContext('2d'), {{
  type:'line',
  data:{{
    labels: chartData.map(d=>d.m),
    datasets:[
      {{ label:'言語感情',   data:chartData.map(d=>d.text),      borderColor:'#2ecc71', borderWidth:2, pointRadius:0, yAxisID:'e' }},
      {{ label:'表情ポジネガ',data:chartData.map(d=>d.face_val),  borderColor:'#e74c3c', borderWidth:2, pointRadius:0, yAxisID:'e' }},
      {{ label:'表情緊張度', data:chartData.map(d=>d.face_aro),  borderColor:'#9b59b6', borderWidth:2, pointRadius:0, yAxisID:'e', borderDash:[5,5] }},
      {{ label:'音声感情',   data:chartData.map(d=>d.audio_val), borderColor:'#3498db', borderWidth:2, pointRadius:0, yAxisID:'e' }},
      {{ label:'USD/JPY',    data:chartData.map(d=>d.close),     borderColor:'#f1c40f', borderWidth:3, pointRadius:0, yAxisID:'f' }},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    scales:{{
      x:{{ title:{{display:true,text:'経過時間 (分足)',color:'#9ca3af'}}, grid:{{color:'#1f2833'}}, ticks:{{color:'#9ca3af'}} }},
      e:{{ position:'left', title:{{display:true,text:'感情スコア [-1,1]',color:'#9ca3af'}}, min:-1.2, max:1.2, grid:{{color:'#1f2833'}}, ticks:{{color:'#9ca3af'}} }},
      f:{{ position:'right', title:{{display:true,text:'USD/JPY',color:'#9ca3af'}}, grid:{{drawOnChartArea:false}}, ticks:{{color:'#9ca3af'}} }}
    }},
    plugins:{{
      legend:{{ position:'top', labels:{{color:'#9ca3af',boxWidth:10,font:{{size:10}}}} }},
      vl:{{ x: 0 }}
    }}
  }}
}});

// ── 再生イベント ─────────────────────────────
const allCards = tc.querySelectorAll('.card');

video.addEventListener('timeupdate', ()=>{{
  const t = video.currentTime;
  timeDisp.innerText = fmt(t);

  // チャート縦線
  chart.config.options.plugins.vl.x = t / 60.0;
  chart.update('none');
}});
</script>
</body>
</html>
"""

components.html(custom_html, height=795, scrolling=False)
