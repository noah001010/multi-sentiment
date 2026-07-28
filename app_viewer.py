import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import html as html_lib
import urllib.request
from pathlib import Path
import streamlit.components.v1 as components

# --- ページ設定 ---
st.set_page_config(page_title="日銀総裁会見 マルチモーダル感情分析", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(99,91,255,0.3) !important;
    }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }
    h1, h2, h3 {
        background: linear-gradient(135deg, #635bff 0%, #a388ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def safe_float(v, default=0.0):
    try:
        f = float(v)
        return default if (f != f or abs(f) == float('inf')) else f
    except Exception:
        return default


def load_forex(csv_path, start_str):
    df = pd.read_csv(csv_path, sep=';', header=None,
                     names=['dt_str','open','high','low','close','vol'])
    df['datetime'] = pd.to_datetime(df['dt_str'], format='%Y%m%d %H%M%S') + pd.Timedelta(hours=14)
    df.sort_values('datetime', inplace=True)
    df['return'] = np.log(df['close'] / df['close'].shift(1)) * 100
    t0 = pd.to_datetime(start_str)
    t1 = t0 + pd.Timedelta(hours=1)
    out = df[(df['datetime'] >= t0) & (df['datetime'] <= t1)].copy()
    out.dropna(subset=['return'], inplace=True)
    return out[['datetime','close','return']]


def normalize(series):
    s = series.fillna(0.0)
    lo, hi = s.min(), s.max()
    if lo == hi:
        return pd.Series(0.0, index=series.index)
    return -1.0 + 2.0 * (s - lo) / (hi - lo)


def alias(df, target, candidates):
    if target not in df.columns:
        for c in candidates:
            if c in df.columns:
                df[target] = df[c]
                return
        df[target] = 0.0


# ── Chart.js をインライン埋め込み（CDN依存・タイミング問題を完全排除）──
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
CHARTJS_PATH = STATIC_DIR / "chart.umd.min.js"

@st.cache_resource
def get_chartjs_script():
    """Chart.jsをローカルファイルから読み込み、なければCDNからダウンロード"""
    if not CHARTJS_PATH.exists():
        try:
            urllib.request.urlretrieve(
                "https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.min.js",
                str(CHARTJS_PATH)
            )
        except Exception:
            return '<script src="https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.min.js"></script>'

    if CHARTJS_PATH.exists():
        content = CHARTJS_PATH.read_text(encoding='utf-8')
        return f"<script>{content}</script>"
    else:
        return '<script src="https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.min.js"></script>'

chartjs_script = get_chartjs_script()


# ── 固定設定 ──
VIDEO_PATH = "data/boj_conference.mp4"
INTEG_PATH = "output/integrated_results.csv"
FOREX_PATH = "data/DAT_ASCII_USDJPY_M1_2023.csv"
START_STR  = "2023-06-16 15:30:00"

# ── サイドバー ──
with st.sidebar:
    st.header("⚡ System Status")
    st.success("Pipeline: ONLINE")
    st.code(f"Video : {VIDEO_PATH}\nCSV   : {INTEG_PATH}\nForex : {FOREX_PATH}\nStart : {START_STR}")

for p in [INTEG_PATH, FOREX_PATH]:
    if not os.path.exists(p):
        st.error(f"ファイルが見つかりません: {p}")
        st.stop()

# ── データ読み込み ──
df = pd.read_csv(INTEG_PATH)
df_fin = load_forex(FOREX_PATH, START_STR)

# 列名吸収
alias(df, 'text',               ['sentence','content','transcript'])
alias(df, 'text_score',         ['sentiment_score','text_score_mean','sentiment'])
alias(df, 'face_emotion_score', ['mean_valence','valence','face_valence'])
alias(df, 'face_arousal_score', ['mean_arousal','arousal','face_arousal'])
alias(df, 'audio_emotion_score',['audio_valence','audio_sentiment'])
alias(df, 'audio_arousal_score',['audio_arousal'])
alias(df, 'start',              ['start_time','start_sec'])
alias(df, 'end',                ['end_time','end_sec'])

# is_governor
if 'is_governor' in df.columns:
    df['is_governor'] = df['is_governor'].astype(str).str.lower().isin(['true','1','t','1.0'])
elif 'speaker' in df.columns:
    df['is_governor'] = df['speaker'].astype(str).str.upper().str.startswith('SPEAKER_00')
else:
    df['is_governor'] = False

# NaN クレンジング
for col in ['start','end','text_score','face_emotion_score','face_arousal_score',
            'audio_emotion_score','audio_arousal_score']:
    df[col] = df[col].fillna(0.0)
df['text'] = df['text'].fillna('').astype(str)

# ── 動画 ──
video_basename = os.path.basename(VIDEO_PATH)
video_static = STATIC_DIR / video_basename
if not video_static.exists() and os.path.exists(VIDEO_PATH):
    import shutil
    shutil.copy(VIDEO_PATH, video_static)

# ── チャートデータ（1分足） ──
t0 = pd.to_datetime(START_STR)
df['datetime'] = t0 + pd.to_timedelta(df['start'], unit='s')
df_1min = df.set_index('datetime').resample('1min').mean(numeric_only=True).reset_index()

avail = [v for v in ['text_score','face_emotion_score','audio_emotion_score',
                     'face_arousal_score','audio_arousal_score'] if v in df_1min.columns]
df_m = pd.merge(df_fin, df_1min, on='datetime', how='inner').dropna(subset=['return']+avail)
df_p = df_m.copy()
for col in avail:
    df_p[col] = normalize(df_p[col])

chart_data = []
for _, row in df_p.iterrows():
    mins = (row['datetime'] - t0).total_seconds() / 60.0
    chart_data.append({
        "m":         round(safe_float(mins), 2),
        "close":     round(safe_float(row['close']), 4),
        "text":      round(safe_float(row.get('text_score', 0)), 4),
        "face_val":  round(safe_float(row.get('face_emotion_score', 0)), 4),
        "audio_val": round(safe_float(row.get('audio_emotion_score', 0)), 4),
    })
chart_json = json.dumps(chart_data)

# ── 発言カード HTML（Python側で生成） ──
def fmt_time(sec):
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
    b_color  = "#0c0d1a" if is_gov else "#111827"
    b_border = "#4f46e5" if is_gov else "#374151"
    sp_color = "#8a7eff" if is_gov else "#9ca3af"

    cards_html += f"""<div style="background:{b_color};border:1px solid {b_border};padding:10px 12px;border-radius:10px;cursor:pointer;margin-bottom:6px;transition:all 0.15s" onclick="seekTo({start_s})" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;font-size:11px">
    <span style="color:{sp_color};font-weight:bold">{speaker}</span>
    <span style="color:#818cf8;font-family:monospace;text-decoration:underline;cursor:pointer" onclick="event.stopPropagation();seekTo({start_s})">{fmt_time(start_s)}</span>
  </div>
  <p style="font-size:13px;line-height:1.5;color:#f1f5f9;margin:0">{text}</p>
  <div style="font-size:10px;font-family:monospace;color:#6366f1;margin-top:4px">感情スコア: {sign}{score:.2f}</div>
</div>
"""

with st.sidebar:
    st.markdown("---")
    st.markdown(f"**発言セグメント:** {len(df)} 件")
    st.markdown(f"**チャートデータ:** {len(chart_data)} 件")
    chartjs_ok = "✅ ローカル" if CHARTJS_PATH.exists() else "⚠️ CDN"
    st.markdown(f"**Chart.js:** {chartjs_ok}")

st.title("日銀総裁会見　マルチモーダル感情分析")

# ── HTML ダッシュボード ──
custom_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
{chartjs_script}
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0b0f19; color: #f8fafc;
  margin: 0; padding: 8px; box-sizing: border-box;
}}
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: #101625; }}
::-webkit-scrollbar-thumb {{ background: #4b5563; border-radius: 3px; }}
</style>
</head>
<body>
<div style="display:grid;grid-template-columns:2fr 1fr;gap:8px;height:630px">

  <!-- 左: 動画 + 感情チャート + USD/JPYサブチャート -->
  <div style="display:flex;flex-direction:column;gap:6px;height:630px;overflow:hidden">

    <!-- 動画 (高さ固定) -->
    <div style="background:#101625;padding:6px;border-radius:12px;border:1px solid #1e1b4b;flex-shrink:0">
      <video id="vid" controls preload="metadata"
             style="width:100%;max-height:160px;border-radius:8px;display:block">
        <source src="http://localhost:8000/{video_basename}" type="video/mp4">
        <source src="/app/static/{video_basename}" type="video/mp4">
      </video>
    </div>

    <!-- 感情分析チャート (残りスペースを占有) -->
    <div style="background:#101625;padding:10px 12px 8px;border-radius:12px;border:1px solid #1e1b4b;flex:1;display:flex;flex-direction:column;min-height:0">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;flex-shrink:0">
        <span style="color:#8a7eff;font-weight:bold;font-size:13px">感情分析の推移</span>
        <span id="timedisp" style="font-family:monospace;font-size:12px;background:rgba(0,0,0,0.5);padding:3px 10px;border-radius:6px;border:1px solid #374151;color:#fff">00:00</span>
      </div>
      <div style="flex:1;min-height:0;position:relative">
        <canvas id="sentChart"></canvas>
      </div>
    </div>

    <!-- USD/JPY サブチャート (下部固定) -->
    <div style="background:#101625;padding:8px 12px 6px;border-radius:12px;border:1px solid #1e1b4b;flex-shrink:0;height:100px">
      <div style="font-size:11px;color:#f1c40f;font-weight:bold;margin-bottom:3px">為替相場 USD/JPY</div>
      <div style="height:70px;position:relative">
        <canvas id="forexChart"></canvas>
      </div>
    </div>
  </div>

  <!-- 右: 発言内容 -->
  <div style="background:#101625;padding:12px;border-radius:12px;border:1px solid #1e1b4b;display:flex;flex-direction:column;height:630px">
    <div style="color:#8a7eff;font-weight:bold;font-size:14px;padding-bottom:8px;border-bottom:1px solid #374151;margin-bottom:8px;flex-shrink:0">発言内容</div>
    <p style="font-size:11px;color:#6b7280;margin:0 0 8px;flex-shrink:0">タイムスタンプをクリックすると該当シーンへジャンプします。</p>
    <div style="flex:1;overflow-y:auto;padding-right:4px">
      {cards_html}
    </div>
  </div>

</div>

<script>
var chartData = {chart_json};

function fmt(s) {{
  if (!isFinite(s) || s < 0) s = 0;
  var m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return ('0'+m).slice(-2) + ':' + ('0'+sec).slice(-2);
}}

function seekTo(t) {{
  var v = document.getElementById('vid');
  if (!v) return;
  v.currentTime = parseFloat(t);
  var p = v.play();
  if (p && typeof p.catch === 'function') {{ p.catch(function(){{}}); }}
}}

(function() {{
  var sentCanvas  = document.getElementById('sentChart');
  var forexCanvas = document.getElementById('forexChart');
  var video = document.getElementById('vid');
  var disp  = document.getElementById('timedisp');
  if (!sentCanvas || !forexCanvas || !video || !disp) return;
  if (typeof Chart === 'undefined') return;

  // 現在再生位置 (分) を両チャートで共有
  var currentMin = 0;

  // 再生位置縦線プラグイン（両チャートに適用）
  var vlPlugin = {{
    id: 'vl',
    afterDraw: function(c) {{
      if (currentMin <= 0) return;
      var xa = c.scales.x, ya = c.scales.y;
      if (!xa || !ya) return;
      var px = xa.getPixelForValue(currentMin);
      if (px < xa.left || px > xa.right) return;
      var ctx = c.ctx;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(px, ya.top);
      ctx.lineTo(px, ya.bottom);
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.shadowColor = '#a78bfa';
      ctx.shadowBlur = 10;
      ctx.stroke();
      ctx.restore();
    }}
  }};
  Chart.register(vlPlugin);

  // ── 感情分析チャート (言語・表情・音声) ──
  var sentChart = new Chart(sentCanvas.getContext('2d'), {{
    type: 'line',
    data: {{
      labels: chartData.map(function(d){{ return d.m; }}),
      datasets: [
        {{ label:'言語感情',
           data: chartData.map(function(d){{ return d.text; }}),
           borderColor:'#2ecc71', backgroundColor:'transparent',
           borderWidth:2, pointRadius:0, tension:0.3 }},
        {{ label:'表情感情',
           data: chartData.map(function(d){{ return d.face_val; }}),
           borderColor:'#e74c3c', backgroundColor:'transparent',
           borderWidth:2, pointRadius:0, tension:0.3 }},
        {{ label:'音声感情',
           data: chartData.map(function(d){{ return d.audio_val; }}),
           borderColor:'#3498db', backgroundColor:'transparent',
           borderWidth:2, pointRadius:0, tension:0.3 }},
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode:'index', intersect:false }},
      scales: {{
        x: {{
          ticks: {{ color:'#6b7280', maxTicksLimit:10, font:{{ size:10 }} }},
          grid:  {{ color:'rgba(255,255,255,0.04)' }}
        }},
        y: {{
          position: 'left',
          title: {{ display:true, text:'感情スコア [-1, 1]', color:'#9ca3af', font:{{ size:10 }} }},
          min: -1.2, max: 1.2,
          grid:  {{ color:'rgba(255,255,255,0.04)' }},
          ticks: {{ color:'#6b7280', font:{{ size:10 }} }}
        }}
      }},
      plugins: {{
        legend: {{ position:'top', align:'start',
          labels: {{ color:'#9ca3af', boxWidth:10, font:{{ size:10 }}, padding:8 }} }},
        vl: {{}}
      }}
    }}
  }});

  // ── USD/JPY サブチャート ──
  var forexChart = new Chart(forexCanvas.getContext('2d'), {{
    type: 'line',
    data: {{
      labels: chartData.map(function(d){{ return d.m; }}),
      datasets: [
        {{ label:'USD/JPY',
           data: chartData.map(function(d){{ return d.close; }}),
           borderColor:'#f1c40f', backgroundColor:'rgba(241,196,15,0.08)',
           borderWidth:2, pointRadius:0, tension:0.3, fill:true }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{
          ticks: {{ color:'#6b7280', maxTicksLimit:10, font:{{ size:9 }} }},
          grid:  {{ color:'rgba(255,255,255,0.04)' }}
        }},
        y: {{
          position: 'right',
          ticks: {{ color:'#f1c40f', font:{{ size:9 }}, maxTicksLimit:3 }},
          grid:  {{ color:'rgba(255,255,255,0.04)' }}
        }}
      }},
      plugins: {{
        legend: {{ display:false }},
        vl: {{}}
      }}
    }}
  }});

  // 動画時間に連動: タイムスタンプ表示 + 両チャートの縦線を同時更新
  video.addEventListener('timeupdate', function() {{
    var t = video.currentTime;
    disp.innerText = fmt(t);
    currentMin = t / 60.0;
    sentChart.update('none');
    forexChart.update('none');
  }});
}})();
</script>
</body>
</html>
"""

components.html(custom_html, height=640, scrolling=False)
