import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Multimodal Discrepancy Analysis", layout="wide")

# ==========================================
# 1. データ準備（バックエンド処理・分析）
# ==========================================
def get_discrepancy_data():
    """
    言語感情と表情・音声感情の乖離点を検出するロジック（ダミーデータ）
    """
    return [
        {
            "timestamp": 15.5, 
            "time_str": "00:00:15",
            "type": "テキスト(ハト派) vs 表情(極めてネガティブ)",
            "score": 0.85,
            "text": "現在の景気は緩やかに回復しており..."
        },
        {
            "timestamp": 42.0, 
            "time_str": "00:00:42",
            "type": "テキスト(タカ派) vs 音声(ニュートラル)",
            "score": 0.62,
            "text": "物価上昇の圧力には引き続き注視が必要で..."
        },
        {
            "timestamp": 120.0, 
            "time_str": "00:02:00",
            "type": "テキスト(中立) vs 表情(怒り/恐れ)",
            "score": 0.78,
            "text": "海外市場の動向については予測が困難ですが..."
        }
    ]

discrepancy_data = get_discrepancy_data()

# ==========================================
# 2. HTML/JS フロントエンドコンポーネントの構築
# ==========================================
data_json = json.dumps(discrepancy_data)

# ローカルの実データ（boj_conference.mp4）をStreamlitのStatic Serving機能で読み込む
video_url = "/app/static/boj_conference.mp4" 

custom_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #fff; background-color: #0e1117; margin: 0; padding: 10px; }}
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
            Your browser does not support the video tag.
        </video>
    </div>

    <div class="list-section" id="list_container">
        <h3 style="margin-top: 0;">⚠️ 乖離ポイント検出</h3>
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

    discrepancyData.forEach((item, index) => {{
        const card = document.createElement("div");
        card.className = "discrepancy-card";
        
        card.onclick = () => jumpToTime(item.timestamp);
        
        card.innerHTML = `
            <div>
                <span class="time-badge">🕒 ${{item.time_str}}</span>
                <span class="score">Score: ${{item.score}}</span>
            </div>
            <div style="margin-top: 8px; font-size: 14px; font-weight: bold;">${{item.type}}</div>
            <div style="margin-top: 5px; font-size: 13px; color: #aaa; font-style: italic;">"${{item.text}}"</div>
        `;
        listContainer.appendChild(card);
    }});
</script>
</body>
</html>
"""

# ==========================================
# 3. UIの組み立て
# ==========================================
st.title("🛡️ マルチモーダル感情分析ダッシュボード")
st.markdown("テキスト感情と非言語（表情・音声）の感情が乖離しているポイントをリスト化しています。リストをクリックすると、動画のその瞬間に即座にジャンプして再生します。")

# 作成したHTMLコンポーネントを描画
components.html(custom_html, height=500, scrolling=False)

st.markdown("---")
st.subheader("分析サマリーチャート")
st.info("ここに Plotly などのチャートを通常通り配置できます。")
