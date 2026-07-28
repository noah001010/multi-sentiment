#!/bin/bash
# ============================================================
# start_demo.sh  — 日銀マルチモーダル感情分析デモ 一発起動スクリプト
# ============================================================
cd "$(dirname "$0")"

echo "🎬 動画配信サーバーをバックグラウンドで起動中 (ポート 8000)..."
python -m http.server 8000 --directory static > /dev/null 2>&1 &
VIDEO_SERVER_PID=$!
echo "   PID: $VIDEO_SERVER_PID"

echo "🚀 Streamlit ダッシュボードを起動中 (ポート 8501)..."
streamlit run app_viewer.py

# Streamlit が終了したら動画サーバーも止める
echo "🛑 動画配信サーバーを停止中..."
kill $VIDEO_SERVER_PID 2>/dev/null
