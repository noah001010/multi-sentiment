#!/bin/bash
# ============================================================
# start_demo.sh  — 日銀マルチモーダル感情分析デモ 一発起動スクリプト
# ============================================================
cd "$(dirname "$0")"

mkdir -p static

# ── 動画を「シーク対応」フォーマットに変換 (faststart) ──────────────
VIDEO_SRC="data/boj_conference.mp4"
VIDEO_DST="static/boj_conference.mp4"

if [ ! -f "$VIDEO_DST" ]; then
  echo "🎬 動画をシーク対応フォーマットに変換中..."
  if command -v ffmpeg &> /dev/null; then
    ffmpeg -y -i "$VIDEO_SRC" -movflags +faststart -c copy "$VIDEO_DST" 2>/dev/null \
      && echo "   ✅ faststart変換完了" \
      || { echo "   ⚠️  ffmpeg変換失敗→単純コピー"; cp "$VIDEO_SRC" "$VIDEO_DST"; }
  else
    echo "   ⚠️  ffmpegなし→単純コピー (シークが不安定な場合はffmpegをインストールしてください)"
    cp "$VIDEO_SRC" "$VIDEO_DST"
  fi
else
  echo "✅ 動画ファイル確認済み: $VIDEO_DST"
fi

# ── Chart.js をローカルにキャッシュ (CDN不要化) ────────────────────
CHARTJS="static/chart.umd.min.js"
if [ ! -f "$CHARTJS" ]; then
  echo "📦 Chart.js をローカルにダウンロード中..."
  curl -fsSL "https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.min.js" -o "$CHARTJS" \
    && echo "   ✅ ダウンロード完了" \
    || echo "   ⚠️  ダウンロード失敗 (CDNを使用します)"
else
  echo "✅ Chart.js キャッシュ確認済み"
fi

# ── 静的ファイル配信サーバー (port 8000) ────────────────────────────
echo "🌐 静的ファイルサーバーをバックグラウンドで起動中 (ポート 8000)..."
python -m http.server 8000 --directory static > /dev/null 2>&1 &
VIDEO_SERVER_PID=$!
echo "   PID: $VIDEO_SERVER_PID"

# ── Streamlit ────────────────────────────────────────────────────────
echo "🚀 Streamlit ダッシュボードを起動中 (ポート 8501)..."
streamlit run app_viewer.py

# Streamlit が終了したら動画サーバーも止める
echo "🛑 動画配信サーバーを停止中..."
kill $VIDEO_SERVER_PID 2>/dev/null
