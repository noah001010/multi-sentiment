#!/bin/bash
# ============================================================
# start_demo.sh  — 日銀マルチモーダル感情分析デモ 一発起動スクリプト
# ============================================================
cd "$(dirname "$0")"
mkdir -p static

# ── 動画をstaticに準備（faststartへの変換も試みる） ──────────────
VIDEO_SRC="data/boj_conference.mp4"
VIDEO_DST="static/boj_conference.mp4"

if [ ! -f "$VIDEO_DST" ]; then
  echo "🎬 動画をstaticにコピー中..."
  if command -v ffmpeg &> /dev/null; then
    ffmpeg -y -i "$VIDEO_SRC" -movflags +faststart -c copy "$VIDEO_DST" 2>/dev/null \
      && echo "   ✅ faststart変換完了" \
      || { echo "   ⚠️  ffmpeg変換失敗→単純コピー"; cp "$VIDEO_SRC" "$VIDEO_DST"; }
  else
    cp "$VIDEO_SRC" "$VIDEO_DST"
    echo "   ✅ コピー完了（ffmpegなし）"
  fi
else
  echo "✅ 動画ファイル確認済み"
fi

# ── Chart.js をローカルにキャッシュ（CDN不要化） ────────────────
CHARTJS="static/chart.umd.min.js"
if [ ! -f "$CHARTJS" ]; then
  echo "📦 Chart.js をダウンロード中..."
  curl -fsSL "https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.min.js" -o "$CHARTJS" \
    && echo "   ✅ ダウンロード完了 ($(du -sh $CHARTJS | cut -f1))" \
    || echo "   ⚠️  ダウンロード失敗 (CDNにフォールバックします)"
else
  echo "✅ Chart.js キャッシュ確認済み"
fi

# ── Range Request 対応動画サーバーを起動（port 8000） ────────────
echo "🧹 古い動画サーバーが残っている場合はクリーンアップします..."
fuser -k 8000/tcp 2>/dev/null || true
sleep 1

echo "🌐 Range-capable 動画サーバーを起動中 (port 8000)..."
python3 video_server.py 8000 static &
VIDEO_SERVER_PID=$!
sleep 1  # サーバーの起動を待つ

# 起動確認
if kill -0 $VIDEO_SERVER_PID 2>/dev/null; then
  echo "   ✅ PID: $VIDEO_SERVER_PID"
else
  echo "   ❌ サーバー起動失敗"
fi

# ── Streamlit ────────────────────────────────────────────────────
echo ""
echo "🚀 Streamlit を起動中 (port 8501)..."
echo "   ブラウザ: http://localhost:8501"
echo ""
streamlit run app_viewer.py

# Streamlit が終了したら動画サーバーも止める
echo "🛑 動画サーバーを停止中..."
kill $VIDEO_SERVER_PID 2>/dev/null
