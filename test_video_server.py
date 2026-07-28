#!/usr/bin/env python3
"""
動画サーバーの Range Request 対応確認スクリプト
vectorサーバーで実行: python3 test_video_server.py
"""
import urllib.request
import subprocess
import os
import sys

VIDEO_FILE = "static/boj_conference.mp4"
PORT = 8000

print("=" * 60)
print("動画サーバー診断ツール")
print("=" * 60)

# 1. 動画ファイル確認
print(f"\n[1] 動画ファイル確認: {VIDEO_FILE}")
if os.path.exists(VIDEO_FILE):
    size = os.path.getsize(VIDEO_FILE)
    print(f"    ✅ 存在: {size:,} bytes ({size/1024/1024:.1f} MB)")
else:
    print(f"    ❌ ファイルが見つかりません")
    print(f"    → start_demo.sh を実行してください")
    sys.exit(1)

# 2. Port 8000 サーバーが起動しているか確認
print(f"\n[2] Port {PORT} サーバー確認")
try:
    req = urllib.request.Request(f"http://localhost:{PORT}/boj_conference.mp4")
    with urllib.request.urlopen(req, timeout=3) as r:
        print(f"    ✅ サーバー応答: HTTP {r.status}")
        print(f"    Content-Length: {r.headers.get('Content-Length', 'N/A')}")
        print(f"    Accept-Ranges: {r.headers.get('Accept-Ranges', 'N/A')}")
        print(f"    Content-Type: {r.headers.get('Content-Type', 'N/A')}")
except Exception as e:
    print(f"    ❌ サーバーに接続できません: {e}")
    print(f"    → start_demo.sh でサーバーを起動してください")
    sys.exit(1)

# 3. Range Request テスト
print(f"\n[3] Range Request テスト (シーク可否確認)")
try:
    req = urllib.request.Request(
        f"http://localhost:{PORT}/boj_conference.mp4",
        headers={"Range": "bytes=0-1023"}
    )
    with urllib.request.urlopen(req, timeout=3) as r:
        status = r.status
        accept_ranges = r.headers.get("Accept-Ranges", "")
        content_range = r.headers.get("Content-Range", "")
        data = r.read(10)  # 最初の10バイトだけ読む

        if status == 206:
            print(f"    ✅ 206 Partial Content → シーク可能!")
            print(f"    Content-Range: {content_range}")
        elif status == 200:
            print(f"    ⚠️  200 OK (Rangeが無視されています) → シーク不可の可能性あり")
            print(f"    Accept-Ranges: {accept_ranges}")
        else:
            print(f"    ❓ HTTP {status}")
except Exception as e:
    print(f"    ❌ Range request 失敗: {e}")

# 4. ffmpeg 確認
print(f"\n[4] ffmpeg (faststart変換) 確認")
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        version = result.stdout.split('\n')[0]
        print(f"    ✅ {version}")
        print(f"    → faststart変換が可能です")
    else:
        print(f"    ❌ ffmpeg が見つかりません")
except Exception as e:
    print(f"    ❌ ffmpeg が見つかりません: {e}")

# 5. moov atom 位置確認
print(f"\n[5] MP4 構造確認 (moov atom位置)")
try:
    with open(VIDEO_FILE, 'rb') as f:
        header = f.read(32)
    # 最初の4バイトがサイズ、次の4バイトがボックスタイプ
    box_type = header[4:8].decode('ascii', errors='ignore')
    if box_type == 'ftyp':
        print(f"    ✅ ftyp ボックスが先頭 → ブラウザ互換")
    elif box_type == 'moov':
        print(f"    ✅ moov が先頭 (faststart) → シーク最適")
    else:
        print(f"    ⚠️  先頭ボックス: '{box_type}' (faststart でない可能性)")
        print(f"    → ffmpeg -movflags +faststart で変換推奨")
except Exception as e:
    print(f"    ❌ ファイル読み込みエラー: {e}")

# 6. Chart.js キャッシュ確認
print(f"\n[6] Chart.js キャッシュ確認")
chartjs = "static/chart.umd.min.js"
if os.path.exists(chartjs):
    size = os.path.getsize(chartjs)
    print(f"    ✅ {chartjs}: {size:,} bytes ({size/1024:.1f} KB)")
else:
    print(f"    ❌ {chartjs} が見つかりません")
    print(f"    → start_demo.sh を再実行すると自動ダウンロードされます")

print("\n" + "=" * 60)
print("診断完了")
print("=" * 60)
