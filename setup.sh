#!/bin/bash
set -e

echo "=== BoJ Multimodal Analysis Pipeline Setup ==="
echo "Python環境の構築を開始します..."

# 1. Start Clean
if [ -d ".venv" ]; then
    echo "Cleaning up previous virtual environment..."
    rm -rf .venv
fi
echo "Creating virtual environment (.venv)..."
python3 -m venv .venv

# 2. Upgrade pip and build tools (using venv python)
echo "Upgrading pip, setuptools, and wheel..."
./.venv/bin/python -m pip install --upgrade pip setuptools wheel

# 2.5 Pre-install critical binary dependencies to avoid source builds
echo "Installing core binary dependencies (numpy, pandas, h5py)..."
./.venv/bin/pip install numpy pandas
./.venv/bin/pip install h5py --only-binary=:all:

# 3. Install Dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    ./.venv/bin/pip install --prefer-binary -r requirements.txt
else
    echo "Error: requirements.txt not found."
    exit 1
fi

# 4. Create Directory Structure
echo "Creating directory structure..."
mkdir -p output/faces
mkdir -p data

echo "=== Setup Complete ==="
echo "分析対象の動画ファイルは以下のディレクトリに配置してください:"
echo "$(pwd)/data/"
echo ""
echo "実行コマンド例:"
echo "  source .venv/bin/activate"
echo "  python main.py --video_path data/your_video.mp4"

