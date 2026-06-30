#!/usr/bin/env python
"""
run_step1_asr.py
================
動画ファイルから音声を抽出し、Whisperを用いて文字起こし（タイムスタンプ付き）を行います。

使い方:
  python scripts/run_step1_asr.py [--video_path data/boj_5min.mp4] [--output_path output/transcription.csv]
"""
import argparse
import logging
import os
import subprocess
import sys
import tempfile
import torch
from pathlib import Path

# プロジェクトのルートディレクトリをシステムパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.whisper_aligner import WhisperAligner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Step1-ASR")


def main():
    parser = argparse.ArgumentParser(description="Step 1: 音声認識 & 文字起こし (ASR)")
    parser.add_argument(
        "--video_path",
        type=str,
        default="data/boj_5min.mp4",
        help="入力動画ファイルのパス",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/transcription.csv",
        help="結果保存先CSVのパス",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="large-v3",
        help="Whisperモデルのサイズ (tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス (cuda, cpu)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="言語コード",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_path = Path(args.output_path)

    if not video_path.exists():
        logger.error(f"入力動画ファイルが見つかりません: {video_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 音声ファイル（wav）の準備
    permanent_wav = video_path.with_suffix(".wav")
    wav_to_process = None
    temp_wav = None

    if video_path.suffix.lower() == ".wav":
        logger.info(f"入力ファイルがすでにWav音声形式です: {video_path}")
        wav_to_process = video_path
    elif permanent_wav.exists():
        logger.info(f"同名の音声ファイルがすでに存在するため、ffmpegでの抽出をスキップします: {permanent_wav}")
        wav_to_process = permanent_wav
    else:
        # ffmpegを用いて動画から音声を抽出
        temp_wav = Path(tempfile.mktemp(suffix=".wav"))
        logger.info(f"動画から一時音声ファイルを作成中: {temp_wav}")
        try:
            # 16kHz モノラルに変換
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-q:a",
                    "0",
                    "-map",
                    "a",
                    str(temp_wav),
                    "-y",
                ],
                check=True,
                capture_output=True,
            )
            import shutil
            shutil.copy(temp_wav, permanent_wav)
            logger.info(f"音声波形ファイルをデータフォルダに保存しました: {permanent_wav}")
            wav_to_process = permanent_wav
        except FileNotFoundError:
            logger.error("エラー: システムに 'ffmpeg' がインストールされていないため、動画から音声を抽出できません。")
            logger.error("Mac の場合は 'brew install ffmpeg' を実行してインストールするか、すでに抽出済みの .wav ファイルを直接 --video_path に指定してください。")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpegによる音声抽出に失敗しました: {e.stderr.decode('utf-8')}")
            sys.exit(1)

    # 2. Whisperによる文字起こし実行
    try:
        aligner = WhisperAligner(
            model_size=args.model_size,
            device=args.device,
            compute_type="float16" if "cuda" in args.device else "int8",
        )
        results = aligner.transcribe(str(wav_to_process), language=args.language)
        aligner.save_results(results, str(output_path))
        logger.info(f"文字起こし完了。結果保存先: {output_path}")

    finally:
        # 一時ファイルのクリーンアップ
        if temp_wav and temp_wav.exists():
            temp_wav.unlink()


if __name__ == "__main__":
    main()
