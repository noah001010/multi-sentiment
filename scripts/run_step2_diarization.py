#!/usr/bin/env python
"""
run_step2_diarization.py
========================
動画から音声を抽出し、pyannote.audio を使用して話者分離（Diarization）を行います。
Hugging Face の認証トークンが必要です。

使い方:
  python scripts/run_step2_diarization.py [--video_path data/boj_5min.mp4] [--output_path output/raw/diarization.csv]
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
import torch
import pandas as pd
from dotenv import load_dotenv

# プロジェクトのルートディレクトリをシステムパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# EXTREMELY AGGRESSIVE PATCH FOR TORCH 2.6+ (SOTA Stability)
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
if hasattr(torch.serialization, 'add_safe_globals'):
    import torch.torch_version
    safe_types = [torch.torch_version.TorchVersion]
    try:
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        safe_types.extend([Specifications, Problem, Resolution])
    except ImportError:
        pass
    torch.serialization.add_safe_globals(safe_types)

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Step2-Diarization")


def main():
    parser = argparse.ArgumentParser(description="Step 2: 話者特定 (Diarization)")
    parser.add_argument(
        "--video_path",
        type=str,
        default="data/boj_5min.mp4",
        help="入力動画ファイルのパス",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/raw/diarization.csv",
        help="話者分離結果のCSV保存パス",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス (cuda, cpu)",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_path = Path(args.output_path)

    if not video_path.exists():
        logger.error(f"入力動画ファイルが見つかりません: {video_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 音声ファイル（wav）の準備
    wav_path = video_path.with_suffix(".wav")
    if not wav_path.exists():
        logger.info(f"一時音声ファイルを抽出中: {wav_path}")
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(video_path), "-ar", "16000", "-ac", "1", str(wav_path), "-y"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpegによる音声抽出に失敗しました: {e.stderr.decode('utf-8')}")
            sys.exit(1)

    # 2. Pyannote話者分離パイプラインの実行
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "PASTE_YOUR_TOKEN_HERE":
        logger.error(
            "HF_TOKEN が設定されていません。pyannote/speaker-diarization-3.1 の利用にはトークンが必要です。\n"
            ".env ファイルに HF_TOKEN=<your_token> を設定してください。"
        )
        sys.exit(1)

    logger.info("pyannote.audio から SOTA 話者分離モデルをロード中...")
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        if "cuda" in args.device:
            pipeline.to(torch.device(args.device))

        logger.info("音声解析を実行中...")
        diarization = pipeline(str(wav_path))

        # 結果をリストに変換
        diar_data = []
        if hasattr(diarization, "itertracks"):
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diar_data.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        elif hasattr(diarization, "speaker_diarization"):
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                diar_data.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        else:
            for turn in diarization:
                diar_data.append({"start": turn.start, "end": turn.end, "speaker": turn.speaker})

        diar_df = pd.DataFrame(diar_data)
        diar_df.to_csv(output_path, index=False)
        logger.info(f"話者分離完了。結果を保存しました: {output_path}")

    except Exception as e:
        logger.error(f"話者分離の実行中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
