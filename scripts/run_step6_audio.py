#!/usr/bin/env python
"""
run_step6_audio.py
==================
音声データと文字起こし結果（CSV）を読み込み、発話セグメント（時間枠）ごとに
音響特徴量（F0・ラウドネス・ジッター等）を OpenSMILE を用いて抽出します。

使い方:
  python scripts/run_step6_audio.py [--wav_path data/boj_conference.wav] [--transcription_path output/transcription.csv] [--output_path output/audio_features.csv]
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import soundfile as sf

# プロジェクトのルートディレクトリをシステムパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.audio_analysis import AudioAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Step6-AudioProsody")


def main():
    parser = argparse.ArgumentParser(description="Step 6: 音声特徴量抽出 (OpenSMILE)")
    parser.add_argument(
        "--wav_path",
        type=str,
        default="data/boj_conference.wav",
        help="入力音声ファイル(.wav)のパス",
    )
    parser.add_argument(
        "--transcription_path",
        type=str,
        default="output/transcription.csv",
        help="Step 1 で出力した文字起こしCSVのパス",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/audio_features.csv",
        help="音声特徴量CSVの保存パス",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav_path)
    trans_path = Path(args.transcription_path)
    output_path = Path(args.output_path)

    if not wav_path.exists():
        logger.error(f"入力音声ファイルが見つかりません: {wav_path}. 先に Step 1 を実行してください。")
        sys.exit(1)

    if not trans_path.exists():
        logger.error(f"文字起こしファイルが見つかりません: {trans_path}. 先に Step 1 を実行してください。")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 音声と文字起こしの読み込み
    logger.info(f"音声ファイルを読み込み中: {wav_path} ...")
    full_audio, sr = sf.read(str(wav_path))
    
    trans_df = pd.read_csv(trans_path)
    logger.info(f"文字起こしデータをロードしました。総セグメント数: {len(trans_df)}")

    # 2. 特徴量抽出の実行
    temp_dir = output_path.parent / "temp_audio_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)

    import concurrent.futures
    
    def slice_and_save(idx, start, end):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        if end_sample > start_sample:
            segment_audio = full_audio[start_sample:end_sample]
            temp_wav = temp_dir / f"temp_{idx}.wav"
            sf.write(str(temp_wav), segment_audio, sr)
            return idx, str(temp_wav)
        return idx, None

    logger.info("発話区間のスライシングと一時保存を並列実行中 (CPU並列最適化)...")
    temp_files = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(slice_and_save, idx, row['start'], row['end']): idx for idx, row in trans_df.iterrows()}
        for future in concurrent.futures.as_completed(futures):
            idx, path = future.result()
            if path:
                temp_files[idx] = path

    logger.info("発話区間ごとの音声特徴量抽出 (OpenSMILE & Wav2Vec2 GPU推論) を開始します...")
    analyzer = AudioAnalyzer()
    audio_features_list = []
    
    # GPUリソースとファイルの衝突を避けるため、推論はメインスレッドで順次実行
    for idx in sorted(temp_files.keys()):
        temp_wav_path = temp_files[idx]
        try:
            prosody = analyzer.extract_prosody(temp_wav_path)
            prosody['sentence_id'] = idx
            audio_features_list.append(prosody)
        except Exception as e:
            logger.error(f"Error processing segment {idx}: {e}")
        finally:
            Path(temp_wav_path).unlink(missing_ok=True)

    # クリーニング
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    audio_df = pd.DataFrame(audio_features_list)
    
    # 3. CSV保存
    audio_df.to_csv(output_path, index=False)
    logger.info(f"音声分析完了。結果保存先: {output_path} (行数: {len(audio_df)})")


if __name__ == "__main__":
    main()
