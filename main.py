import argparse
import logging
import pandas as pd
import torch
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Torch compatibility patches
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
import torch.serialization as serial
serial.load = _patched_load

import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

from dotenv import load_dotenv
load_dotenv()

# Compatibility patches
try:
    import scipy.integrate
    if not hasattr(scipy.integrate, 'simps'):
        scipy.integrate.simps = scipy.integrate.simpson
except ImportError:
    pass

import types
if 'lib2to3' not in sys.modules:
    lib2to3 = types.ModuleType('lib2to3')
    lib2to3.pytree = types.ModuleType('lib2to3.pytree')
    lib2to3.pytree.convert = lambda x: x
    sys.modules['lib2to3'] = lib2to3
    sys.modules['lib2to3.pytree'] = lib2to3.pytree

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BoJ-Pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="日銀総裁会見 マルチモーダル分析パイプライン (学術標準版)")
    parser.add_argument("--video_path", type=str, default="data/boj_conference.mp4", help="入力MP4動画ファイルのパス")
    parser.add_argument("--date_code", type=str, default="23_0616", help="会見日付コード (例: 23_0616)")
    parser.add_argument("--output_root", type=str, default="output", help="成果物出力ルートディレクトリ")
    parser.add_argument("--gpu_id", type=int, default=0, help="使用するGPU ID (Default: 0)")
    parser.add_argument(
        "--text_model_path",
        type=str,
        default=None,
        help="テキスト推論に使用するローカルHFモデルのディレクトリ (未指定時はデフォルトのFin-BERT/ModernBERT)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    video_path = Path(args.video_path)
    output_root = Path(args.output_root)
    
    # 会見日付ごとのディレクトリ構成
    target_dir = output_root / args.date_code
    face_dir = target_dir / "face"
    audio_dir = target_dir / "audio"
    text_dir = target_dir / "text"

    for d in [face_dir, audio_dir, text_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        sys.exit(1)

    logger.info(f"=== 分析パイプライン開始: {video_path.name} (保存先: {target_dir}) ===")

    # ---------------------------------------------------------
    # Step 1. 前処理 (ASR & Diarization)
    # ---------------------------------------------------------
    logger.info("--- Step 1: 前処理 (音声文字起こし & 話者特定) ---")
    
    # 1-1. ASR (Whisper)
    from src.preprocessing.whisper_aligner import WhisperAligner
    transcription_path = text_dir / "transcription.csv"
    
    if transcription_path.exists():
        logger.info(f"既存の文字起こし結果を使用します: {transcription_path}")
        asr_result = {"segments": pd.read_csv(transcription_path).to_dict('records')}
    else:
        device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper-large-v3 による文字起こしを実行中 (Device: {device})...")
        aligner = WhisperAligner(model_size="large-v3", device=device)
        asr_result = aligner.transcribe(str(video_path))
        aligner.save_results(asr_result, str(transcription_path))
    
    # 1-2. Diarization
    diarization_path = text_dir / "diarization.csv"
    if diarization_path.exists():
        logger.info(f"既存の話者分離結果を使用します: {diarization_path}")
        diarization_df = pd.read_csv(diarization_path)
    else:
        logger.info("pyannote.audio による話者分離を実行中...")
        import os
        from pyannote.audio import Pipeline
        hf_token = os.getenv("HF_TOKEN")
        
        if hf_token:
            try:
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
                if torch.cuda.is_available():
                    pipeline.to(torch.device(f"cuda:{args.gpu_id}"))
                
                wav_path = Path(str(video_path).replace(".mp4", ".wav"))
                if not wav_path.exists():
                    import subprocess
                    subprocess.run(["ffmpeg", "-i", str(video_path), "-ar", "16000", "-ac", "1", str(wav_path)], check=True)
                
                diarization = pipeline(str(wav_path))
                diar_data = []
                if hasattr(diarization, "itertracks"):
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        diar_data.append({"start": turn.start, "end": turn.end, "speaker": speaker})
                elif hasattr(diarization, "speaker_diarization"):
                    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                        diar_data.append({"start": turn.start, "end": turn.end, "speaker": speaker})
                else:
                    for turn in diarization:
                        diar_data.append({"start": turn.start, "end": turn.end, "speaker": getattr(turn, 'speaker', 'SPEAKER_00')})

                diarization_df = pd.DataFrame(diar_data)
                diarization_df.to_csv(diarization_path, index=False)
            except Exception as e:
                logger.warning(f"Diarization スキップ (Mock): {e}")
                diarization_df = pd.DataFrame([{"start": 0, "end": 3600, "speaker": "SPEAKER_00"}])
        else:
            diarization_df = pd.DataFrame([{"start": 0, "end": 3600, "speaker": "SPEAKER_00"}])
            diarization_df.to_csv(diarization_path, index=False)

    # ---------------------------------------------------------
    # Step 2. モダリティ別特徴量抽出
    # ---------------------------------------------------------
    logger.info("--- Step 2: マルチモーダル特徴量の抽出 ---")

    # 2-1. テキスト感情推論 (FinBERT)
    text_feat_path = text_dir / "text_features.csv"
    if text_feat_path.exists():
        logger.info(f"既存のテキスト感情データを使用します: {text_feat_path}")
        text_features_df = pd.read_csv(text_feat_path)
    else:
        from src.features.text_analysis import TextAnalyzer
        logger.info("FinBERT/ModernBERT モデルによるテキスト感情推論中...")
        text_analyzer = TextAnalyzer(model_path=getattr(args, 'text_model_path', None))
        texts = [s['text'] for s in asr_result['segments']]
        text_features_df = text_analyzer.analyze_texts(texts)
        text_features_df['start'] = [s['start'] for s in asr_result['segments']]
        text_features_df['end'] = [s['end'] for s in asr_result['segments']]
        text_features_df.to_csv(text_feat_path, index=False)

    # 2-2. 表情感情解析 (Py-Feat 動画全体連続解析)
    facial_feat_path = face_dir / "facial_features.csv"
    if facial_feat_path.exists():
        logger.info(f"既存の全フレーム表情データを使用します: {facial_feat_path}")
        visual_df = pd.read_csv(facial_feat_path)
    else:
        from src.features.facial_analysis import FacialAnalyzer
        logger.info("Py-Feat による動画全編の直接表情解析（全フレーム連続）を開始します...")
        face_analyzer = FacialAnalyzer()
        visual_df = face_analyzer.process_video(str(video_path))
        if not visual_df.empty:
            visual_df.to_csv(facial_feat_path, index=False)
            logger.info(f"表情解析完了: {facial_feat_path} (全 {len(visual_df)} フレーム)")
        else:
            logger.warning("表情解析データが取得できませんでした。")

    # 2-3. 音声感情解析 (Wav2Vec2: audio_arousal, audio_valence)
    audio_feat_path = audio_dir / "audio_features.csv"
    if audio_feat_path.exists():
        logger.info(f"既存の音声感情データを使用します: {audio_feat_path}")
        audio_df = pd.read_csv(audio_feat_path)
    else:
        from src.features.audio_analysis import AudioAnalyzer
        logger.info("Wav2Vec2 による音声感情解析 (audio_arousal, audio_valence) を実行中...")
        audio_analyzer = AudioAnalyzer()
        
        import soundfile as sf
        wav_path = Path(str(video_path).replace(".mp4", ".wav"))
        if not wav_path.exists():
            import subprocess
            subprocess.run(["ffmpeg", "-i", str(video_path), "-ar", "16000", "-ac", "1", str(wav_path)], check=True)
            
        full_audio, sr = sf.read(str(wav_path))
        audio_features_list = []
        
        # 一時ファイル用の一時ディレクトリ
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for idx, row in text_features_df.iterrows():
                start_sample = int(row['start'] * sr)
                end_sample = int(row['end'] * sr)
                
                if end_sample > start_sample:
                    segment_audio = full_audio[start_sample:end_sample]
                    temp_wav = tmp_path / f"temp_{idx}.wav"
                    sf.write(str(temp_wav), segment_audio, sr)
                    
                    prosody = audio_analyzer.extract_prosody(str(temp_wav))
                    prosody['sentence_id'] = idx
                    audio_features_list.append(prosody)
                    
        audio_df = pd.DataFrame(audio_features_list)
        audio_df.to_csv(audio_feat_path, index=False)
        logger.info(f"音声感情解析完了: {audio_feat_path}")

    # ---------------------------------------------------------
    # Step 3. データ統合 (Integration & Alignment)
    # ---------------------------------------------------------
    logger.info("--- Step 3: データ統合とアライメント ---")
    from src.analysis.integrator import MultimodalIntegrator
    integrator = MultimodalIntegrator()
    
    text_audio = text_features_df.copy()
    if not audio_df.empty:
        text_audio = pd.merge(text_audio, audio_df, left_index=True, right_on="sentence_id", how="left")
    
    final_df = integrator.align_and_merge(text_audio, visual_df, audio_df, diarization_df=diarization_df)
    
    output_path = target_dir / "integrated_results.csv"
    final_df.to_csv(output_path, index=False)
    
    logger.info("==========================================")
    logger.info(f"全パイプライン処理完了！")
    logger.info(f"保存フォルダ: {target_dir}")
    logger.info(f"  ├── face/facial_features.csv")
    logger.info(f"  ├── audio/audio_features.csv")
    logger.info(f"  ├── text/text_features.csv")
    logger.info(f"  └── integrated_results.csv")
    logger.info("==========================================")

if __name__ == "__main__":
    main()
