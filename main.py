import argparse
import logging
import pandas as pd
import torch
import os
import sys

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
import torch.serialization as serial
serial.load = _patched_load

from pathlib import Path

# SOTA Compatibility Patches
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

from dotenv import load_dotenv
load_dotenv() # Load HF_TOKEN from .env

# Monkey-patch scipy.integrate.simps to simpson for compatibility with older Py-Feat
try:
    import scipy.integrate
    if not hasattr(scipy.integrate, 'simps'):
        scipy.integrate.simps = scipy.integrate.simpson
except ImportError:
    pass

# Stub lib2to3 for compatibility with older ResMaskNet in Py-Feat
import types
if 'lib2to3' not in sys.modules:
    lib2to3 = types.ModuleType('lib2to3')
    lib2to3.pytree = types.ModuleType('lib2to3.pytree')
    lib2to3.pytree.convert = lambda x: x
    sys.modules['lib2to3'] = lib2to3
    sys.modules['lib2to3.pytree'] = lib2to3.pytree

# ログ設定（日本語対応）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BoJ-Pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="日銀総裁会見 マルチモーダル分析パイプライン")
    parser.add_argument("--video_path", type=str, required=True, help="入力MP4動画ファイルのパス")
    parser.add_argument("--output_dir", type=str, default="output", help="結果出力ディレクトリ")
    parser.add_argument("--gpu_id", type=int, default=0, help="使用するGPU ID (Default: 0)")
    parser.add_argument(
        "--text_model_path",
        type=str,
        default=None,
        help="テキスト推論に使用するローカルHFモデルのディレクトリ (config.json が常驐するディレクトリ). "
        "未指定の場合はデフォルトのFin-BERTを使用。",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        logger.error(f"動画ファイルが見つかりません: {video_path}")
        sys.exit(1)

    logger.info(f"分析を開始します: {video_path.name}")
    logger.info(f"出力先: {output_dir}")

    # ---------------------------------------------------------
    # Step 1. 前処理 (Preprocessing)
    # ---------------------------------------------------------
    logger.info("--- Step 1: 前処理 (音声認識 & 話者特定) ---")
    
    # 1-1. ASR (Whisper-large-v3)
    from src.preprocessing.whisper_aligner import WhisperAligner
    
    transcription_path = output_dir / "transcription.csv"
    if transcription_path.exists():
        logger.info(f"既存の文字起こし結果が見見つかりました: {transcription_path}. ASRをスキップします。")
        asr_result = {"segments": pd.read_csv(transcription_path).to_dict('records')}
    else:
        device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper-large-v3 モデルをロード中 (Device: {device})...")
        
        # SOTA Parameter: large-v3, float16
        aligner = WhisperAligner(model_size="large-v3", device=device)
        asr_result = aligner.transcribe(str(video_path))
        aligner.save_results(asr_result, str(transcription_path))
    
    # 1-2. SOTA Diarization (pyannote.audio)
    diarization_out_path = output_dir / "raw" / "diarization.csv"
    if diarization_out_path.exists():
        logger.info(f"既存の話者分離結果が見つかりました: {diarization_out_path}. Diarizationをスキップします。")
        diarization_df = pd.read_csv(diarization_out_path)
    else:
        logger.info("pyannote.audio による SOTA 話者分離を実行中...")
        # Check for HF Token in environment or ask
        import os
        from pyannote.audio import Pipeline
        hf_token = os.getenv("HF_TOKEN")
        
        if hf_token:
            try:
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
                if torch.cuda.is_available():
                    pipeline.to(torch.device(f"cuda:{args.gpu_id}"))
                
                # Extract audio for diarization
                wav_path = Path(str(video_path).replace(".mp4", ".wav"))
                if not wav_path.exists():
                    logger.info("ffmpeg を使用して音声を抽出中...")
                    import subprocess
                    subprocess.run(["ffmpeg", "-i", str(video_path), "-ar", "16000", "-ac", "1", str(wav_path)], check=True)
                
                diarization = pipeline(str(wav_path))
                
                # Convert to DataFrame (Support both pyannote.audio <4.0 and 4.0+ DiarizeOutput)
                diar_data = []
                if hasattr(diarization, "itertracks"):
                    # Standard pyannote.audio Annotation
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        diar_data.append({"start": turn.start, "end": turn.end, "speaker": speaker})
                elif hasattr(diarization, "speaker_diarization"):
                    # pyannote.audio 4.0+ DiarizeOutput
                    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                        diar_data.append({"start": turn.start, "end": turn.end, "speaker": speaker})
                elif hasattr(diarization, "turns"):
                    # pyannoteai-sdk DiarizeOutput fallback
                    for turn in diarization.turns:
                        diar_data.append({"start": turn.start, "end": turn.end, "speaker": turn.speaker})
                else:
                    logger.warning(f"Unexpected diarization result type: {type(diarization)}. Trying to iterate turns...")
                    try:
                        for turn in diarization:
                            diar_data.append({"start": turn.start, "end": turn.end, "speaker": turn.speaker})
                    except:
                        raise ValueError(f"Could not parse diarization output of type {type(diarization)}")

                diarization_df = pd.DataFrame(diar_data)
                diarization_df.to_csv(diarization_out_path, index=False)
                logger.info(f"Diarization 完了. 結果を保存しました: {diarization_out_path}")
            except Exception as e:
                logger.error(f"Diarization 実行エラー (Token未設定の可能性があります): {e}")
                logger.warning("Mock方式にフォールバックします...")
                diarization_df = pd.DataFrame([{"start": 0, "end": 3600, "speaker": "SPEAKER_00"}]) # Dummy
        else:
            logger.warning("HF_TOKEN が設定されていないため、SOTA Diarization をスキップし Mock 方式を使用します。")
            diarization_df = pd.DataFrame([{"start": 0, "end": 3600, "speaker": "SPEAKER_00"}])
    
    # Step 1.3: Face Extraction (Now handled inside the chunked loop for stability)
    from src.preprocessing.asd_pipeline import VisualASD
    asd = VisualASD()
    
    # ---------------------------------------------------------
    # Step 2. 特徴量抽出 (Feature Extraction)
    # ---------------------------------------------------------
    logger.info("--- Step 2: マルチモーダル特徴量の抽出 ---")
    
    # 2-1. テキスト分析 (BERT)
    text_feat_path = output_dir / "text_features.csv"
    if text_feat_path.exists():
        logger.info(f"既存のテキスト分析結果が見つかりました: {text_feat_path}. BERTをスキップします。")
        text_features_df = pd.read_csv(text_feat_path)
    else:
        from src.features.text_analysis import TextAnalyzer
        logger.info("BERTモデルによるテキスト分析（感情・不確実性）を実行中...")
        text_analyzer = TextAnalyzer(model_path=getattr(args, 'text_model_path', None))
        
        # ASR結果からテキストリストを作成
        texts = [s['text'] for s in asr_result['segments']]
        text_features_df = text_analyzer.analyze_texts(texts)
        
        # タイムスタンプ情報を結合
        text_features_df['start'] = [s['start'] for s in asr_result['segments']]
        text_features_df['end'] = [s['end'] for s in asr_result['segments']]
        text_features_df.to_csv(text_feat_path, index=False)
    
    # Export Clean Transcription
    trans_clean_path = output_dir / "transcription_clean.csv"
    text_features_df.to_csv(trans_clean_path, index=False)
    logger.info(f"Cleaned transcription saved to {trans_clean_path}")
    
    # 2-2. 画像（表情）分析 (Py-Feat) - CHUNKED for Stability
    visual_feat_path = output_dir / "visual_features.csv"
    if visual_feat_path.exists():
        logger.info(f"既存の表情分析結果が見つかりました: {visual_feat_path}. Py-Featをスキップします。")
        visual_df = pd.read_csv(visual_feat_path)
    else:
        from src.features.facial_analysis import FacialAnalyzer
        # 10分ごとのチャタリング
        chunk_size_sec = 600 # 10 minutes
        total_duration = diarization_df['end'].max()
        logger.info(f"表情分析を {chunk_size_sec}秒ごとのチャンクに分割して実行します (Total: {total_duration:.1f}s)")
        
        all_visual_chunks = []
        for start_t in range(0, int(total_duration) + 1, chunk_size_sec):
            end_t = min(start_t + chunk_size_sec, total_duration)
            logger.info(f">>> チャンク開始: {start_t}s - {end_t}s")
            
            chunk_diar = diarization_df[(diarization_df['start'] < end_t) & (diarization_df['end'] > start_t)].copy()
            if chunk_diar.empty:
                continue
                
            # Adjust start/end to chunk boundaries
            chunk_diar['start'] = chunk_diar['start'].clip(lower=start_t)
            chunk_diar['end'] = chunk_diar['end'].clip(upper=end_t)
            
            chunk_faces_dir = output_dir / "faces" / f"chunk_{start_t}"
            chunk_faces_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract faces for this chunk
            asd.extract_active_face_crops(str(video_path), chunk_diar, str(chunk_faces_dir))
            
            # Analyze
            try:
                if list(chunk_faces_dir.glob("*.jpg")):
                    face_analyzer = FacialAnalyzer()
                    chunk_visual_df = face_analyzer.process_face_crops(str(chunk_faces_dir))
                    
                    if not chunk_visual_df.empty:
                        # Temporary save
                        chunk_out = output_dir / "raw" / f"visual_features_{start_t}.csv"
                        chunk_visual_df.to_csv(chunk_out, index=False)
                        all_visual_chunks.append(chunk_visual_df)
                        logger.info(f"チャンク終了: {start_t}s まで完了。中間ファイルを保存しました。")
                    
                    # Force Memory cleanup
                    del face_analyzer
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    logger.warning(f"チャンク {start_t}s: 顔画像が見つかりませんでした。")
            except Exception as e:
                logger.error(f"チャンク {start_t}s でエラーが発生しました: {e}")
                
        if all_visual_chunks:
            visual_df = pd.concat(all_visual_chunks, ignore_index=True).sort_values("frame")
            visual_df.to_csv(visual_feat_path, index=False)
            logger.info(f"全チャンクの統合完了: {visual_feat_path}")
        else:
            visual_df = pd.DataFrame()
            logger.warning("表情分析結果が空です。")

    # Export Clean Facial Features
    if not visual_df.empty:
        visual_clean_path = output_dir / "facial_features_clean.csv"
        # Map AU codes to human-readable names if needed, but keeping for now for precision
        visual_df.to_csv(visual_clean_path, index=False)
        logger.info(f"Cleaned facial features saved to {visual_clean_path}")

    # 2-3. 音声分析 (OpenSMILE)
    audio_feat_path = output_dir / "audio_features.csv"
    if audio_feat_path.exists():
        logger.info(f"既存の音声分析結果が見つかりました: {audio_feat_path}. OpenSMILEをスキップします。")
        audio_df = pd.read_csv(audio_feat_path)
    else:
        from src.features.audio_analysis import AudioAnalyzer
        logger.info("OpenSMILEによる音声プロソディ分析を実行中...")
        audio_analyzer = AudioAnalyzer()
        
        import soundfile as sf
        wav_path = Path(str(video_path).replace(".mp4", ".wav"))
        if not wav_path.exists():
             import subprocess
             logger.info(f"Extracting temporary wav for audio analysis: {wav_path}")
             subprocess.run(["ffmpeg", "-i", str(video_path), "-ar", "16000", "-ac", "1", str(wav_path)], check=True)
        
        audio_input_path = wav_path
        # 音声読み込み（全体）
        logger.info(f"音声ファイルを読み込み中: {audio_input_path}")
        full_audio, sr = sf.read(str(audio_input_path))
        
        audio_features_list = []
        # 各発話セグメントごとに音声を切り出して分析
        for idx, row in text_features_df.iterrows():
            start_sample = int(row['start'] * sr)
            end_sample = int(row['end'] * sr)
            
            # サンプル数チェック
            if end_sample > start_sample:
                segment_audio = full_audio[start_sample:end_sample]
                temp_wav = output_dir / f"temp_{idx}.wav"
                sf.write(str(temp_wav), segment_audio, sr)
                
                # 特徴量抽出
                prosody = audio_analyzer.extract_prosody(str(temp_wav))
                prosody['sentence_id'] = idx
                audio_features_list.append(prosody)
                
                # 一時ファイル削除
                temp_wav.unlink(missing_ok=True)
                
        audio_df = pd.DataFrame(audio_features_list)
        
        # Raw Logging
        raw_audio_path = output_dir / "raw" / "raw_audio_prosody.csv"
        audio_df.to_csv(raw_audio_path, index=False)
        logger.info(f"Raw audio prosody logged to {raw_audio_path}")
        
        audio_df.to_csv(audio_feat_path, index=False)

    # Export Clean Audio Features
    if not audio_df.empty:
        audio_clean_path = output_dir / "audio_features_clean.csv"
        audio_df.to_csv(audio_clean_path, index=False)
        logger.info(f"Cleaned audio features saved to {audio_clean_path}")
    
    # ---------------------------------------------------------
    # Step 3. データ統合 (Integration)
    # ---------------------------------------------------------
    logger.info("--- Step 3: データ統合 ---")
    from src.analysis.integrator import MultimodalIntegrator
    integrator = MultimodalIntegrator()
    
    # 各データフレームを結合
    # axis=1での結合はインデックスが揃っている前提（textとaudioはsentence_idで同期）
    text_audio = text_features_df.copy()
    if not audio_df.empty:
        # sentence_id順になっていると仮定
        text_audio = pd.concat([text_audio, audio_df], axis=1)
    
    # 統合処理（Visualはタイムスタンプベースでマージ, AudioはSentenceベース, DiarizationはOverlapベース）
    final_df = integrator.align_and_merge(text_audio, visual_df, audio_df, diarization_df=diarization_df)
    
    output_path = output_dir / "integrated_results.csv"
    final_df.to_csv(output_path, index=False)
    
    logger.info("==========================================")
    logger.info(f"パイプライン完了。統合データ: {output_path}")
    logger.info("可視化コマンド: streamlit run src/analysis/visualization.py")
    logger.info("==========================================")

if __name__ == "__main__":
    main()
