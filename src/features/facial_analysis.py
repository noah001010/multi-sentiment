import os
import sys
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import scipy.stats
if not hasattr(scipy.stats, 'binom_test'):
    scipy.stats.binom_test = scipy.stats.binomtest

import numpy as np
if not hasattr(np, 'ComplexWarning'):
    class ComplexWarning(Warning):
        pass
    np.ComplexWarning = ComplexWarning

if not hasattr(np, 'mat'):
    np.mat = np.asmatrix

import torchvision.io
if not hasattr(torchvision.io, 'read_video'):
    torchvision.io.read_video = lambda *args, **kwargs: None

# Stub lib2to3
import types
if 'lib2to3' not in sys.modules:
    lib2to3 = types.ModuleType('lib2to3')
    lib2to3.pytree = types.ModuleType('lib2to3.pytree')
    lib2to3.pytree.convert = lambda x: x
    sys.modules['lib2to3'] = lib2to3
    sys.modules['lib2to3.pytree'] = lib2to3.pytree

# Stub torchcodec to prevent import failure due to ffmpeg/torchcodec compiled binary mismatch
if 'torchcodec' not in sys.modules:
    class MockVideoDecoder:
        def __init__(self, *args, **kwargs):
            pass
    torchcodec = types.ModuleType('torchcodec')
    torchcodec.decoders = types.ModuleType('torchcodec.decoders')
    torchcodec.decoders.VideoDecoder = MockVideoDecoder
    sys.modules['torchcodec'] = torchcodec
    sys.modules['torchcodec.decoders'] = torchcodec.decoders

import logging
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
try:
    from feat import Detector
except ImportError:
    from feat import Detectorv1 as Detector

def check_cuda_working() -> bool:
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(1, 1).to("cuda")
        y = torch.nn.functional.linear(x, x)
        return True
    except Exception as e:
        logger.warning(f"CUDA is available in PyTorch, but kernel execution failed: {e}. Falling back to CPU for safety.")
        return False

logger = logging.getLogger(__name__)

def format_time_str(seconds: float) -> str:
    """seconds (float) を 'MM:SS' フォーマットに変換する"""
    sec = max(0, int(seconds))
    mins = sec // 60
    secs = sec % 60
    return f"{mins:02d}:{secs:02d}"


import tempfile

class FacialAnalyzer:
    def __init__(self):
        """
        Initialize Py-Feat Detector.
        Uses 'retinaface' for detection, 'resmasknet' for basic emotions.
        """
        device = "cuda" if check_cuda_working() else "cpu"
        logger.info(f"Initializing Py-Feat Detector on {device}...")
        self.detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            emotion_model="resmasknet",
            device=device
        )

    def _run_detection(self, batch_files: list) -> pd.DataFrame:
        import torch
        with torch.no_grad():
            try:
                if hasattr(self.detector, "detect_image"):
                    return self.detector.detect_image(batch_files, batch_size=len(batch_files), progress_bar=False)
                elif hasattr(self.detector, "detect"):
                    return self.detector.detect(batch_files, batch_size=len(batch_files), progress_bar=False)
                else:
                    return self.detector.detect_video(batch_files[0])
            except TypeError:
                if hasattr(self.detector, "detect_image"):
                    return self.detector.detect_image(batch_files, batch_size=len(batch_files))
                elif hasattr(self.detector, "detect"):
                    return self.detector.detect(batch_files, batch_size=len(batch_files))
                else:
                    return self.detector.detect_video(batch_files[0])

    def process_video(self, video_path: str, skip_frames: int = 1, fps: float = 30.0, batch_size: int = 32) -> pd.DataFrame:
        """
        Process a full video directly using OpenCV frame extraction and Py-Feat detection.
        Extracts 7 basic emotion probabilities and computes face_negative_score for each frame.
        Includes 'frame', 'timestamp' (seconds), and 'time_str' (MM:SS).
        """
        import torch
        logger.info(f"Processing full video via OpenCV frame extraction: {video_path} (fps={fps}, skip_frames={skip_frames}, batch_size={batch_size})...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file with OpenCV: {video_path}")
            return pd.DataFrame()

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps and video_fps > 0:
            fps = float(video_fps)

        output_data = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frame_count = 0
            saved_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames == 0:
                    img_name = f"frame_{frame_count:07d}.jpg"
                    img_file = tmp_path / img_name
                    cv2.imwrite(str(img_file), frame)
                    saved_frames.append((frame_count, str(img_file)))
                
                frame_count += 1

            cap.release()

            if not saved_frames:
                logger.warning(f"No frames extracted from video: {video_path}")
                return pd.DataFrame()

            total_frames = len(saved_frames)
            logger.info(f"Extracted {total_frames} frames for facial analysis. Running Py-Feat Detector...")
            last_logged_pct = -1

            for i in range(0, total_frames, batch_size):
                batch = saved_frames[i:i + batch_size]
                batch_files = [f[1] for f in batch]
                frame_ids = [f[0] for f in batch]

                processed = i + len(batch)
                pct = int((processed / total_frames) * 100)
                if pct % 5 == 0 and pct != last_logged_pct:
                    logger.info(f"表情解析進捗: {processed} / {total_frames} フレーム完了 ({pct}%)")
                    last_logged_pct = pct

                try:
                    detected = self._run_detection(batch_files)
                except Exception as e:
                    logger.warning(f"Py-Feat batch detection warning at index {i} ({e}). Clearing GPU cache & retrying in smaller sub-batches...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    sub_results = []
                    sub_size = 8
                    for j in range(0, len(batch_files), sub_size):
                        sub_files = batch_files[j:j + sub_size]
                        try:
                            sub_det = self._run_detection(sub_files)
                            if sub_det is not None and len(sub_det) > 0:
                                sub_results.append(sub_det)
                        except Exception as sub_e:
                            logger.error(f"Sub-batch detection error: {sub_e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    detected = pd.concat(sub_results, ignore_index=True) if sub_results else None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if detected is None or len(detected) == 0:
                    continue

                for row_idx, row in detected.iterrows():
                    if row_idx < len(frame_ids):
                        f_id = frame_ids[row_idx]
                    else:
                        f_id = frame_ids[-1]

                    timestamp = f_id / fps
                    time_str = format_time_str(timestamp)

                    anger = float(row.get("anger", row.get("angry", 0.0)))
                    disgust = float(row.get("disgust", 0.0))
                    fear = float(row.get("fear", 0.0))
                    happiness = float(row.get("happiness", row.get("happy", 0.0)))
                    sadness = float(row.get("sadness", row.get("sad", 0.0)))
                    surprise = float(row.get("surprise", 0.0))
                    neutral = float(row.get("neutral", 0.0))

                    face_negative_score = anger + disgust + fear

                    output_data.append({
                        "frame": f_id,
                        "timestamp": round(timestamp, 2),
                        "time_str": time_str,
                        "anger": anger,
                        "disgust": disgust,
                        "fear": fear,
                        "happiness": happiness,
                        "sadness": sadness,
                        "surprise": surprise,
                        "neutral": neutral,
                        "face_negative_score": face_negative_score
                    })

        if not output_data:
            logger.warning("No face detections obtained from Py-Feat.")
            return pd.DataFrame()

        df_out = pd.DataFrame(output_data).sort_values("frame")
        return df_out

    def process_face_crops(self, crop_dir: str, batch_size: int = 256, fps: float = 30.0) -> pd.DataFrame:
        """
        Backward compatible crop folder processing with timestamp and time_str.
        """
        image_paths = sorted(list(Path(crop_dir).glob("*.jpg")))
        if not image_paths:
            logger.warning(f"No images found in {crop_dir}")
            return pd.DataFrame()
            
        logger.info(f"Processing {len(image_paths)} face crops...")
        path_strs = [str(p) for p in image_paths]
        
        all_results = []
        for i in range(0, len(path_strs), batch_size):
            batch_files = path_strs[i:i+batch_size]
            try:
                if hasattr(self.detector, "detect"):
                    detected = self.detector.detect(batch_files, batch_size=batch_size, output_size=(224, 224), progress_bar=False)
                else:
                    detected = self.detector.detect_image(batch_files, batch_size=batch_size)
                all_results.append(detected)
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue
                
        if not all_results:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_results, ignore_index=True)
        output_data = []
        
        for idx, row in combined_df.iterrows():
            filepath = row.get("input", "")
            try:
                frame_id = int(Path(filepath).stem.split('_')[1])
            except (IndexError, ValueError):
                frame_id = idx

            timestamp = frame_id / fps
            time_str = format_time_str(timestamp)

            anger = float(row.get("anger", row.get("angry", 0.0)))
            disgust = float(row.get("disgust", 0.0))
            fear = float(row.get("fear", 0.0))
            happiness = float(row.get("happiness", row.get("happy", 0.0)))
            sadness = float(row.get("sadness", row.get("sad", 0.0)))
            surprise = float(row.get("surprise", 0.0))
            neutral = float(row.get("neutral", 0.0))
            
            face_negative_score = anger + disgust + fear
            
            output_data.append({
                "frame": frame_id,
                "timestamp": round(timestamp, 2),
                "time_str": time_str,
                "anger": anger,
                "disgust": disgust,
                "fear": fear,
                "happiness": happiness,
                "sadness": sadness,
                "surprise": surprise,
                "neutral": neutral,
                "face_negative_score": face_negative_score
            })
            
        return pd.DataFrame(output_data).sort_values("frame")


if __name__ == "__main__":
    pass
