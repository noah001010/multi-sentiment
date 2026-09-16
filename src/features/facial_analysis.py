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

    def process_video(self, video_path: str, skip_frames: int = 1, fps: float = 30.0) -> pd.DataFrame:
        """
        Process a full video directly without face crop extraction.
        Extracts 7 basic emotion probabilities and computes face_negative_score for each frame.
        Includes 'frame', 'timestamp' (seconds), and 'time_str' (MM:SS).
        """
        logger.info(f"Processing full video directly: {video_path} (fps={fps}, skip_frames={skip_frames})...")
        
        try:
            # Py-Feat video detection
            if hasattr(self.detector, "detect_video"):
                detected = self.detector.detect_video(video_path, skip_frames=skip_frames)
            else:
                detected = self.detector.detect(video_path, skip_frames=skip_frames)
        except Exception as e:
            logger.error(f"Error executing Py-Feat detect_video on {video_path}: {e}")
            return pd.DataFrame()

        if detected is None or len(detected) == 0:
            logger.warning(f"No face detections in video: {video_path}")
            return pd.DataFrame()

        output_data = []
        for idx, row in detected.iterrows():
            frame_id = int(row.get("frame", idx * skip_frames))
            timestamp = frame_id / fps
            time_str = format_time_str(timestamp)

            anger = float(row.get("anger", row.get("angry", 0.0)))
            disgust = float(row.get("disgust", 0.0))
            fear = float(row.get("fear", 0.0))
            happiness = float(row.get("happiness", row.get("happy", 0.0)))
            sadness = float(row.get("sadness", row.get("sad", 0.0)))
            surprise = float(row.get("surprise", 0.0))
            neutral = float(row.get("neutral", 0.0))

            # Curti & Kazinnik (2023) 準拠: Negative Facial Score
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
