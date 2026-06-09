import os
import sys
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

# Stub lib2to3
import types
if 'lib2to3' not in sys.modules:
    lib2to3 = types.ModuleType('lib2to3')
    lib2to3.pytree = types.ModuleType('lib2to3.pytree')
    sys.modules['lib2to3'] = lib2to3
    sys.modules['lib2to3.pytree'] = lib2to3.pytree

import logging
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from feat import Detector
from scipy.spatial import distance

logger = logging.getLogger(__name__)

class FacialAnalyzer:
    def __init__(self):
        """
        Initialize Py-Feat Detector.
        We use 'retinaface' for detection, 'resnet' for AUs.
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Py-Feat Detector on {device}...")
        # SOTA Setup: RetinaFace for detection, ResNet-50 for AUs (Higher precision than SVM/RF)
        self.detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb", # High precision model available in this environment
            emotion_model="resmasknet",
            device=device
        )
        
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio."""
        # eye_points has shape (6, 2)
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        C = distance.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def process_face_crops(self, crop_dir: str) -> pd.DataFrame:
        """
        Process a directory of face crops to extract AUs and Blink info.
        Assumes filenames are formatted as 'face_{frame_id:06d}.jpg'.
        """
        image_paths = sorted(list(Path(crop_dir).glob("*.jpg")))
        if not image_paths:
            logger.warning(f"No images found in {crop_dir}")
            return pd.DataFrame()
            
        logger.info(f"Processing {len(image_paths)} face crops...")
        
        # Py-Feat batch processing
        # Note: detector.detect_image can take a list of filenames
        # For large lists, we should batch.
        # RTX 5080 (16GB) can handle large batches. Let's aim for 128-256.
        batch_size = 128
        all_results = []
        
        path_strs = [str(p) for p in image_paths]
        
        from tqdm import tqdm
        logger.info(f"Batch processing {len(path_strs)} images (batch_size={batch_size})...")
        
        # Iterate in batches
        for i in tqdm(range(0, len(path_strs), batch_size), desc="Facial AU Analysis"):
            batch_files = path_strs[i:i+batch_size]
            try:
                # Detect
                detected = self.detector.detect_image(batch_files)
                # Results is a DataFrame
                all_results.append(detected)
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue
                
        if not all_results:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Post-process: Extract useful columns and calculate EAR
        output_data = []
        
        # In Py-Feat, combined_df has an 'input' column containing the file path
        # or 'frame' / 'filename' depending on the version.
        
        for _, row in combined_df.iterrows():
            filepath = row.get("input", "")
            if not filepath:
                # Fallback: if 'input' is missing, skip row
                continue
                
            try:
                frame_id = int(Path(filepath).stem.split('_')[1])
            except (IndexError, ValueError):
                continue
            
            # Action Units: AU04 (Brow Lowerer), AU12 (Lip Corner Puller)
            au4 = row.get("AU04", np.nan)
            au12 = row.get("AU12", np.nan)
            
            # Blink Detection via EAR
            ear = 0.3 # Default open
            if 'landmarks' in row and row['landmarks'] is not None:
                try:
                    lms = np.array(row['landmarks'])
                    if lms.ndim == 2 and lms.shape[0] == 68:
                        left_eye = lms[36:42]
                        right_eye = lms[42:48]
                        ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0
                except:
                    pass
            
            output_data.append({
                "frame": frame_id,
                "AU04": au4,
                "AU12": au12,
                "EAR": ear,
                "is_blink": 1 if ear < 0.20 else 0
            })
            
        return pd.DataFrame(output_data).sort_values("frame")

if __name__ == "__main__":
    # Test stub
    pass
