import logging
import cv2
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class VisualASD:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Visual ASD module (e.g., TalkNet-ASD).
        Args:
            model_path: Path to the pretrained TalkNet model weights.
        """
        self.model_path = model_path
        # In a real implementation, load TalkNet model here
        # self.model = TalkNet(...) 
        logger.info("Initialized VisualASD (Mock/Placeholder for TalkNet)")

    def detect_active_speaker(self, video_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run Active Speaker Detection on the video.
        
        Args:
            video_path: Path to the input video.
            output_path: Path to save the ASD result CSV.
        
        Returns:
            pd.DataFrame: DataFrame containing [frame, timestamp, face_id, score, is_speaking, bbox].
        """
        logger.info(f"Running Visual ASD on {video_path}...")
        
        # ---------------------------------------------------------------------
        # TODO: INTEGRATE TALKNET HERE
        # 1. Detect faces in every frame (using RetinaFace or MediaPipe)
        # 2. Extract face crops
        # 3. Feed audio + face crops sequence to TalkNet
        # 4. Get 'is_speaking' probability
        # ---------------------------------------------------------------------
        
        logger.warning("TalkNet is not fully implemented in this generated code due to dependency complexity.")
        logger.warning("Returning dummy data for pipeline verification.")

        # Dummy implementation for verification
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        # Mock: Assume Speaker 0 is active for the first half, then silent
        data = []
        for i in range(0, frame_count, int(fps)): # Sample once per second
            timestamp = i / fps
            is_speaking = 1 if timestamp < (duration / 2) else 0
            data.append({
                "timestamp": timestamp,
                "frame": i,
                "face_id": 0,
                "score": 0.95 if is_speaking else 0.1,
                "is_speaking": is_speaking,
                "bbox": [100, 100, 200, 200] # Dummy bbox
            })
        
        df = pd.DataFrame(data)
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved ASD results to {output_path}")

        return df

    def extract_active_face_crops(self, video_path: str, asd_df: pd.DataFrame, output_dir: str):
        """
        Extract face images for frames where the target speaker is active.
        This is used for subsequent Py-Feat analysis.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        # Support both frame-based ASD and segment-based Diarization
        if 'is_speaking' in asd_df.columns:
            active_units = asd_df[asd_df['is_speaking'] == 1]
        elif 'start' in asd_df.columns and 'end' in asd_df.columns:
            # For Diarization, we sample 1 frame per second within each segment
            active_units = []
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            for _, row in asd_df.iterrows():
                # For BoJ Governor analysis, we might want to filter for SPEAKER_00 etc.
                # Here we just take all indicated turns as "active" for cropping
                s, e = row['start'], row['end']
                for t in range(int(s), int(e) + 1):
                    active_units.append({
                        "frame": int(t * fps),
                        "bbox": [100, 100, 400, 400] # Default center-ish crop if bbox missing
                    })
            active_units = pd.DataFrame(active_units)
        else:
            logger.warning("Unknown dataframe format for face extraction. Skipping.")
            cap.release()
            return

        # Initialize a basic face detector for better cropping if bbox is missing
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for _, row in active_units.iterrows():
            frame_idx = int(row['frame'])
            if frame_idx >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                bbox = row.get('bbox', None)
                if bbox is None or bbox == [100, 100, 400, 400]:
                    # Try to detect face in the frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        # Take the largest face
                        (fx, fy, fw, fh) = max(faces, key=lambda f: f[2]*f[3])
                        # Pad a bit
                        pad = int(fw * 0.2)
                        bbox = [fx-pad, fy-pad, fx+fw+pad, fy+fh+pad]
                    else:
                        # Fallback to center of frame
                        h, w, _ = frame.shape
                        cw, ch = w // 2, h // 2
                        bbox = [cw-200, ch-200, cw+200, ch+200]

                if isinstance(bbox, str):
                    import ast
                    bbox = ast.literal_eval(bbox)
                x1, y1, x2, y2 = bbox
                # Ensure valid bbox
                h, w, _ = frame.shape
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    cv2.imwrite(str(output_dir / f"face_{frame_idx:06d}.jpg"), face_crop)
        
        cap.release()
        logger.info(f"Extracted active face crops to {output_dir}")

if __name__ == "__main__":
    # Stub for testing
    import sys
    if len(sys.argv) > 1:
        asd = VisualASD()
        asd.detect_active_speaker(sys.argv[1], "asd_test.csv")
