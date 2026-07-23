import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MultimodalIntegrator:
    def __init__(self):
        pass

    def align_and_merge(
        self,
        text_df: pd.DataFrame,
        visual_df: pd.DataFrame,
        audio_prosody: pd.DataFrame,
        diarization_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Merge all modalities onto the Text timeline (sentence level).
        """
        logger.info("Integrating multimodal data with SOTA Speaker IDs...")
        
        merged_data = []

        # Ensure visual_df has timestamp if it has frame
        if 'timestamp' not in visual_df.columns and 'frame' in visual_df.columns:
            # Assume 30fps if not specified, but caller should ideally provide timestamp
            visual_df['timestamp'] = visual_df['frame'] / 30.0

        for idx, row in text_df.iterrows():
            start = row['start']
            end = row['end']
            
            # Slice Visual Data
            vis_slice = visual_df[
                (visual_df['timestamp'] >= start) & 
                (visual_df['timestamp'] <= end)
            ]
            
            # Aggregate Visual
            if not vis_slice.empty:
                vis_metrics = {
                    "mean_AU04": vis_slice['AU04'].mean() if 'AU04' in vis_slice.columns else 0.0,
                    "max_AU04": vis_slice['AU04'].max() if 'AU04' in vis_slice.columns else 0.0,
                    "mean_AU12": vis_slice['AU12'].mean() if 'AU12' in vis_slice.columns else 0.0,
                    "mean_valence": vis_slice['valence'].mean() if 'valence' in vis_slice.columns else 0.0,
                    "mean_arousal": vis_slice['arousal'].mean() if 'arousal' in vis_slice.columns else 0.0,
                    "blink_rate": vis_slice['is_blink'].mean() * 60 if 'is_blink' in vis_slice.columns else 0.0, # blinks per minute (approx)
                    "face_confidence": 1.0 # Placeholder
                }
            else:
                vis_metrics = {
                    "mean_AU04": 0.0, "max_AU04": 0.0, 
                    "mean_AU12": 0.0, "mean_valence": 0.0, "mean_arousal": 0.0,
                    "blink_rate": 0.0, "face_confidence": 0.0
                }

            # --- Speaker Alignment ---
            speaker_id = "UNKNOWN"
            if diarization_df is not None and not diarization_df.empty:
                # Find speaker with max overlap in this [start, end] window
                overlap = diarization_df[
                    (diarization_df['start'] < end) & (diarization_df['end'] > start)
                ].copy()
                if not overlap.empty:
                    # Simple heuristic: take the speaker of the longest overlap
                    overlap['duration'] = np.minimum(overlap['end'], end) - np.maximum(overlap['start'], start)
                    speaker_id = overlap.sort_values('duration', ascending=False).iloc[0]['speaker']
            
            # --- Discrepancy Reasoning ---
            # Sentiment (Text) vs Valence (Face/Audio)
            sentiment = row['sentiment_score']
            facial_val = vis_metrics['mean_valence']
            audio_val = row.get('audio_valence', 0.0)
            
            reasoning = "N/A"
            discrepancy_score = 0.0
            
            # Case: Positive statement but negative face or audio (tension/concern)
            if sentiment > 0.5 and (facial_val < -0.2 or audio_val < -0.2):
                discrepancy_score = abs(sentiment - facial_val) + abs(sentiment - audio_val)
                reasoning = f"ポジティブな発信内容に対し、表情（Valence={facial_val:.2f}）または音声トーン（Valence={audio_val:.2f}）にネガティブな兆候が検出されました。"
            # Case: Negative statement but positive face or audio (calm/relief)
            elif sentiment < -0.5 and (facial_val > 0.2 or audio_val > 0.2):
                discrepancy_score = abs(sentiment - facial_val) + abs(sentiment - audio_val)
                reasoning = f"ネガティブな発信内容に対し、表情（Valence={facial_val:.2f}）または音声トーン（Valence={audio_val:.2f}）にポジティブなシグナルが検出されました。"
                
            # Create merged row
            merged_row = row.to_dict()
            merged_row.update(vis_metrics)
            merged_row['speaker'] = speaker_id
            merged_row['discrepancy_reasoning'] = reasoning
            merged_row['discrepancy_score_sota'] = discrepancy_score
            
            merged_data.append(merged_row)
            
        return pd.DataFrame(merged_data)

if __name__ == "__main__":
    pass
