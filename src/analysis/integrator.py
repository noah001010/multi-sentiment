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
                    "mean_AU04": vis_slice['AU04'].mean(),
                    "max_AU04": vis_slice['AU04'].max(),
                    "mean_AU12": vis_slice['AU12'].mean(),
                    "blink_rate": vis_slice['is_blink'].mean() * 60, # blinks per minute (approx)
                    "face_confidence": 1.0 # Placeholder
                }
            else:
                vis_metrics = {
                    "mean_AU04": 0.0, "max_AU04": 0.0, 
                    "mean_AU12": 0.0, "blink_rate": 0.0,
                    "face_confidence": 0.0
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
            # Sentiment vs Emotions (AU12 positive, AU04 negative)
            sentiment = row['sentiment_score']
            facial_pos = vis_metrics['mean_AU12']
            facial_neg = vis_metrics['mean_AU04']
            
            reasoning = "N/A"
            discrepancy_score = 0.0
            
            # Case: Positive words but negative face
            if sentiment > 0.3 and facial_neg > 0.4:
                discrepancy_score = abs(sentiment) + facial_neg
                reasoning = f"ポジティブな発信内容に対し、眉間の寄せ（AU04={facial_neg:.2f}）が強く検出されました。"
            # Case: Negative words but positive face (masked emotion?)
            elif sentiment < -0.3 and facial_pos > 0.4:
                discrepancy_score = abs(sentiment) + facial_pos
                reasoning = f"ネガティブな内容の発信ですが、口角の引き上げ（AU12={facial_pos:.2f}）が検出されました。"
                
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
