import logging
import pandas as pd
import numpy as np
import opensmile
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self):
        """
        Initialize Audio Analyzer.
        """
        # OpenSMILE
        try:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            self.has_smile = True
        except Exception as e:
            logger.warning(f"OpenSMILE init failed: {e}. Prosody features will be limited.")
            self.has_smile = False

        # Wav2Vec2 - Disabled because not used in main pipeline and causes loading errors
        # logger.info("Loading Wav2Vec2 model...")
        # self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        # self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        # self.device = "cpu" # Force CPU to avoid Blackwell kernel errors
        # self.model.to(self.device)

    def extract_prosody(self, audio_path: str) -> Dict[str, float]:
        """
        Extract Jitter, Shimmer, F0, Loudness using OpenSMILE.
        """
        if not self.has_smile:
            return {"jitter": 0.0, "shimmer": 0.0, "F0_mean": 0.0, "loudness": 0.0}

        try:
            # eGeMAPS returns a huge DF. We select key metrics.
            df = self.smile.process_file(audio_path)
            
            # Mapping common names to eGeMAPS columns (approximate)
            # F0semitoneFrom27.5Hz_sma3nz_amean -> F0
            # jitterLocal_sma3nz_amean -> Jitter
            # shimmerLocaldB_sma3nz_amean -> Shimmer
            # loudness_sma3_amean -> Loudness
            
            result = {
                "F0_mean": df.get("F0semitoneFrom27.5Hz_sma3nz_amean", [0.0])[0],
                "jitter": df.get("jitterLocal_sma3nz_amean", [0.0])[0],
                "shimmer": df.get("shimmerLocaldB_sma3nz_amean", [0.0])[0],
                "loudness": df.get("loudness_sma3_amean", [0.0])[0]
            }
            
            # Diagnostic: check for flat-line
            if all(v == 0.0 for v in result.values()):
                logger.warning(f"Audio features for {audio_path} are all zero. OpenSMILE may have failed to extract meaningful data.")
            
            return result
        except Exception as e:
            logger.error(f"Prosody extraction failed for {audio_path}: {e}")
            return {"jitter": 0.0, "shimmer": 0.0, "F0_mean": 0.0, "loudness": 0.0}

    def extract_embedding(self, audio_file: str):
        """
        Extract Wav2Vec2 embedding (pooled).
        """
        waveform, sr = torchaudio.load(audio_file)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        input_values = self.processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            # Use last hidden state mean as simple embedding
            hidden_states = outputs.last_hidden_state
            embedding = torch.mean(hidden_states, dim=1).cpu().numpy().flatten()
            
        return embedding

if __name__ == "__main__":
    pass
