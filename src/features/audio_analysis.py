import logging
import pandas as pd
import numpy as np
import opensmile
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self):
        """
        Initialize Audio Analyzer with OpenSMILE (eGeMAPSv02).
        Raises RuntimeError if OpenSMILE fails to initialize.
        """
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        logger.info("OpenSMILE (eGeMAPSv02) initialized successfully.")

    def extract_prosody(self, audio_path: str) -> Dict[str, float]:
        """
        Extract Jitter, Shimmer, F0, Loudness using OpenSMILE.
        Raises on failure (no fallback).
        """
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

if __name__ == "__main__":
    pass
