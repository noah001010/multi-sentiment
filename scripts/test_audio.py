import opensmile
import audiofile
import numpy as np

try:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print("OpenSMILE Initialized.")
    
    # Test on the conference wav if exists
    wav_path = "data/boj_conference.wav"
    import os
    if os.path.exists(wav_path):
        signal, sampling_rate = audiofile.read(wav_path, duration=10)
        feats = smile.process_signal(signal, sampling_rate)
        print("Feature extraction SUCCESS.")
        print(feats[['jitterLocal_sma_amean', 'shimmerLocal_sma_amean', 'F0semitoneFrom27.5Hz_sma_amean']])
    else:
        print(f"Wav file not found: {wav_path}")
except Exception as e:
    print(f"OpenSMILE ERROR: {e}")
