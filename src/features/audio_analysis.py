import logging
import pandas as pd
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoProcessor, AutoModelForAudioClassification

logger = logging.getLogger(__name__)

def check_cuda_working() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(1, 1).to("cuda")
        y = torch.nn.functional.linear(x, x)
        return True
    except Exception as e:
        logger.warning(f"CUDA is available in PyTorch, but kernel execution failed: {e}. Falling back to CPU.")
        return False


class AudioAnalyzer:
    def __init__(self):
        """
        Initialize Audio Analyzer with Wav2Vec2 Deep Emotion Model:
        audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
        Outputs: Valence (index 1), Arousal (index 0), Dominance (index 2).
        """
        self.device = torch.device("cuda" if check_cuda_working() else "cpu")
        logger.info(f"Initializing Wav2Vec2 Emotion model on device: {self.device}")
        try:
            model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_config(config)
            
            import huggingface_hub
            model_path = huggingface_hub.hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            
            key_mapping = {
                "classifier.dense.weight": "projector.weight",
                "classifier.dense.bias": "projector.bias",
                "classifier.out_proj.weight": "classifier.weight",
                "classifier.out_proj.bias": "classifier.bias"
            }
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = key_mapping.get(k, k)
                new_state_dict[new_k] = v
                
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Wav2Vec2 Audio Emotion model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Wav2Vec2 model: {e}")
            raise RuntimeError(f"Wav2Vec2 initialization failed: {e}")

    def extract_prosody(self, audio_path: str) -> Dict[str, float]:
        """
        Extract continuous emotion dimensions (Arousal, Valence, Dominance) via Wav2Vec2.
        Returns:
            Dict[str, float]: {"audio_arousal": float, "audio_valence": float, "audio_dominance": float}
        """
        result = {}

        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample to 16000Hz if necessary
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
                sr = 16000
                
            # Mix down to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            waveform_1d = waveform.squeeze(0)
            
            inputs = self.processor(waveform_1d, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                
            logits_np = logits.squeeze(0).cpu().numpy()
            
            # audeering outputs: 0=Arousal, 1=Valence, 2=Dominance
            result["audio_arousal"] = float(logits_np[0])
            result["audio_valence"] = float(logits_np[1])
            result["audio_dominance"] = float(logits_np[2])
            
        except Exception as e:
            logger.error(f"Wav2Vec2 extraction error for {audio_path}: {e}")
            result["audio_arousal"] = 0.0
            result["audio_valence"] = 0.0
            result["audio_dominance"] = 0.0

        return result

if __name__ == "__main__":
    pass
