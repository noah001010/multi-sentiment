import logging
import torch
from faster_whisper import WhisperModel
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class WhisperAligner:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda" if torch.cuda.is_available() else "cpu", compute_type: str = "float16"):
        """
        Initialize the Whisper ASR model.
        Args:
            model_size: Size of the Whisper model (default: "large-v3")
            device: Device to run on ("cuda" or "cpu")
            compute_type: Quantization type (default: "float16" for GPU)
        """
        # Parse device and index (e.g. "cuda:0" -> device="cuda", device_index=0)
        device_index = 0
        if isinstance(device, str) and ":" in device and device.startswith("cuda"):
            parts = device.split(":")
            device = parts[0]
            if len(parts) > 1 and parts[1].isdigit():
                device_index = int(parts[1])

        logger.info(f"Loading Whisper model '{model_size}' on {device}:{device_index} with {compute_type}...")
        try:
            # device_index expects a list or int
            self.model = WhisperModel(model_size, device=device, device_index=device_index, compute_type=compute_type)
        except Exception as e:
            logger.warning(f"Failed to load Whisper on {device}: {e}. Falling back to CPU/int8.")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str, language: str = "ja") -> Dict[str, Any]:
        """
        Transcribe audio and return segments with word-level timestamps.
        Args:
            audio_path: Path to the audio file.
            language: Target language code (default: "ja").
        Returns:
            Dict containing 'text', 'segments' (list of dicts with start, end, text, words).
        """
        logger.info(f"Transcribing {audio_path}...")
        segments_generator, info = self.model.transcribe(
            audio_path, 
            language=language, 
            word_timestamps=True,
            vad_filter=True, # Use Silero VAD to filter silence
        )

        results = []
        full_text = ""
        
        for segment in segments_generator:
            seg_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in segment.words] if segment.words else []
            }
            results.append(seg_data)
            full_text += segment.text + " "

        logger.info(f"Transcription complete. Detected language '{info.language}' with probability {info.language_probability:.2f}")
        return {"text": full_text.strip(), "segments": results}

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save transcription results to CSV."""
        df = pd.DataFrame(results['segments'])
        df.to_csv(output_path, index=False)
        logger.info(f"Saved transcriptions to {output_path}")

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 1:
        aligner = WhisperAligner(model_size="tiny", device="cpu", compute_type="int8") # Use tiny for quick test
        print(aligner.transcribe(sys.argv[1]))
