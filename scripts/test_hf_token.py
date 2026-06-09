import os
import torch
import torchaudio

# Global fix for PyTorch 2.6+ security
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

# SOTA Compatibility Patches
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

print(f"Token Found: {'Yes' if token and token != 'PASTE_YOUR_TOKEN_HERE' else 'No'}")

try:
    print("Loading pyannote/speaker-diarization-3.1 ...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    if pipeline:
        print("SUCCESS: Pipeline loaded successfully.")
    else:
        print("FAILED: Pipeline object is None.")
except Exception as e:
    print(f"ERROR: {e}")
