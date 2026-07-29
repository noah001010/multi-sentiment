import os
import sys
# pyrefly: ignore [missing-import]
import torch

# EXTREMELY AGGRESSIVE PATCH FOR TORCH 2.6+
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

if hasattr(torch.serialization, 'add_safe_globals'):
    import torch.torch_version
    safe_types = [torch.torch_version.TorchVersion]
    try:
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        safe_types.extend([Specifications, Problem, Resolution])
    except ImportError:
        pass
    torch.serialization.add_safe_globals(safe_types)

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)

torch.load = _patched_load
import torch.serialization as serial
serial.load = _patched_load

import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print(f"Loading pipeline with token: {hf_token[:5]}...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)

print("Processing audio...")
diarization = pipeline("data/boj_conference.wav")

print(f"Type of diarization result: {type(diarization)}")
print(f"Attributes: {dir(diarization)}")

if hasattr(diarization, "itertracks"):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"{turn.start:.2f} - {turn.end:.2f}: {speaker}")
else:
    print("Result does not have itertracks. Checking common alternatives...")
    try:
        # Check if it's a dict or has other iteration methods
        print(f"Diarization object: {diarization}")
    except Exception as e:
        print(f"Failed to print: {e}")
