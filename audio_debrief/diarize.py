"""Stage 3: pyannote speaker-diarization-3.1 -> turns + overlap regions."""
from __future__ import annotations

from pathlib import Path

from .convert import pyannote_input
from .pipeline import SetupError

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
GATED_MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
]

SETUP_STEPS = (
    "No HuggingFace token found. pyannote's models are gated; do this once:\n"
    "  1. Logged in on huggingface.co, accept the terms on each model page:\n"
    + "".join(f"       https://huggingface.co/{m}\n" for m in GATED_MODELS)
    + "  2. In the app folder run:  uv run hf auth login   (paste a read token)\n"
    "  3. Re-run this command."
)


def require_token() -> str:
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise SetupError(SETUP_STEPS)
    return token


def diarize(wav: str | Path, device: str, token: str) -> dict:
    import torch
    from ._torch_compat import patch_torch_load
    patch_torch_load()
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    if pipe is None:
        raise SetupError("Pipeline.from_pretrained returned None — model terms not accepted?\n" + SETUP_STEPS)
    dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    pipe.to(torch.device(dev))
    annotation = pipe(pyannote_input(wav))
    turns = [{"speaker": label, "start": round(seg.start, 3), "end": round(seg.end, 3)}
             for seg, _, label in annotation.itertracks(yield_label=True)]
    turns.sort(key=lambda t: t["start"])
    overlaps = [{"start": round(seg.start, 3), "end": round(seg.end, 3)} for seg in annotation.get_overlap()]
    return {"turns": turns, "overlaps": overlaps, "model": DIARIZATION_MODEL, "device": dev,
            "speakers": sorted({t["speaker"] for t in turns})}
