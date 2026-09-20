"""Stage 1: ffmpeg -> 16 kHz mono PCM WAV, plus waveform loading shared by later stages."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .pipeline import SetupError

TARGET_SR = 16000
FFMPEG_HINT = (
    "ffmpeg not found on PATH. Install it (Windows: `winget install Gyan.FFmpeg`, then open a new "
    "terminal) and re-run."
)


def to_wav(audio: str | Path, out: str | Path, use_cache: bool = True) -> Path:
    audio, out = Path(audio), Path(out)
    if use_cache and out.exists() and out.stat().st_size > 0:
        return out
    if not audio.exists():
        raise FileNotFoundError(f"audio file not found: {audio}")
    if shutil.which("ffmpeg") is None:
        raise SetupError(FFMPEG_HINT)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio), "-ac", "1", "-ar", str(TARGET_SR),
           "-c:a", "pcm_s16le", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
    return out


def load_waveform(wav: str | Path) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1] and the sample rate."""
    import soundfile as sf

    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def pyannote_input(wav: str | Path) -> dict:
    """{"waveform": tensor[1, N], "sample_rate": sr} — avoids torchaudio I/O entirely."""
    import torch

    y, sr = load_waveform(wav)
    return {"waveform": torch.from_numpy(y).unsqueeze(0), "sample_rate": sr}


def duration_s(wav: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(wav))
    return info.frames / info.samplerate
