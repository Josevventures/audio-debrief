"""Stage 2: faster-whisper with word timestamps."""
from __future__ import annotations

from pathlib import Path


def pick_compute(device: str) -> tuple[str, str, str]:
    """Return (device, compute_type, warning). CUDA -> float16; otherwise CPU int8 with a warning."""
    if device == "cuda":
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16", ""
            warning = "CUDA requested but ctranslate2 sees no CUDA device; fell back to CPU int8 (slow)."
        except Exception as exc:  # noqa: BLE001
            warning = f"CUDA check failed ({exc}); fell back to CPU int8 (slow)."
        return "cpu", "int8", warning
    return "cpu", "int8", "Running Whisper on CPU int8 by request."


def build_prompt(hotwords: list[str]) -> str:
    seen, ordered = set(), []
    for h in hotwords:
        if h and h.lower() not in seen:
            seen.add(h.lower())
            ordered.append(h)
    return "Job interview. " + ", ".join(ordered[:60]) + "."


def transcribe(wav: str | Path, model_name: str, device: str, hotwords: list[str]) -> dict:
    from faster_whisper import WhisperModel

    dev, compute_type, warning = pick_compute(device)
    model = WhisperModel(model_name, device=dev, compute_type=compute_type)
    segments, info = model.transcribe(
        str(wav), language="en", word_timestamps=True, vad_filter=True,
        initial_prompt=build_prompt(hotwords), beam_size=5, condition_on_previous_text=False,
    )
    words = []
    for seg in segments:  # generator: transcription happens here
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"text": text, "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                              "prob": round(float(w.probability), 3)})
    return {
        "words": words,
        "info": {"model": model_name, "device": dev, "compute_type": compute_type, "warning": warning,
                 "language": info.language, "duration_s": round(float(info.duration), 2)},
    }
