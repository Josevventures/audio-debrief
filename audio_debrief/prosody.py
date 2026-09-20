"""Stage 7: tone proxies per answer by the analyzed speaker (librosa pyin F0 + RMS).

Baseline = mean of the per-utterance values across all of that speaker's utterances on the call;
z = (answer value - baseline mean) / std of those per-utterance values, so an answer is
compared against the spread of comparable-sized units, not against frame-level noise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .convert import load_waveform

FRAME, HOP = 1024, 256
F0_MIN, F0_MAX = 65.0, 400.0
KEYS = ("f0_median_hz", "f0_std_hz", "rms_mean", "rms_var")
Z_KEYS = {"f0_median_hz": "f0_median", "f0_std_hz": "f0_std", "rms_mean": "rms_mean", "rms_var": "rms_var"}


def _stats(f0: np.ndarray, rms: np.ndarray) -> dict:
    voiced = f0[~np.isnan(f0)]
    return {"f0_median_hz": float(np.median(voiced)) if voiced.size else None,
            "f0_std_hz": float(np.std(voiced)) if voiced.size else None,
            "rms_mean": float(np.mean(rms)) if rms.size else None,
            "rms_var": float(np.var(rms)) if rms.size else None,
            "voiced_fraction": float(voiced.size / f0.size) if f0.size else None}


def baseline_from_utterances(per_utt: list[dict]) -> dict:
    base: dict = {"utterance_count": len(per_utt), "spread": {}}
    for k in KEYS:
        vals = [u[k] for u in per_utt if u.get(k) is not None]
        base[k] = float(np.mean(vals)) if vals else None
        base["spread"][k] = float(np.std(vals)) if vals else None
    return base


def _z(value, center, scale):
    if value is None or center is None or not scale:
        return None
    return round((value - center) / scale, 2)


def zscores(stats: dict, base: dict) -> dict:
    return {Z_KEYS[k]: _z(stats.get(k), base.get(k), base["spread"].get(k)) for k in KEYS}


def _rms(x: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else None


def speech_rms(y: np.ndarray, sr: int, utterances: list[dict], speaker: str) -> float | None:
    """RMS over all of one speaker's utterance samples — the baseline pauses are judged against."""
    parts = [y[int(u["start"] * sr):int(u["end"] * sr)] for u in utterances if u["speaker"] == speaker]
    parts = [p for p in parts if p.size]
    return _rms(np.concatenate(parts)) if parts else None


def annotate_pauses(pauses: list[dict], y: np.ndarray, sr: int, silence_ratio: float = 0.6,
                    baseline_rms: float | None = None, before_s: float = 2.0) -> list[dict]:
    """Add gap_rms (RMS inside the gap), gap_rms_ratio (gap_rms / baseline) and kind: 'silence'
    below silence_ratio, else 'unvoiced/possible backchannel' (energy Whisper did not transcribe:
    the other speaker's "mm-hmm", a laugh, breath). The baseline is the speaker's whole-call
    speech RMS when given (preferred: the 2 s before a pause is usually a trailing-off tail, which
    made real silences look loud on a real call); otherwise the before_s seconds preceding
    the gap. Mutates and returns pauses."""
    for p in pauses:
        a, b = int(p["at"] * sr), int((p["at"] + p["gap_s"]) * sr)
        inside = _rms(y[a:b])
        p["gap_rms"] = round(inside, 4) if inside is not None else None
        base = baseline_rms if baseline_rms else (_rms(y[max(0, int((p["at"] - before_s) * sr)):a]) if a > 0 else None)
        if base is None or inside is None or base == 0:
            p["gap_rms_ratio"], p["kind"] = None, "unknown"
            continue
        p["gap_rms_ratio"] = round(inside / base, 3)
        p["kind"] = "silence" if p["gap_rms_ratio"] < silence_ratio else "unvoiced/possible backchannel"
    return pauses


def _utterance_frames(y: np.ndarray, sr: int, u: dict):
    import librosa

    seg = y[int(u["start"] * sr):int(u["end"] * sr)]
    if len(seg) < FRAME * 2:
        return None
    f0, _, _ = librosa.pyin(seg, fmin=F0_MIN, fmax=F0_MAX, sr=sr, frame_length=FRAME, hop_length=HOP)
    rms = librosa.feature.rms(y=seg, frame_length=FRAME, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    t = u["start"] + librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP)
    return t, f0[:n], rms[:n]


def analyze(wav: str | Path, utterances: list[dict], answers: list[dict], me: str) -> dict:
    y, sr = load_waveform(wav)
    times, f0s, rmss, per_utt = [], [], [], []
    for u in utterances:
        if u["speaker"] != me:
            continue
        frames = _utterance_frames(y, sr, u)
        if frames is None:
            continue
        t, f0, rms = frames
        times.append(t), f0s.append(f0), rmss.append(rms)
        per_utt.append({**_stats(f0, rms), "start": u["start"]})
    if not times:
        raise RuntimeError(f"no {me} speech found for prosody baseline")
    t, f0, rms = np.concatenate(times), np.concatenate(f0s), np.concatenate(rmss)
    base = baseline_from_utterances(per_utt)
    per_answer = []
    for a in answers:
        mask = (t >= a["start"]) & (t <= a["end"])
        s = _stats(f0[mask], rms[mask])
        per_answer.append({**s, "start": a["start"], "z": zscores(s, base)})
    return {"baseline": base, "answers": per_answer, "frames": int(t.size),
            "method": "librosa.pyin + rms, frame 1024 / hop 256 @ 16 kHz; z vs per-utterance spread"}
