"""Stage 4: speaker embeddings, voiceprint match, naming, enrollment."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .pipeline import app_root

EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


def voiceprint_path(name: str = "self") -> Path:
    return app_root() / "voiceprints" / f"{name}.npy"


# --- pure helpers (unit-tested) ---------------------------------------------------

def clean_segments(turns: list[dict], overlaps: list[dict], label: str,
                   max_total_s: float = 60.0, min_len_s: float = 1.0) -> list[dict]:
    """Longest segments of `label` that touch no overlap region, up to ~max_total_s in total."""
    def clean(t: dict) -> bool:
        return all(t["end"] <= o["start"] or t["start"] >= o["end"] for o in overlaps)

    cands = [t for t in turns if t["speaker"] == label and t["end"] - t["start"] >= min_len_s and clean(t)]
    cands.sort(key=lambda t: t["end"] - t["start"], reverse=True)
    out, total = [], 0.0
    for t in cands:
        out.append(t)
        total += t["end"] - t["start"]
        if total >= max_total_s:
            break
    return out


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def match_self(embeddings: dict[str, np.ndarray], voiceprint: np.ndarray, threshold: float) -> tuple[str | None, dict]:
    scores = {label: round(cosine(e, voiceprint), 4) for label, e in embeddings.items()}
    if not scores:
        return None, scores
    best = max(scores, key=scores.get)
    return (best if scores[best] >= threshold else None), scores


def name_speakers(turns: list[dict], self_label: str | None, other_names: list[str], self_name: str = "Me") -> dict:
    """The user's label -> self_name; the rest take other_names in order of first appearance, then 'Speaker N'."""
    order: list[str] = []
    for t in sorted(turns, key=lambda t: t["start"]):
        if t["speaker"] not in order:
            order.append(t["speaker"])
    mapping, names = {}, list(other_names)
    n = 0
    for label in order:
        if label == self_label:
            mapping[label] = self_name
        else:
            n += 1
            mapping[label] = names.pop(0) if names else f"Speaker {n}"
    return mapping


def sample_text(words_with_speaker: list[dict], label: str, seconds: float = 10.0) -> str:
    ws = [w for w in words_with_speaker if w.get("speaker") == label]
    if not ws:
        return ""
    t0 = ws[0]["start"]
    return " ".join(w["text"] for w in ws if w["start"] <= t0 + seconds)


# --- model-backed -------------------------------------------------------------

def speaker_embeddings(wav: str | Path, turns: list[dict], overlaps: list[dict], device: str, token: str) -> dict:
    import torch
    from ._torch_compat import patch_torch_load
    patch_torch_load()
    from pyannote.audio import Inference
    from pyannote.core import Segment

    from .convert import pyannote_input

    dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    inference = Inference(EMBED_MODEL, window="whole", use_auth_token=token, device=torch.device(dev))
    file = pyannote_input(wav)
    total = file["waveform"].shape[1] / file["sample_rate"]
    out: dict[str, list[float]] = {}
    for label in sorted({t["speaker"] for t in turns}):
        segs = clean_segments(turns, overlaps, label) or \
            sorted([t for t in turns if t["speaker"] == label], key=lambda t: t["start"] - t["end"])[:3]
        embs = []
        for s in segs:
            emb = np.asarray(inference.crop(file, Segment(s["start"], min(s["end"], total))), dtype=np.float32)
            embs.append(l2_normalize(emb.reshape(-1) if emb.ndim == 1 or emb.shape[0] == 1 else emb.mean(axis=0)))
        if embs:
            out[label] = l2_normalize(np.mean(embs, axis=0)).tolist()
    return out


def save_voiceprint(embedding, name: str = "self") -> Path:
    path = voiceprint_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, l2_normalize(np.asarray(embedding, dtype=np.float32)))
    return path


def load_voiceprint(name: str = "self") -> np.ndarray | None:
    path = voiceprint_path(name)
    return np.load(path) if path.exists() else None
