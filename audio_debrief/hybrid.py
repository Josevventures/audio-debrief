"""Hybrid diarizer: Whisper VAD word units + WeSpeaker embeddings + voiceprint / clustering.

Why: on a phone recording of a video call the far side is loudspeaker audio,
which pyannote's segmentation model largely ignores while Whisper's VAD does not. So we
take Whisper's words as the speech evidence, embed short units of them, and label each
unit by cosine similarity to the user's enrolled voiceprint (or by clustering when none exists).
Overlaps are not observable this way; the pipeline reports them as not measured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .speakers import EMBED_MODEL, cosine, l2_normalize

# --- pure helpers (unit-tested) ---------------------------------------------------


def _unit(words: list[dict], idx: list[int]) -> dict:
    return {"start": words[idx[0]]["start"], "end": words[idx[-1]]["end"], "word_idx": idx}


def make_units(words: list[dict], gap_s: float = 0.3, max_len_s: float = 3.0) -> list[dict]:
    """Split the word stream at silences > gap_s and cap units at max_len_s (on word boundaries)."""
    units, cur = [], []
    for i, w in enumerate(words):
        if cur and (w["start"] - words[cur[-1]]["end"] > gap_s or w["end"] - words[cur[0]]["start"] > max_len_s):
            units.append(_unit(words, cur))
            cur = []
        cur.append(i)
    if cur:
        units.append(_unit(words, cur))
    return units


def embed_window(unit: dict, total_s: float, min_len_s: float, pad_s: float) -> tuple[float, float]:
    s, e = unit["start"], unit["end"]
    if e - s < min_len_s:
        s, e = s - pad_s, e + pad_s
    return round(max(0.0, s), 3), round(min(total_s, e), 3)


def label_units(scores: list[float], match: float, ambiguous_low: float) -> list[bool]:
    """Self when cosine >= match; the ambiguous zone [ambiguous_low, match) inherits the previous label."""
    out, prev = [], False
    for s in scores:
        cur = True if s >= match else (prev if s >= ambiguous_low else False)
        out.append(cur)
        prev = cur
    return out


def cluster_centroids(embs, labels: list[int]) -> dict[int, np.ndarray]:
    X = np.asarray(embs, dtype=np.float64)
    return {c: l2_normalize(X[[i for i, l in enumerate(labels) if l == c]].mean(axis=0)) for c in sorted(set(labels))}


def cluster_units(embs, threshold: float, durations: list[float], min_cluster_s: float,
                  merge_sim: float = 0.5) -> list[int]:
    """Speaker clusters for the units not claimed by the voiceprint.

    1. Average-linkage agglomerative clustering on cosine distance (unit-level, noisy on far-side audio).
    2. Anchors = clusters with at least min_cluster_s of speech (a real participant talks that long);
       if none qualifies, the biggest cluster is the anchor.
    3. Anchors whose centroids agree above merge_sim are one speaker (a speaker split by noise).
    4. Every other cluster is absorbed into the anchor with the most similar centroid.
    """
    X = np.asarray(embs, dtype=np.float64)
    n = len(X)
    if n == 0:
        return []
    if n == 1:
        return [0]
    from scipy.cluster.hierarchy import fcluster, linkage

    labels = [int(c) - 1 for c in fcluster(linkage(X, method="average", metric="cosine"), t=threshold, criterion="distance")]
    durations = np.asarray(durations, dtype=np.float64)
    total = {c: float(durations[[i for i, l in enumerate(labels) if l == c]].sum()) for c in set(labels)}
    anchors = [c for c in sorted(total, key=total.get, reverse=True) if total[c] >= min_cluster_s] or \
        [max(total, key=total.get)]
    while len(anchors) > 1:  # merge the most similar anchor pair while it clears merge_sim
        cents = cluster_centroids(X, labels)
        pairs = [(cosine(cents[a], cents[b]), a, b) for i, a in enumerate(anchors) for b in anchors[i + 1:]]
        sim, a, b = max(pairs)
        if sim < merge_sim:
            break
        labels = [a if l == b else l for l in labels]
        anchors.remove(b)
    cents = cluster_centroids(X, labels)
    for c in set(labels) - set(anchors):
        target = max(anchors, key=lambda a: cosine(cents[c], cents[a]))
        labels = [target if l == c else l for l in labels]
    return labels


def refine_labels(embs, scores: list[float], flags: list[bool], cluster_ids: list, centroids: dict,
                  match: float, refine_min: float):
    """Nearest-centroid pass for units the absolute voiceprint rule did not claim (score < match):
    a non-self unit closer to the voiceprint than to every speaker centroid (and >= refine_min) becomes self;
    an inherited-self unit closer to a speaker centroid than to the voiceprint goes to that speaker."""
    X = np.asarray(embs, dtype=np.float64)
    flags, cluster_ids = list(flags), list(cluster_ids)
    stats = {"refined_to_self": 0, "refined_from_self": 0}
    if not centroids:
        return flags, cluster_ids, stats
    for i in range(len(X)):
        if scores[i] >= match:
            continue
        best = max(centroids, key=lambda c: cosine(X[i], centroids[c]))
        best_sim = cosine(X[i], centroids[best])
        if flags[i] and best_sim > scores[i]:
            flags[i], cluster_ids[i] = False, best
            stats["refined_from_self"] += 1
        elif not flags[i] and scores[i] >= refine_min and scores[i] > best_sim:
            flags[i], cluster_ids[i] = True, None
            stats["refined_to_self"] += 1
    return flags, cluster_ids, stats


def assign_labels(self_flags: list[bool] | None, cluster_ids: list) -> list[str]:
    """Self -> SPEAKER_00 (when any unit is self); clusters -> SPEAKER_NN by first appearance."""
    mapping, labels = {}, []
    next_id = 1 if self_flags and any(self_flags) else 0
    for i, c in enumerate(cluster_ids):
        if self_flags and self_flags[i]:
            labels.append("SPEAKER_00")
            continue
        if c not in mapping:
            mapping[c] = f"SPEAKER_{next_id:02d}"
            next_id += 1
        labels.append(mapping[c])
    return labels


def units_to_turns(units: list[dict], labels: list[str]) -> list[dict]:
    turns: list[dict] = []
    for u, label in zip(units, labels):
        if turns and turns[-1]["speaker"] == label:
            turns[-1]["end"] = u["end"]
        else:
            turns.append({"speaker": label, "start": u["start"], "end": u["end"]})
    return turns


def assemble(units: list[dict], embs, voiceprint, vcfg: dict) -> dict:
    """Label units (voiceprint + clustering) and build the diarization.json payload."""
    embs = np.asarray(embs, dtype=np.float32)
    n = len(units)
    durations = [u["end"] - u["start"] for u in units]
    flags, summary = None, {"self_units": 0, "ambiguous_units": 0}
    if voiceprint is not None and n:
        scores = (embs @ l2_normalize(np.asarray(voiceprint, dtype=np.float32))).tolist()
        flags = label_units(scores, vcfg["match_threshold"], vcfg["ambiguous_low"])
        self_scores = [s for s, f in zip(scores, flags) if f]
        other_scores = [s for s, f in zip(scores, flags) if not f]
        summary = {"self_units": sum(flags),
                   "ambiguous_units": sum(vcfg["ambiguous_low"] <= s < vcfg["match_threshold"] for s in scores),
                   "self_min": round(min(self_scores), 3) if self_scores else None,
                   "other_max": round(max(other_scores), 3) if other_scores else None}
    non_idx = [i for i in range(n) if not (flags and flags[i])]
    cl = cluster_units(embs[non_idx], vcfg["cluster_threshold"], [durations[i] for i in non_idx],
                       vcfg["min_cluster_s"], vcfg.get("centroid_merge_sim", 0.5))
    cluster_ids: list = [None] * n
    for k, i in enumerate(non_idx):
        cluster_ids[i] = cl[k]
    summary.update({"refined_to_self": 0, "refined_from_self": 0})
    if flags is not None and non_idx:
        centroids = cluster_centroids(embs[non_idx], cl)
        flags, cluster_ids, refined = refine_labels(embs, scores, flags, cluster_ids, centroids,
                                                    vcfg["match_threshold"], vcfg.get("refine_min", 0.25))
        summary.update(refined)
        summary["self_units"] = sum(flags)
    labels = assign_labels(flags, cluster_ids)
    speakers = sorted(set(labels))
    embeddings = {sp: l2_normalize(embs[[i for i, l in enumerate(labels) if l == sp]].mean(axis=0)).tolist()
                  for sp in speakers}
    return {"turns": units_to_turns(units, labels), "overlaps": [], "speakers": speakers, "model": EMBED_MODEL,
            "method": "hybrid", "self_label": "SPEAKER_00" if flags and any(flags) else None,
            "unit_count": n, "scores": summary, "embeddings": embeddings}


# --- model-backed ---------------------------------------------------------------


def embed_units(wav, units: list[dict], device: str, token: str, min_len_s: float, pad_s: float,
                cache_path: Path | None = None, log=print) -> np.ndarray:
    starts = np.array([u["start"] for u in units], dtype=np.float64)
    ends = np.array([u["end"] for u in units], dtype=np.float64)
    if cache_path and cache_path.exists():
        z = np.load(cache_path)
        if len(z["starts"]) == len(starts) and np.allclose(z["starts"], starts) and np.allclose(z["ends"], ends):
            return z["embs"]
    import torch
    from pyannote.audio import Inference
    from pyannote.core import Segment

    from ._torch_compat import patch_torch_load
    from .convert import pyannote_input

    patch_torch_load()
    dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    inference = Inference(EMBED_MODEL, window="whole", use_auth_token=token, device=torch.device(dev))
    file = pyannote_input(wav)
    total = file["waveform"].shape[1] / file["sample_rate"]
    out = []
    for k, u in enumerate(units):
        s, e = embed_window(u, total, min_len_s, pad_s)
        emb = np.asarray(inference.crop(file, Segment(s, e)), dtype=np.float32)
        out.append(l2_normalize(emb.reshape(-1) if emb.ndim == 1 or emb.shape[0] == 1 else emb.mean(axis=0)))
        if (k + 1) % 200 == 0:
            log(f"[diarize] embedded {k + 1}/{len(units)} units")
    embs = np.stack(out) if out else np.zeros((0, 1), dtype=np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, starts=starts, ends=ends, embs=embs)
    return embs


def diarize(wav, words: list[dict], voiceprint, cfg: dict, device: str, token: str, cache_dir: Path, log=print) -> dict:
    h = cfg["hybrid"]
    units = make_units(words, h["unit_gap_s"], h["unit_max_s"])
    log(f"[diarize] hybrid: {len(units)} units from {len(words)} words")
    embs = embed_units(wav, units, device, token, h["embed_min_len_s"], h["embed_pad_s"],
                       cache_dir / "unit_embeddings.npz", log)
    return assemble(units, embs, voiceprint, cfg["voiceprint"])
