"""Word -> speaker assignment and utterance building."""
from __future__ import annotations


def _distance(mid: float, turn: dict) -> float:
    if turn["start"] <= mid <= turn["end"]:
        return 0.0
    return min(abs(mid - turn["start"]), abs(mid - turn["end"]))


def _overlap(word: dict, turn: dict) -> float:
    return min(word["end"], turn["end"]) - max(word["start"], turn["start"])


def _assign(words: list[dict], turns: list[dict]) -> tuple[list[dict], int]:
    out, fallback = [], 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        covering = [t for t in turns if t["start"] <= mid <= t["end"]]
        if len(covering) == 1:
            speaker = covering[0]["speaker"]
        elif covering:
            speaker = max(covering, key=lambda t: (_overlap(w, t), -t["start"]))["speaker"]
        elif turns:
            speaker = min(turns, key=lambda t: _distance(mid, t))["speaker"]
            fallback += 1
        else:
            speaker = "UNKNOWN"
            fallback += 1
        out.append({**w, "speaker": speaker})
    return out, fallback


def assign_speakers(words: list[dict], turns: list[dict]) -> list[dict]:
    """Each word takes the speaker whose turn covers its midpoint.

    Several covering turns (overlap regions): the one sharing the most time
    with the word, ties to the earlier turn. No covering turn: nearest turn.
    """
    return _assign(words, turns)[0]


def align_with_stats(words: list[dict], turns: list[dict], max_gap: float = 1.5) -> tuple[list[dict], dict]:
    tagged, fallback = _assign(words, turns)
    return build_utterances(tagged, max_gap=max_gap), {"words": len(words), "covered": len(words) - fallback,
                                                        "fallback": fallback}


def build_utterances(words: list[dict], max_gap: float = 1.5) -> list[dict]:
    """Merge consecutive same-speaker words; split on silence > max_gap seconds."""
    utterances: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            utterances.append({
                "speaker": current[0]["speaker"],
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(w["text"] for w in current),
                "words": list(current),
            })

    for w in words:
        if current and (w["speaker"] != current[-1]["speaker"] or w["start"] - current[-1]["end"] > max_gap):
            flush()
            current = []
        current.append(w)
    flush()
    return utterances


def align(words: list[dict], turns: list[dict], max_gap: float = 1.5) -> list[dict]:
    return build_utterances(assign_speakers(words, turns), max_gap=max_gap)


def rename_turns(turns: list[dict], mapping: dict) -> list[dict]:
    return [{**t, "speaker": mapping.get(t["speaker"], t["speaker"])} for t in turns]


def rename_utterances(utterances: list[dict], mapping: dict) -> list[dict]:
    out = []
    for u in utterances:
        words = [{**w, "speaker": mapping.get(w.get("speaker"), w.get("speaker"))} for w in u.get("words", [])]
        out.append({**u, "speaker": mapping.get(u["speaker"], u["speaker"]), "words": words})
    return out
