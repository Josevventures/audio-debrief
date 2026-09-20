"""Delivery metrics: talk time, answers, fillers, pace, pauses, overlaps, modes.

`me` throughout is the display name of the speaker being analyzed (the enrolled voiceprint)."""
from __future__ import annotations

import re
import string
from collections import Counter, defaultdict

from rapidfuzz import fuzz

from .pipeline import mmss

_PUNCT = string.punctuation + "’“”…"
DEFAULT_QUESTION_WORDS = [
    "who", "what", "when", "where", "why", "how", "which", "tell", "describe", "walk", "give",
    "explain", "talk", "share", "can", "could", "would", "do", "does", "did", "is", "are", "was",
    "were", "have", "has", "should",
]
NOT_MEASURED_NOTE = ("Interruptions and overlaps are not measurable with this capture method (far-side audio); "
                     "overlaps require --diarizer pyannote, which under-detects far-side speech.")


def _norm(tok: str) -> str:
    return tok.lower().strip(_PUNCT)


def _clean(text: str) -> str:
    return " ".join(_norm(t) for t in text.split() if _norm(t))


def is_question(text: str, question_words: list[str] | None = None) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return _norm(t.split()[0]) in (question_words or DEFAULT_QUESTION_WORDS)


def words_per_minute(n_words: int, duration_s: float) -> float:
    return round(n_words / duration_s * 60, 2) if duration_s > 0 else 0.0


# --- per speaker -------------------------------------------------------------

def speaker_stats(utterances: list[dict]) -> dict:
    durations: dict[str, list[float]] = defaultdict(list)
    for u in utterances:
        durations[u["speaker"]].append(u["end"] - u["start"])
    total = sum(sum(v) for v in durations.values()) or 1.0
    return {
        sp: {"talk_time_s": round(sum(d), 2), "share": round(sum(d) / total, 4), "turn_count": len(d),
             "mean_turn_s": round(sum(d) / len(d), 2), "max_turn_s": round(max(d), 2)}
        for sp, d in durations.items()
    }


# --- fillers -----------------------------------------------------------------

def count_fillers(words: list[dict], fillers: list[str]) -> dict:
    phrases = sorted(((f.lower().split(), f) for f in fillers), key=lambda p: len(p[0]), reverse=True)
    raw = [w["text"].lower() for w in words]
    norm = [_norm(t) for t in raw]
    counts: Counter = Counter({f: 0 for f in fillers})

    def match(ptok: str, i: int) -> bool:
        return raw[i].rstrip(".,!") == ptok if ptok.endswith("?") else norm[i] == ptok

    i = 0
    while i < len(raw):
        for ptoks, original in phrases:
            n = len(ptoks)
            if i + n <= len(raw) and all(match(ptoks[k], i + k) for k in range(n)):
                counts[original] += 1
                i += n
                break
        else:
            i += 1
    total = sum(counts.values())
    return {"total": total, "by_filler": dict(counts), "rate_per_100": round(total / len(raw) * 100, 2) if raw else 0.0}


# --- pauses ------------------------------------------------------------------

def _pause(preceding: list[dict], gap: float) -> dict:
    at = preceding[-1]["end"]
    return {"at": round(at, 2), "timestamp": mmss(at), "gap_s": round(gap, 2), "resumes_at": round(at + gap, 2),
            "preceding_words": " ".join(w["text"] for w in preceding[-5:])}


def find_pauses(utterances: list[dict], me: str, threshold: float) -> list[dict]:
    pauses = []
    for idx, u in enumerate(utterances):
        if u["speaker"] != me:
            continue
        ws = u["words"]
        for k in range(1, len(ws)):
            if ws[k]["start"] - ws[k - 1]["end"] > threshold:
                pauses.append(_pause(ws[:k], ws[k]["start"] - ws[k - 1]["end"]))
        if idx + 1 < len(utterances) and utterances[idx + 1]["speaker"] == me:
            gap = utterances[idx + 1]["start"] - u["end"]
            if gap > threshold and ws:
                pauses.append(_pause(ws, gap))
    return pauses


# --- overlaps / interruptions ---------------------------------------------------

def overlap_events(turns: list[dict], overlaps: list[dict], min_s: float) -> list[dict]:
    events = []
    for ov in overlaps:
        if ov["end"] - ov["start"] < min_s:
            continue
        involved = [t for t in turns if t["start"] < ov["end"] and t["end"] > ov["start"]]
        if len({t["speaker"] for t in involved}) < 2:
            continue
        by_start = sorted(involved, key=lambda t: t["start"])
        events.append({"start": ov["start"], "end": ov["end"], "timestamp": mmss(ov["start"]),
                       "duration_s": round(ov["end"] - ov["start"], 2),
                       "already_speaking": by_start[0]["speaker"], "interrupter": by_start[-1]["speaker"],
                       "yielded": min(involved, key=lambda t: t["end"])["speaker"]})
    return events


def overlap_counts(events: list[dict]) -> dict:
    counts: dict[str, dict] = {}
    for e in events:
        for sp in (e["already_speaking"], e["interrupter"], e["yielded"]):
            counts.setdefault(sp, {"interrupted": 0, "was_interrupted": 0, "yielded": 0})
        counts[e["interrupter"]]["interrupted"] += 1
        counts[e["already_speaking"]]["was_interrupted"] += 1
        counts[e["yielded"]]["yielded"] += 1
    return counts


# --- answers -------------------------------------------------------------------

def match_target(question: str, targets: dict) -> tuple[str, int] | None:
    for key, spec in (targets or {}).items():
        if re.search(spec["pattern"], question, re.IGNORECASE):
            return key, int(spec["seconds"])
    return None


def target_for(answer: dict, targets: dict) -> dict | None:
    m = match_target(answer["question"], targets)
    if m is None:
        return None
    return {"key": m[0], "target_s": m[1], "over": answer["duration_s"] > m[1]}


_FUNCTION_WORDS = set("""
that this with have what when your about tell from they them will would could there their been were just like
some then than into also very more much know think yeah okay well does doing done make made want need really
""".split())


def _content_tokens(text: str) -> set[str]:
    return {t for t in _clean(text).split() if len(t) >= 4 and t not in _FUNCTION_WORDS}


def qa_match(question: str, qa_questions: list[str]) -> tuple[float, str]:
    """Best (token_set_ratio, qa question). A qa question only counts when it shares at least one
    content token with the trigger: on a 40-word trigger token_set_ratio's fallback term sits near
    60 from shared letters alone, which tagged unrelated prompts as rehearsed on a real call."""
    best = (0.0, "")
    trigger_tokens = _content_tokens(question)
    for q in qa_questions:
        if not (_content_tokens(q) & trigger_tokens):
            continue
        score = fuzz.token_set_ratio(_clean(q), _clean(question))
        if score > best[0]:
            best = (score, q)
    return best


def tag_mode(question: str, qa_questions: list[str] | None, threshold: float, targets: dict | None = None) -> str:
    """rehearsed if the trigger matches a prepped category keyword or a qa_prep question (qa_match)."""
    if match_target(question, targets or {}):
        return "rehearsed"
    if not qa_questions:
        return "unknown"
    return "rehearsed" if qa_match(question, qa_questions)[0] >= threshold else "improvised"


def _is_backchannel(u: dict, max_words: int, qwords) -> bool:
    return len(u["words"]) <= max_words and not is_question(u["text"], qwords)


def find_answers(utterances: list[dict], me: str, cfg: dict) -> list[dict]:
    """An answer = contiguous speech by `me` (backchannels from others do not break it) of at least
    answer_min_s that follows someone else's speech. The trigger is the tail of that other-speaker run."""
    min_s, tail, bc = cfg["answer_min_s"], cfg["question_tail_words"], cfg["backchannel_max_words"]
    qwords = cfg.get("question_words") or DEFAULT_QUESTION_WORDS
    answers, i, n = [], 0, len(utterances)
    while i < n:
        if utterances[i]["speaker"] != me:
            i += 1
            continue
        j = i
        while True:  # extend over other speakers' backchannels
            k = j + 1
            while k < n and utterances[k]["speaker"] != me and _is_backchannel(utterances[k], bc, qwords):
                k += 1
            if k < n and utterances[k]["speaker"] == me:
                j = k
            else:
                break
        block = [u for u in utterances[i:j + 1] if u["speaker"] == me]
        run, p = [], i - 1  # contiguous other-speaker speech before the block (skipping the user's backchannels)
        while p >= 0 and (utterances[p]["speaker"] != me or _is_backchannel(utterances[p], bc, qwords)):
            if utterances[p]["speaker"] != me:
                run.insert(0, utterances[p])
            p -= 1
        start, end = block[0]["start"], block[-1]["end"]
        if run and end - start >= min_s:
            trigger = " ".join(" ".join(u["text"] for u in run).split()[-tail:])
            words = [w for u in block for w in u["words"]]
            answers.append({
                "start": start, "end": end, "timestamp": mmss(start), "duration_s": round(end - start, 2),
                "question": trigger, "is_question": is_question(trigger, qwords),
                "question_speaker": run[-1]["speaker"], "latency_s": round(start - run[-1]["end"], 2),
                "word_count": len(words), "wpm": words_per_minute(len(words), end - start),
                "fillers": count_fillers(words, cfg["fillers"]),
                "pauses": [], "mode": "unknown", "target": None, "prosody": None,
            })
        i = j + 1
    return answers


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def compare_modes(answers: list[dict]) -> dict:
    out = {}
    for mode in ("rehearsed", "improvised"):
        subset = [a for a in answers if a["mode"] == mode]
        out[mode] = {"count": len(subset), "mean_duration_s": _mean([a["duration_s"] for a in subset]),
                     "mean_wpm": _mean([a["wpm"] for a in subset]),
                     "mean_filler_rate": _mean([a["fillers"]["rate_per_100"] for a in subset]),
                     "mean_pause_count": _mean([len(a["pauses"]) for a in subset])}
    return out


def compute_metrics(utterances, turns, overlaps, me, qa_questions, cfg, overlaps_measured: bool = True) -> dict:
    answers = find_answers(utterances, me, cfg)
    pauses = find_pauses(utterances, me, cfg["pause_threshold_s"])
    for a in answers:
        a["pauses"] = [p for p in pauses if a["start"] <= p["at"] <= a["end"]]
        a["mode"] = tag_mode(a["question"], qa_questions, cfg["rehearsed_match_threshold"], cfg["answer_targets"])
        a["target"] = target_for(a, cfg["answer_targets"])
        score, q = qa_match(a["question"], qa_questions) if qa_questions else (0.0, "")
        a["qa_match"] = {"score": score, "question": q} if q else None
    stats = speaker_stats(utterances)
    fillers, pace = {}, {}
    for sp in stats:
        words = [w for u in utterances if u["speaker"] == sp for w in u["words"]]
        fillers[sp] = count_fillers(words, cfg["fillers"])
        pace[sp] = {"word_count": len(words), "wpm": words_per_minute(len(words), stats[sp]["talk_time_s"])}
    if overlaps_measured:
        events = overlap_events(turns, overlaps, cfg["overlap_min_s"])
        interruptions = {"status": "measured", "events": events, "counts": overlap_counts(events)}
    else:
        interruptions = {"status": "not_measured", "note": NOT_MEASURED_NOTE, "events": [], "counts": {}}
    return {"self": me, "speakers": stats, "answers": answers, "fillers": fillers, "pace": pace, "pauses": pauses,
            "interruptions": interruptions, "mode_comparison": compare_modes(answers),
            "qa_prep_available": bool(qa_questions)}
